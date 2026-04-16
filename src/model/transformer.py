import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from .embeddings import Embeddings


@dataclass
class TransformerConfig:
    d_model: int = 128
    n_layers: int = 4
    n_heads: int = 4
    ffn_dim: int = 512
    vocab_size: int = 1972
    max_seq_len: int = 256
    dropout: float = 0.1

    @property
    def head_dim(self) -> int:
        return self.d_model // self.n_heads


def precompute_rope_freqs(
    head_dim: int, max_seq_len: int, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    theta = 1.0 / (
        10000.0 ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim)
    )
    positions = torch.arange(max_seq_len, device=device).float()
    angles = torch.outer(positions, theta)  # [max_seq_len, head_dim/2]
    return torch.cos(angles), torch.sin(angles)


def apply_rope(
    x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> torch.Tensor:
    x_f = x.float()
    x_even = x_f[..., ::2]
    x_odd = x_f[..., 1::2]
    cos = cos.unsqueeze(0).unsqueeze(0)
    sin = sin.unsqueeze(0).unsqueeze(0)
    out_even = x_even * cos - x_odd * sin
    out_odd = x_even * sin + x_odd * cos
    return torch.stack([out_even, out_odd], dim=-1).flatten(-2).type_as(x)


class MultiHeadAttention(nn.Module):
    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.n_heads = config.n_heads
        self.head_dim = config.head_dim

        self.w_q = nn.Linear(config.d_model, config.d_model, bias=False)
        self.w_k = nn.Linear(config.d_model, config.d_model, bias=False)
        self.w_v = nn.Linear(config.d_model, config.d_model, bias=False)
        self.w_o = nn.Linear(config.d_model, config.d_model, bias=False)

        self.attn_dropout = nn.Dropout(config.dropout)

    def forward(
        self, x: torch.Tensor, rope_cos: torch.Tensor, rope_sin: torch.Tensor
    ) -> torch.Tensor:
        batch, seq_len, _ = x.shape

        q = self.w_q(x)
        k = self.w_k(x)
        v = self.w_v(x)

        q = q.view(batch, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch, seq_len, self.n_heads, self.head_dim).transpose(1, 2)

        q = apply_rope(q, rope_cos, rope_sin)
        k = apply_rope(k, rope_cos, rope_sin)

        out = F.scaled_dot_product_attention(
            q,
            k,
            v,
            dropout_p=self.attn_dropout.p if self.training else 0.0,
            is_causal=True,
        )

        out = out.transpose(1, 2).contiguous().view(batch, seq_len, -1)

        return self.w_o(out)


class FeedForward(nn.Module):
    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.w1 = nn.Linear(config.d_model, config.ffn_dim, bias=False)
        self.w2 = nn.Linear(config.ffn_dim, config.d_model, bias=False)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.w2(F.silu(self.w1(x))))


class TransformerBlock(nn.Module):
    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.attn_norm = nn.LayerNorm(config.d_model)
        self.attn = MultiHeadAttention(config)
        self.ffn_norm = nn.LayerNorm(config.d_model)
        self.ffn = FeedForward(config)

    def forward(
        self, x: torch.Tensor, rope_cos: torch.Tensor, rope_sin: torch.Tensor
    ) -> torch.Tensor:
        x = x + self.attn(self.attn_norm(x), rope_cos, rope_sin)
        x = x + self.ffn(self.ffn_norm(x))
        return x


class Transformer(nn.Module):
    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.config = config

        self.embedding = Embeddings(config.vocab_size, config.d_model)
        self.embed_dropout = nn.Dropout(config.dropout)

        self.blocks = nn.ModuleList(
            [TransformerBlock(config) for _ in range(config.n_layers)]
        )

        self.norm = nn.LayerNorm(config.d_model)
        self.output = nn.Linear(config.d_model, config.vocab_size, bias=False)

        cos, sin = precompute_rope_freqs(
            config.head_dim, config.max_seq_len, device=torch.device("cpu")
        )
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)

        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        seq_len = tokens.size(1)
        assert seq_len <= self.config.max_seq_len, (
            f"Sequence length {seq_len} exceeds max_seq_len {self.config.max_seq_len}"
        )

        x = self.embedding(tokens)  # [batch, seq, d_model]
        x = self.embed_dropout(x)
        rope_cos = self.rope_cos[:seq_len]
        rope_sin = self.rope_sin[:seq_len]
        for block in self.blocks:
            x = block(x, rope_cos, rope_sin)

        x = self.norm(x)
        logits = self.output(x)  # [batch, seq, vocab_size]

        return logits

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())
