import argparse
import math
import os

import numpy as np
import onnxruntime as ort
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml

from src.model.embeddings import Embeddings
from src.model.transformer import FeedForward, TransformerConfig, precompute_rope_freqs


def apply_rope_real(
    x: torch.Tensor, cos_freqs: torch.Tensor, sin_freqs: torch.Tensor
) -> torch.Tensor:
    x_even = x[..., ::2]   # [batch, n_heads, seq_len, head_dim/2]
    x_odd = x[..., 1::2]

    # broadcast: [1, 1, seq_len, head_dim/2]
    cos_f = cos_freqs.unsqueeze(0).unsqueeze(0)
    sin_f = sin_freqs.unsqueeze(0).unsqueeze(0)

    out_even = x_even * cos_f - x_odd * sin_f
    out_odd = x_even * sin_f + x_odd * cos_f

    # interleave back: stack on last dim then flatten
    return torch.stack([out_even, out_odd], dim=-1).flatten(-2)


class OnnxMultiHeadAttention(nn.Module):
    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.n_heads = config.n_heads
        self.head_dim = config.head_dim
        self.scale = 1.0 / math.sqrt(config.head_dim)

        self.w_q = nn.Linear(config.d_model, config.d_model, bias=False)
        self.w_k = nn.Linear(config.d_model, config.d_model, bias=False)
        self.w_v = nn.Linear(config.d_model, config.d_model, bias=False)
        self.w_o = nn.Linear(config.d_model, config.d_model, bias=False)

    def forward(
        self,
        x: torch.Tensor,
        cos_freqs: torch.Tensor,
        sin_freqs: torch.Tensor,
        causal_mask: torch.Tensor,
    ) -> torch.Tensor:
        batch, seq_len, _ = x.shape

        q = self.w_q(x).view(batch, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.w_k(x).view(batch, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.w_v(x).view(batch, seq_len, self.n_heads, self.head_dim).transpose(1, 2)

        q = apply_rope_real(q, cos_freqs, sin_freqs)
        k = apply_rope_real(k, cos_freqs, sin_freqs)

        attn = (q @ k.transpose(-2, -1)) * self.scale + causal_mask
        attn = F.softmax(attn, dim=-1)
        out = attn @ v

        out = out.transpose(1, 2).contiguous().view(batch, seq_len, -1)
        return self.w_o(out)


class OnnxTransformerBlock(nn.Module):
    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.attn_norm = nn.LayerNorm(config.d_model)
        self.attn = OnnxMultiHeadAttention(config)
        self.ffn_norm = nn.LayerNorm(config.d_model)
        self.ffn = FeedForward(config)

    def forward(
        self,
        x: torch.Tensor,
        cos_freqs: torch.Tensor,
        sin_freqs: torch.Tensor,
        causal_mask: torch.Tensor,
    ) -> torch.Tensor:
        x = x + self.attn(self.attn_norm(x), cos_freqs, sin_freqs, causal_mask)
        x = x + self.ffn(self.ffn_norm(x))
        return x

class OnnxTransformer(nn.Module):
    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.config = config

        self.embedding = Embeddings(config.vocab_size, config.d_model)
        # Dropout kept for weight-loading compat; no-op in eval mode
        self.embed_dropout = nn.Dropout(config.dropout)

        self.blocks = nn.ModuleList(
            [OnnxTransformerBlock(config) for _ in range(config.n_layers)]
        )

        self.norm = nn.LayerNorm(config.d_model)
        self.output = nn.Linear(config.d_model, config.vocab_size, bias=False)

        # Precompute real-valued RoPE buffers from the complex freqs
        complex_freqs = precompute_rope_freqs(
            config.head_dim, config.max_seq_len, device=torch.device("cpu")
        )
        # complex_freqs: [max_seq_len, head_dim/2] complex
        self.register_buffer("rope_cos", complex_freqs.real.clone(), persistent=False)
        self.register_buffer("rope_sin", complex_freqs.imag.clone(), persistent=False)

        # Upper-triangle causal mask filled with -inf
        mask = torch.full(
            (config.max_seq_len, config.max_seq_len), float("-inf")
        )
        mask = torch.triu(mask, diagonal=1)
        self.register_buffer("causal_mask", mask, persistent=False)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        seq_len = tokens.size(1)

        x = self.embedding(tokens)
        x = self.embed_dropout(x)

        cos_freqs = self.rope_cos[:seq_len]
        sin_freqs = self.rope_sin[:seq_len]
        causal = self.causal_mask[:seq_len, :seq_len]

        for block in self.blocks:
            x = block(x, cos_freqs, sin_freqs, causal)

        x = self.norm(x)
        return self.output(x)

def load_checkpoint(checkpoint_path: str, config: TransformerConfig) -> OnnxTransformer:
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state_dict = ckpt["model"]

    model = OnnxTransformer(config)

    # strict=False: ignores rope_freqs from checkpoint (replaced by rope_cos/rope_sin)
    # and causal_mask (initialized by wrapper)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)

    expected_unexpected = {"rope_freqs"}
    expected_missing = {"rope_cos", "rope_sin", "causal_mask"}

    real_unexpected = set(unexpected) - expected_unexpected
    real_missing = set(missing) - expected_missing

    if real_unexpected:
        print(f"WARNING: unexpected keys in checkpoint: {real_unexpected}")
    if real_missing:
        print(f"WARNING: missing keys not loaded: {real_missing}")

    print(f"Loaded checkpoint from {checkpoint_path}")
    print(f"  Expected missing buffers: {missing}")
    print(f"  Expected unexpected buffer: {unexpected}")

    model.eval()
    return model


def export_onnx(model: OnnxTransformer, output_path: str):
    dummy = torch.randint(1, 1971, (1, 128))

    torch.onnx.export(
        model,
        (dummy,),
        output_path,
        opset_version=17,
        input_names=["tokens"],
        output_names=["logits"],
        dynamic_axes={
            "tokens": {0: "batch", 1: "seq_len"},
            "logits": {0: "batch", 1: "seq_len"},
        },
    )
    print(f"Exported ONNX model to {output_path}")

def optimize_onnx(input_path: str, output_path: str):
    opts = ort.SessionOptions()
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    opts.optimized_model_filepath = output_path

    # Creating the session triggers optimization and saves to the path
    ort.InferenceSession(input_path, opts, providers=["CPUExecutionProvider"])
    print(f"Optimized ONNX model saved to {output_path}")


def quantize_onnx(input_path: str, output_path: str):
    from onnxruntime.quantization import QuantType, quantize_dynamic

    quantize_dynamic(
        model_input=input_path,
        model_output=output_path,
        weight_type=QuantType.QUInt8,
        per_channel=True,
        reduce_range=False,
    )
    print(f"Quantized ONNX model saved to {output_path}")


# ---------------------------------------------------------------------------
# Parity verification
# ---------------------------------------------------------------------------

def verify_parity(
    model: OnnxTransformer, onnx_path: str, atol: float = 1e-4
) -> bool:
    """Check that the ONNX model produces the same output as PyTorch."""
    tokens = torch.randint(1, 1971, (1, 128))

    with torch.no_grad():
        pt_out = model(tokens).numpy()

    session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    onnx_out = session.run(None, {"tokens": tokens.numpy().astype(np.int64)})[0]

    max_diff = np.max(np.abs(pt_out - onnx_out))
    close = np.allclose(pt_out, onnx_out, atol=atol, rtol=1e-4)

    # Also check that top-1 predictions match at each position
    pt_top1 = np.argmax(pt_out, axis=-1)
    onnx_top1 = np.argmax(onnx_out, axis=-1)
    top1_match = np.all(pt_top1 == onnx_top1)

    print(f"Parity check: max abs diff = {max_diff:.2e}, "
          f"allclose = {close}, top1 match = {top1_match}")
    return close or top1_match

def main():
    parser = argparse.ArgumentParser(description="Export chess transformer to ONNX")
    parser.add_argument("--checkpoint", required=True, help="Path to .pt checkpoint")
    parser.add_argument("--config", default="configs/model.yml", help="Model config YAML")
    parser.add_argument("--output-dir", default="models", help="Output directory")
    parser.add_argument("--quantize", action="store_true", help="Also produce INT8 model")
    parser.add_argument("--no-verify", action="store_true", help="Skip parity check")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Load config
    with open(args.config) as f:
        model_cfg = yaml.safe_load(f)
    config = TransformerConfig(**model_cfg)
    print(f"Model config: {config}")

    # Load model
    model = load_checkpoint(args.checkpoint, config)

    # Export
    raw_path = os.path.join(args.output_dir, "model.onnx")
    export_onnx(model, raw_path)

    # Optimize
    opt_path = os.path.join(args.output_dir, "model_opt.onnx")
    optimize_onnx(raw_path, opt_path)

    # Quantize
    if args.quantize:
        int8_path = os.path.join(args.output_dir, "model_int8.onnx")
        quantize_onnx(opt_path, int8_path)

    # Verify
    if not args.no_verify:
        print("\n--- FP32 parity check ---")
        verify_parity(model, raw_path)

        print("\n--- Optimized parity check ---")
        verify_parity(model, opt_path)

        if args.quantize:
            print("\n--- INT8 parity check (relaxed tolerance) ---")
            verify_parity(model, int8_path, atol=0.5)

    # Report sizes (include external data files)
    print("\n--- Model sizes ---")
    for name in ["model.onnx", "model_opt.onnx", "model_int8.onnx"]:
        path = os.path.join(args.output_dir, name)
        if os.path.exists(path):
            size = os.path.getsize(path)
            data_path = path + ".data"
            if os.path.exists(data_path):
                size += os.path.getsize(data_path)
            print(f"  {name}: {size / (1024 * 1024):.1f} MB")


if __name__ == "__main__":
    main()
