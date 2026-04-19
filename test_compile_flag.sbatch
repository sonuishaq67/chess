#!/bin/bash -l
#SBATCH -p gaudi
#SBATCH -q class_gaudi
#SBATCH -N 1
#SBATCH -G 1
#SBATCH -t 0:15:00
#SBATCH -c 16
#SBATCH --mem=32G
#SBATCH -o test_compile_flag.%j.out
#SBATCH -e test_compile_flag.%j.err

# Test: does PT_HPU_USE_JIT_FORK=0 make torch.compile(hpu_backend) work on a
# graph complex enough to trigger pass_compile_clusters_jit_fork_version?
# The earlier test_diffusion_env.sbatch used a 3-layer Linear that was too
# simple to exercise the cluster compiler — it passed misleadingly.
# This uses 2 real TransformerBlocks from src/model/transformer.py so the
# compile path matches what the training job will hit.

source /home/ishaik4/miniconda3/etc/profile.d/conda.sh
conda activate /packages/envs/gaudi-pytorch-diffusion-1.22.0.740

echo "Python: $(which python)"
python -c "import torch; print(f'torch: {torch.__version__}')"

export PYTHONUNBUFFERED=1
export PT_HPU_LAZY_MODE=1
export PT_HPU_USE_JIT_FORK=0

cd ~/chess
python - <<'EOF'
import sys, time
import torch
import habana_frameworks.torch.core as htcore
from src.model.transformer import Transformer, TransformerConfig

device = torch.device("hpu")

# Small but realistic: 2 TransformerBlocks, full attention + FFN + LayerNorm +
# RoPE + causal mask. Enough to trigger the cluster compiler path.
cfg = TransformerConfig(d_model=256, n_layers=2, n_heads=4, ffn_dim=1024,
                        vocab_size=1972, max_seq_len=256, dropout=0.1)
model = Transformer(cfg).to(device)

x = torch.randint(0, cfg.vocab_size, (4, 256), device=device)

# 1) Eager forward sanity
y = model(x)
htcore.mark_step(); torch.hpu.synchronize()
print(f"EAGER OK: y.shape={tuple(y.shape)}", flush=True)

# 2) Compiled forward
try:
    mc = torch.compile(model, backend="hpu_backend")
    t0 = time.time()
    y2 = mc(x)
    htcore.mark_step(); torch.hpu.synchronize()
    print(f"COMPILE FWD OK in {time.time()-t0:.1f}s, y2.shape={tuple(y2.shape)}", flush=True)

    # 3) Compiled backward
    target = torch.randint(0, cfg.vocab_size, (4, 256), device=device)
    loss = torch.nn.functional.cross_entropy(y2.reshape(-1, cfg.vocab_size), target.reshape(-1))
    t0 = time.time()
    loss.backward()
    htcore.mark_step(); torch.hpu.synchronize()
    print(f"COMPILE BWD OK in {time.time()-t0:.1f}s, loss={loss.item():.4f}", flush=True)
    print("PASS: PT_HPU_USE_JIT_FORK=0 fixes it", flush=True)
except Exception as e:
    print(f"FAIL: {type(e).__name__}: {e}", flush=True)
    sys.exit(2)
EOF
