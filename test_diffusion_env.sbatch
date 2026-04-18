#!/bin/bash -l
#SBATCH -p gaudi
#SBATCH -q class_gaudi
#SBATCH -N 1
#SBATCH -G 1
#SBATCH -t 0:15:00
#SBATCH -c 16
#SBATCH --mem=32G
#SBATCH -o test_diffusion.%j.out
#SBATCH -e test_diffusion.%j.err

# Purpose: verify whether habana-torch-plugin 1.22.0.740 (in the diffusion env)
# supports torch.compile(backend='hpu_backend') on a Gaudi compute node. The
# 1.23.0.695 version in pytorch-2.9.0-gaudi fails with an undefined C++ symbol.
# If this test prints "DIFFUSION OK", we retarget training at this env.

source /home/ishaik4/miniconda3/etc/profile.d/conda.sh
conda activate /packages/envs/gaudi-pytorch-diffusion-1.22.0.740

echo "Python: $(which python)"
python -c "import torch; print(f'torch: {torch.__version__}')" || { echo "torch import failed"; exit 1; }
python -c "import habana_frameworks.torch as htorch; print(f'habana_frameworks version OK')" || { echo "habana import failed"; exit 1; }

export PYTHONUNBUFFERED=1
export PT_HPU_LAZY_MODE=1

python - <<'EOF'
import sys
import torch
import habana_frameworks.torch.core as htcore

device = torch.device("hpu")

# 1) Plain forward on HPU (sanity: env works at all)
m = torch.nn.Sequential(
    torch.nn.Linear(128, 256),
    torch.nn.GELU(),
    torch.nn.Linear(256, 128),
).to(device)
x = torch.randn(4, 128, device=device)
y = m(x)
htcore.mark_step()
torch.hpu.synchronize()
print(f"EAGER OK: y.shape={tuple(y.shape)} y[0,0]={y[0,0].item():.4f}", flush=True)

# 2) torch.compile with hpu_backend (the actual question)
try:
    mc = torch.compile(m, backend="hpu_backend")
    y2 = mc(x)           # triggers Dynamo trace + backend compile
    htcore.mark_step()
    torch.hpu.synchronize()
    print(f"DIFFUSION OK: compiled forward works, y2[0,0]={y2[0,0].item():.4f}", flush=True)

    # 3) Back-prop through compiled model (training-relevant)
    target = torch.randn_like(y2)
    loss = ((y2 - target) ** 2).mean()
    loss.backward()
    htcore.mark_step()
    torch.hpu.synchronize()
    print(f"DIFFUSION BWD OK: backward through compiled model works, loss={loss.item():.4f}", flush=True)
except Exception as e:
    print(f"DIFFUSION FAIL: {type(e).__name__}: {e}", flush=True)
    sys.exit(2)
EOF
