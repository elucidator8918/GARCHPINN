#!/usr/bin/env bash
# Set up GARCHPINN on a Linux machine with NVIDIA GPU(s).
#   CUDA=cu124 PY=python3.12 bash scripts/setup_linux.sh
# CUDA picks the PyTorch wheel index (cu118, cu121, cu124, cu126, ... or cpu); match it to
# the driver (`nvidia-smi` shows the highest CUDA version the driver supports).
set -euo pipefail
CUDA="${CUDA:-cu124}"
PY="${PY:-python3}"
VENV="${VENV:-.venv}"
cd "$(dirname "$0")/.."

"$PY" -m venv "$VENV"
source "$VENV/bin/activate"
python -m pip install --upgrade pip
python -m pip install torch --index-url "https://download.pytorch.org/whl/${CUDA}"
grep -v '^torch' requirements.txt | python -m pip install -r /dev/stdin

python - <<'PY'
import os, torch
print("torch", torch.__version__, "| CUDA build", torch.version.cuda,
      "| CUDA_VISIBLE_DEVICES =", os.environ.get("CUDA_VISIBLE_DEVICES"))
print("cuda available:", torch.cuda.is_available(), "| GPUs:", torch.cuda.device_count(),
      "| CPU cores:", os.cpu_count())
for i in range(torch.cuda.device_count()):
    p = torch.cuda.get_device_properties(i)
    x = torch.randn(1000, 1000, dtype=torch.float64, device=f"cuda:{i}")
    ok = torch.allclose((x @ x).cpu(), x.cpu() @ x.cpu())
    print(f"  cuda:{i} {p.name} {p.total_memory / 2**30:.1f} GiB, float64 matmul ok={ok}")
PY

python -m pytest tests -q   # includes the CUDA-only tests when a GPU is present
