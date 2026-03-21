#!/bin/bash
# Chess Transformer - Project CLI
# Usage: ./cli.sh <command>

set -e

ENV_NAME="chess"
PYTHON_VERSION="3.12"

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

info()  { echo -e "${GREEN}[INFO]${NC} $1"; }
warn()  { echo -e "${YELLOW}[WARN]${NC} $1"; }
error() { echo -e "${RED}[ERROR]${NC} $1"; exit 1; }

case "$1" in
    setup)
        info "Creating conda environment '${ENV_NAME}' with Python ${PYTHON_VERSION}..."
        conda create -n "$ENV_NAME" python="$PYTHON_VERSION" -y

        info "Activating environment..."
        eval "$(conda shell.bash hook)" 2>/dev/null || source "$(conda info --base)/etc/profile.d/conda.sh"
        set +e; conda activate "$ENV_NAME"; set -e

        info "Installing PyTorch with CUDA 12.4 support..."
        pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

        info "Installing pip dependencies from requirements.txt..."
        pip install -r requirements.txt

        info "Setup complete! Run 'conda activate ${ENV_NAME}' to get started."
        ;;

    install)
        info "Installing pip dependencies from requirements.txt..."
        pip install -r requirements.txt
        info "Done!"
        ;;

    clean)
        warn "Removing conda environment '${ENV_NAME}'..."
        conda deactivate 2>/dev/null || true
        conda remove -n "$ENV_NAME" --all -y
        info "Environment removed."
        ;;

    check)
        info "Checking all imports..."
        python -c "
import torch
import chess
import huggingface_hub
import pyarrow
import tokenizers
import onnx
import onnxruntime
print('All imports OK!')
print(f'PyTorch: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA device: {torch.cuda.get_device_name(0)}')
"
        ;;

    data)
        info "Running full data pipeline: download -> process..."
        python src/data/extract_zst.py all
        ;;

    download)
        info "Downloading parquet files from HuggingFace..."
        python src/data/extract_zst.py download
        ;;

    process)
        info "Converting parquet to UCI (parallel)..."
        python src/data/extract_zst.py process ${2:+--workers $2}
        ;;

    *)
        echo "Usage: ./cli.sh <command> [options]"
        echo ""
        echo "Environment:"
        echo "  setup      Create conda env + install PyTorch (CUDA) + pip deps"
        echo "  install    Install/update pip dependencies only"
        echo "  clean      Remove the conda environment"
        echo "  check      Verify all packages import correctly"
        echo ""
        echo "Data Pipeline:"
        echo "  data       Run full pipeline: download -> process"
        echo "  download   Download parquet files from HuggingFace"
        echo "  process    Convert parquet to UCI format (parallel, optional: workers count)"
        echo ""
        echo "Examples:"
        echo "  ./cli.sh data          # run full pipeline"
        echo "  ./cli.sh process 64    # convert parquet to UCI using 64 workers"
        ;;
esac
