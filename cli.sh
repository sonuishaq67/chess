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
        info "Streaming download + process pipeline (parallel)..."
        python src/data/extract_zst.py all ${2:+--workers $2} ${3:+--limit $3} "${@:4}"
        ;;

    download)
        info "Downloading parquet files from HuggingFace..."
        python src/data/extract_zst.py download ${2:+--download-workers $2} ${3:+--limit $3} "${@:4}"
        ;;

    tokenize)
        info "Tokenizing UCI files to binary format..."
        python src/data/uci_tokenizer.py
        ;;

    build-packing)
        info "Pre-building chunk-packing arrays for distributed training..."
        python -m src.data.build_packing "${@:2}"
        ;;

    train)
        info "Starting training..."
        python -m src.training.train ${2:+--config $2} ${3:+--model-config $3} "${@:4}"
        ;;

    train-gaudi)
        info "Submitting 8-Gaudi DDP job to SLURM (class_gaudi QOS)..."
        sbatch train_gaudi.sbatch
        ;;

    deploy-install)
        info "Installing minimal inference/bot dependencies..."
        python -m venv venv 2>/dev/null || true
        source venv/bin/activate
        pip install --upgrade pip
        pip install -r requirements-deploy.txt
        info "Done! Activate with: source venv/bin/activate"
        ;;

    deploy-setup)
        info "Setting up systemd service for chess bot..."

        if [ -z "$2" ]; then
            error "Usage: ./cli.sh deploy-setup <email>"
        fi
        ALERT_EMAIL="$2"
        BOT_USER="$(whoami)"
        BOT_DIR="$(pwd)"

        # Write .env file
        if [ ! -f .env ]; then
            warn "Creating .env file — edit it to add your LICHESS_BOT_TOKEN"
            echo "LICHESS_BOT_TOKEN=" > .env
            echo "ALERT_EMAIL=${ALERT_EMAIL}" >> .env
        else
            # Ensure ALERT_EMAIL is set
            grep -q "^ALERT_EMAIL=" .env && \
                sed -i "s|^ALERT_EMAIL=.*|ALERT_EMAIL=${ALERT_EMAIL}|" .env || \
                echo "ALERT_EMAIL=${ALERT_EMAIL}" >> .env
        fi

        # Generate service file from template with correct paths
        sed -e "s|%i|${BOT_USER}|g" \
            -e "s|WorkingDirectory=.*|WorkingDirectory=${BOT_DIR}|" \
            -e "s|EnvironmentFile=.*|EnvironmentFile=-${BOT_DIR}/.env|" \
            -e "s|ExecStart=.*|ExecStart=${BOT_DIR}/venv/bin/python -m src.serving.bot --model models/model_opt.onnx --vocab dataset/vocab.json --strategy greedy --threads 4|" \
            deploy/chess-bot.service > /tmp/chess-bot.service

        sed -e "s|EnvironmentFile=.*|EnvironmentFile=-${BOT_DIR}/.env|" \
            deploy/chess-bot-notify@.service > /tmp/chess-bot-notify@.service

        sudo cp /tmp/chess-bot.service /etc/systemd/system/chess-bot.service
        sudo cp /tmp/chess-bot-notify@.service /etc/systemd/system/chess-bot-notify@.service
        sudo systemctl daemon-reload
        sudo systemctl enable chess-bot.service

        info "Systemd service installed. Commands:"
        info "  sudo systemctl start chess-bot    # start the bot"
        info "  sudo systemctl stop chess-bot     # stop the bot"
        info "  sudo systemctl status chess-bot   # check status"
        info "  journalctl -u chess-bot -f        # follow logs"
        info ""
        info "Edit ${BOT_DIR}/.env to set LICHESS_BOT_TOKEN before starting."
        info "Crash alerts will be sent to ${ALERT_EMAIL}"
        ;;

    bot-start)
        info "Starting chess bot..."
        sudo systemctl start chess-bot
        sudo systemctl status chess-bot --no-pager
        ;;

    bot-stop)
        info "Stopping chess bot..."
        sudo systemctl stop chess-bot
        ;;

    bot-status)
        sudo systemctl status chess-bot --no-pager
        echo ""
        info "Recent logs:"
        journalctl -u chess-bot -n 20 --no-pager
        ;;

    bot-logs)
        journalctl -u chess-bot -f
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
        echo "  data       Stream download+process in parallel (3 DL + 16 proc workers)"
        echo "  download   Download parquet files only (for HPC split workflows)"
        echo "  tokenize   Tokenize UCI text files into binary .bin/.idx format"
        echo "  build-packing  Pre-build chunk-packing arrays (run once before train-gaudi)"
        echo ""
        echo "Training:"
        echo "  train          Train the model (single GPU / local)"
        echo "  gaudi-install  Install project deps into gaudi-pytorch env (run once on Sol)"
        echo "  train-gaudi    Submit 8-Gaudi DDP job to SLURM (class_gaudi QOS)"
        echo ""
        echo "Deployment (EC2):"
        echo "  deploy-install             Install minimal bot deps into venv"
        echo "  deploy-setup <email>       Install systemd service + crash email alerts"
        echo "  bot-start                  Start the bot (survives SSH disconnect)"
        echo "  bot-stop                   Stop the bot"
        echo "  bot-status                 Show status + recent logs"
        echo "  bot-logs                   Follow live logs"
        echo ""
        echo "Examples:"
        echo "  ./cli.sh data              # stream download+process (3 DL, 16 proc)"
        echo "  ./cli.sh data 16           # use 16 processing workers"
        echo "  ./cli.sh data 8 2          # 8 workers, process only 2 files (test)"
        echo "  ./cli.sh data 1 1 --remote # remote httpfs (small batches only)"
        echo "  ./cli.sh data 16 '' --download-workers 5  # custom DL+proc workers"
        echo "  ./cli.sh download          # download only (process later)"
        echo "  ./cli.sh tokenize          # tokenize all UCI files to binary"
        echo "  ./cli.sh train             # train with default configs"
        echo "  ./cli.sh train configs/training.yml configs/model.yml  # custom configs"
        echo "  ./cli.sh train '' '' --resume checkpoints/epoch_3.pt  # resume training"
        echo ""
        echo "  # EC2 deployment:"
        echo "  ./cli.sh deploy-install                  # install deps"
        echo "  ./cli.sh deploy-setup you@email.com      # install service"
        echo "  # Edit .env to set LICHESS_BOT_TOKEN"
        echo "  ./cli.sh bot-start                       # start (background)"
        ;;
esac
