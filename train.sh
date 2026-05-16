#!/usr/bin/env bash

set -euo pipefail

# Default configuration.
# Override these from the shell or SLURM, for example:
#   DATA_DIR=/datasets/openbhb GPUS=4 ./train.sh
# CLI flags below override both these defaults and exported environment values.
DATA_DIR="${DATA_DIR:-data/openbhb}"
OUTPUT_DIR="${OUTPUT_DIR:-work_dirs}"
BATCH_SIZE="${BATCH_SIZE:-32}"
LEARNING_RATE="${LEARNING_RATE:-1e-4}"
EPOCHS="${EPOCHS:-100}"
GPUS="${GPUS:-1}"
MODEL_CONFIG="${MODEL_CONFIG:-configs/gmflow3d_openbhb_k4.py}"
DRY_RUN=0

usage() {
    cat <<'EOF'
Usage: ./train.sh [options]

Options:
  --data_dir PATH        Dataset directory.
  --output_dir PATH      Directory for checkpoints/logs.
  --batch_size N         Batch size. Default: 32.
  --learning_rate LR     Learning rate. Default: 1e-4.
  --epochs N             Number of epochs. Default: 100.
  --gpus N               Number of GPUs. Default: 1.
  --model_config PATH    Path to YAML/JSON/Python config.
  --dry-run              Print the final command without executing it.
  -h, --help             Show this help text.

Examples:
  ./train.sh --batch_size 64 --gpus 4
  DATA_DIR=/scratch/openbhb OUTPUT_DIR=/scratch/runs ./train.sh --dry-run
EOF
}

# Standard long-option parser. Each flag updates the variable used to build
# the final command, so exported env vars remain useful but are easy to override.
while [[ $# -gt 0 ]]; do
    case "$1" in
        --data_dir)
            DATA_DIR="${2:?Missing value for --data_dir}"
            shift 2
            ;;
        --output_dir)
            OUTPUT_DIR="${2:?Missing value for --output_dir}"
            shift 2
            ;;
        --batch_size)
            BATCH_SIZE="${2:?Missing value for --batch_size}"
            shift 2
            ;;
        --learning_rate)
            LEARNING_RATE="${2:?Missing value for --learning_rate}"
            shift 2
            ;;
        --epochs)
            EPOCHS="${2:?Missing value for --epochs}"
            shift 2
            ;;
        --gpus)
            GPUS="${2:?Missing value for --gpus}"
            shift 2
            ;;
        --model_config)
            MODEL_CONFIG="${2:?Missing value for --model_config}"
            shift 2
            ;;
        --dry-run)
            DRY_RUN=1
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "Unknown option: $1" >&2
            usage >&2
            exit 2
            ;;
    esac
done

if [[ ! -d "$DATA_DIR" ]]; then
    echo "ERROR: DATA_DIR does not exist: $DATA_DIR" >&2
    echo "Set it with DATA_DIR=/path/to/data ./train.sh or --data_dir /path/to/data" >&2
    exit 1
fi

mkdir -p "$OUTPUT_DIR"

CMD=(
    python train.py
    --config "$MODEL_CONFIG"
    --data_dir "$DATA_DIR"
    --output_dir "$OUTPUT_DIR"
    --batch_size "$BATCH_SIZE"
    --learning_rate "$LEARNING_RATE"
    --epochs "$EPOCHS"
    --gpus "$GPUS"
)

if [[ "$DRY_RUN" -eq 1 ]]; then
    printf 'Dry run command:\n'
    printf '  %q' "${CMD[@]}"
    printf '\n'
    exit 0
fi

echo "Launching training:"
printf '  %q' "${CMD[@]}"
printf '\n'

"${CMD[@]}"
