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
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-}"
RESUME_FROM="${RESUME_FROM:-}"
SEED="${SEED:-}"
DRY_RUN=0
NO_VALIDATE=0
DETERMINISTIC=0

usage() {
    cat <<'EOF'
Usage: ./train.sh [options]

Options:
  --data_dir PATH              Dataset directory verified before launch.
  --output_dir PATH            Directory for checkpoints/logs.
  --batch_size N               Exported as BATCH_SIZE. Default: 32.
  --learning_rate LR           Exported as LEARNING_RATE. Default: 1e-4.
  --epochs N                   Exported as EPOCHS. Default: 100.
  --gpus N                     Number of GPUs to expose when CUDA_VISIBLE_DEVICES is unset.
  --cuda_visible_devices LIST  Explicit CUDA_VISIBLE_DEVICES, e.g. 0,1,2,3.
  --model_config PATH          Training config passed as positional train.py argument.
  --resume_from PATH           Checkpoint path passed to train.py.
  --seed N                     Random seed passed to train.py.
  --no_validate                Disable validation.
  --deterministic              Enable deterministic CUDNN behavior.
  --dry-run                    Print the final command without executing it.
  -h, --help                   Show this help text.

Examples:
  ./train.sh --batch_size 64 --gpus 4
  DATA_DIR=/scratch/openbhb OUTPUT_DIR=/scratch/runs ./train.sh --dry-run
  CUDA_VISIBLE_DEVICES=2,3 ./train.sh --gpus 2

Notes:
  This repository's train.py accepts: config, --work-dir, --resume-from,
  --no-validate, --gpu-ids, --seed, and --deterministic. DATA_DIR,
  BATCH_SIZE, LEARNING_RATE, EPOCHS, and GPUS are exported for configs or
  wrappers that read environment variables; unsupported flags are not passed
  directly to train.py.
EOF
}

die() {
    echo "ERROR: $*" >&2
    exit 1
}

require_value() {
    local flag="$1"
    local value="${2-}"
    if [[ -z "$value" || "$value" == --* ]]; then
        die "Missing value for $flag"
    fi
}

is_positive_int() {
    [[ "$1" =~ ^[1-9][0-9]*$ ]]
}

make_cuda_visible_devices() {
    local count="$1"
    local ids=()
    local i

    for ((i = 0; i < count; i++)); do
        ids+=("$i")
    done

    local IFS=,
    echo "${ids[*]}"
}

# Standard long-option parser. Each flag updates the variable used to build
# the final command, so exported env vars remain useful but are easy to override.
while [[ $# -gt 0 ]]; do
    case "$1" in
        --data_dir)
            require_value "$1" "${2-}"
            DATA_DIR="$2"
            shift 2
            ;;
        --output_dir)
            require_value "$1" "${2-}"
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --batch_size)
            require_value "$1" "${2-}"
            BATCH_SIZE="$2"
            shift 2
            ;;
        --learning_rate)
            require_value "$1" "${2-}"
            LEARNING_RATE="$2"
            shift 2
            ;;
        --epochs)
            require_value "$1" "${2-}"
            EPOCHS="$2"
            shift 2
            ;;
        --gpus)
            require_value "$1" "${2-}"
            GPUS="$2"
            shift 2
            ;;
        --cuda_visible_devices)
            require_value "$1" "${2-}"
            CUDA_VISIBLE_DEVICES="$2"
            shift 2
            ;;
        --model_config)
            require_value "$1" "${2-}"
            MODEL_CONFIG="$2"
            shift 2
            ;;
        --resume_from)
            require_value "$1" "${2-}"
            RESUME_FROM="$2"
            shift 2
            ;;
        --seed)
            require_value "$1" "${2-}"
            SEED="$2"
            shift 2
            ;;
        --no_validate)
            NO_VALIDATE=1
            shift
            ;;
        --deterministic)
            DETERMINISTIC=1
            shift
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

is_positive_int "$BATCH_SIZE" || die "BATCH_SIZE must be a positive integer: $BATCH_SIZE"
is_positive_int "$EPOCHS" || die "EPOCHS must be a positive integer: $EPOCHS"
is_positive_int "$GPUS" || die "GPUS must be a positive integer: $GPUS"

[[ -d "$DATA_DIR" ]] || die "DATA_DIR does not exist: $DATA_DIR"
[[ -f "$MODEL_CONFIG" ]] || die "MODEL_CONFIG does not exist: $MODEL_CONFIG"
mkdir -p "$OUTPUT_DIR"

if [[ -z "$CUDA_VISIBLE_DEVICES" ]]; then
    CUDA_VISIBLE_DEVICES="$(make_cuda_visible_devices "$GPUS")"
fi

export DATA_DIR OUTPUT_DIR BATCH_SIZE LEARNING_RATE EPOCHS GPUS CUDA_VISIBLE_DEVICES

CMD=(
    python train.py
    "$MODEL_CONFIG"
    --work-dir "$OUTPUT_DIR"
)

if [[ -n "$RESUME_FROM" ]]; then
    [[ -f "$RESUME_FROM" ]] || die "RESUME_FROM does not exist: $RESUME_FROM"
    CMD+=(--resume-from "$RESUME_FROM")
fi

if [[ "$NO_VALIDATE" -eq 1 ]]; then
    CMD+=(--no-validate)
fi

if [[ -n "$SEED" ]]; then
    is_positive_int "$SEED" || die "SEED must be a positive integer: $SEED"
    CMD+=(--seed "$SEED")
fi

if [[ "$DETERMINISTIC" -eq 1 ]]; then
    CMD+=(--deterministic)
fi

print_command() {
    printf 'CUDA_VISIBLE_DEVICES=%q' "$CUDA_VISIBLE_DEVICES"
    printf ' %q' "${CMD[@]}"
    printf '\n'
}

if [[ "$DRY_RUN" -eq 1 ]]; then
    printf 'Dry run command:\n  '
    print_command
    exit 0
fi

echo "Launching training:"
printf '  '
print_command

"${CMD[@]}"
