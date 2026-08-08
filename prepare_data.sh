#!/usr/bin/env bash

set -euo pipefail

# Default OpenBHB preprocessing configuration.
# Override from the environment:
#   DATA_ROOT=/scratch/openbhb/quasiraw_3d OUTPUT_DIR=/scratch/openbhb/cache ./prepare_data.sh
# Or override for one run with CLI flags:
#   ./prepare_data.sh --target_shape 96 96 96 --clip_range 2.5
DATA_ROOT="${DATA_ROOT:-/home/fred/Projetos/Einstein/OpenBHB_Dataset/openbhb_train_sample/train/quasiraw_3d}"
METADATA="${METADATA:-/home/fred/Projetos/Einstein/OpenBHB_Dataset/openbhb_train_sample/train/quasiraw_3d/metadata.tsv}"
OUTPUT_DIR="${OUTPUT_DIR:-/home/fred/Projetos/Einstein/OpenBHB_Dataset/openbhb_train_sample/train/train_cache_64}"
TARGET_SHAPE_D="${TARGET_SHAPE_D:-64}"
TARGET_SHAPE_H="${TARGET_SHAPE_H:-64}"
TARGET_SHAPE_W="${TARGET_SHAPE_W:-64}"
NPY_SUFFIX="${NPY_SUFFIX:-_quasiraw_3d}"
CLIP_RANGE="${CLIP_RANGE:-3.0}"
SPLIT="${SPLIT:-}"
DRY_RUN=0

usage() {
    cat <<'EOF'
Usage: ./prepare_data.sh [options]

Options:
  --data_root PATH        Directory containing OpenBHB .npy volumes.
  --metadata PATH         Path to metadata.tsv or train.tsv.
  --output_dir PATH       Directory where preprocessed .pt files are written.
  --target_shape D H W    Target 3D shape. Default: 64 64 64.
  --npy_suffix SUFFIX     Suffix used in .npy filenames. Default: _quasiraw_3d.
  --clip_range FLOAT      Z-score clipping range. Default: 3.0.
  --split NAME            Optional split filter if metadata has a split column.
  --dry-run               Print the command without executing it.
  -h, --help              Show this help text.

Examples:
  ./prepare_data.sh
  ./prepare_data.sh --data_root data/openbhb/train/quasiraw_3d --target_shape 64 64 64
  DATA_ROOT=/scratch/openbhb METADATA=/scratch/openbhb/metadata.tsv ./prepare_data.sh --dry-run
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

while [[ $# -gt 0 ]]; do
    case "$1" in
        --data_root)
            require_value "$1" "${2-}"
            DATA_ROOT="$2"
            shift 2
            ;;
        --metadata)
            require_value "$1" "${2-}"
            METADATA="$2"
            shift 2
            ;;
        --output_dir)
            require_value "$1" "${2-}"
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --target_shape)
            require_value "$1" "${2-}"
            require_value "$1" "${3-}"
            require_value "$1" "${4-}"
            TARGET_SHAPE_D="$2"
            TARGET_SHAPE_H="$3"
            TARGET_SHAPE_W="$4"
            shift 4
            ;;
        --npy_suffix)
            require_value "$1" "${2-}"
            NPY_SUFFIX="$2"
            shift 2
            ;;
        --clip_range)
            require_value "$1" "${2-}"
            CLIP_RANGE="$2"
            shift 2
            ;;
        --split)
            require_value "$1" "${2-}"
            SPLIT="$2"
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

[[ -d "$DATA_ROOT" ]] || die "DATA_ROOT does not exist: $DATA_ROOT"
[[ -f "$METADATA" ]] || die "METADATA file does not exist: $METADATA"
is_positive_int "$TARGET_SHAPE_D" || die "Target depth must be a positive integer: $TARGET_SHAPE_D"
is_positive_int "$TARGET_SHAPE_H" || die "Target height must be a positive integer: $TARGET_SHAPE_H"
is_positive_int "$TARGET_SHAPE_W" || die "Target width must be a positive integer: $TARGET_SHAPE_W"

mkdir -p "$OUTPUT_DIR"

CMD=(
    python tools/prepare_openbhb.py
    --data_root "$DATA_ROOT"
    --metadata "$METADATA"
    --output_dir "$OUTPUT_DIR"
    --target_shape "$TARGET_SHAPE_D" "$TARGET_SHAPE_H" "$TARGET_SHAPE_W"
    --npy_suffix "$NPY_SUFFIX"
    --clip_range "$CLIP_RANGE"
)

if [[ -n "$SPLIT" ]]; then
    CMD+=(--split "$SPLIT")
fi

if [[ "$DRY_RUN" -eq 1 ]]; then
    printf 'Dry run command:\n'
    printf '  %q' "${CMD[@]}"
    printf '\n'
    exit 0
fi

echo "Preparing OpenBHB cache:"
printf '  %q' "${CMD[@]}"
printf '\n'

"${CMD[@]}"
