#!/usr/bin/env bash

set -euo pipefail

# Train a controlled GMFlow3D architecture-ablation suite. Every run starts
# from the same historical checkpoint and loads only name/shape-compatible
# model and EMA tensors through --resume_weights_only.

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

PYTHON_BIN="${PYTHON_BIN:-/home/fred/anaconda3/envs/gmflow/bin/python}"
TRAINER="${TRAINER:-tools/train_standalone.py}"
COMPARATOR="${COMPARATOR:-tools/compare_architecture_runs.py}"

OPENBHB_ROOT="${OPENBHB_ROOT:-/media/fred/FRED5TB/Einstein/Open_BHB_processado}"
DATA_ROOT="${DATA_ROOT:-${OPENBHB_ROOT}/train/quasiraw_3d}"
METADATA="${METADATA:-${OPENBHB_ROOT}/train.tsv}"
CACHE_DIR="${CACHE_DIR:-}"
CHECKPOINT="${CHECKPOINT:-/media/fred/FRED5TB/work_dirs/gmflow3d_openbhb_k4_gpu5_20260801_101648/checkpoints/latest.pt}"

RUN_TAG="${RUN_TAG:-$(date +%Y%m%d_%H%M%S)}"
OUTPUT_ROOT="${OUTPUT_ROOT:-work_dirs/architecture_ablation_${RUN_TAG}}"
ARCHITECTURES="${ARCHITECTURES:-all}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

TOTAL_ITERS="${TOTAL_ITERS:-5000}"
WARMUP_ITERS="${WARMUP_ITERS:-500}"
BATCH_SIZE="${BATCH_SIZE:-2}"
GRAD_ACCUM="${GRAD_ACCUM:-4}"
LEARNING_RATE="${LEARNING_RATE:-1e-4}"
NUM_WORKERS="${NUM_WORKERS:-4}"
LOG_INTERVAL="${LOG_INTERVAL:-10}"
SAVE_INTERVAL="${SAVE_INTERVAL:-5000}"
SAMPLE_INTERVAL="${SAMPLE_INTERVAL:-1000}"
SEED="${SEED:-42}"

VOLUME_SIZE="${VOLUME_SIZE:-64}"
PATCH_SIZE="${PATCH_SIZE:-2}"
VOXEL_PATCH_SIZE="${VOXEL_PATCH_SIZE:-4}"
NUM_GAUSSIANS="${NUM_GAUSSIANS:-4}"
NUM_HEADS="${NUM_HEADS:-12}"
HEAD_DIM="${HEAD_DIM:-64}"
NUM_LAYERS="${NUM_LAYERS:-12}"
REFINEMENT_HIDDEN_CHANNELS="${REFINEMENT_HIDDEN_CHANNELS:-64}"
REFINEMENT_NUM_LAYERS="${REFINEMENT_NUM_LAYERS:-2}"

MIXTURE_MEAN_WEIGHT="${MIXTURE_MEAN_WEIGHT:-0.05}"
VOXEL_GRADIENT_WEIGHT="${VOXEL_GRADIENT_WEIGHT:-0.25}"
BOUNDARY_WEIGHT="${BOUNDARY_WEIGHT:-2.0}"
SAMPLE_TIMESTEPS="${SAMPLE_TIMESTEPS:-25}"
SAMPLE_SUBSTEPS="${SAMPLE_SUBSTEPS:-4}"
SAMPLE_ORDER="${SAMPLE_ORDER:-2}"
AMP_DTYPE="${AMP_DTYPE:-bfloat16}"

LOGGER="${LOGGER:-tensorboard}"
WANDB_PROJECT="${WANDB_PROJECT:-gmflow3d-architecture-ablation}"
WANDB_ENTITY="${WANDB_ENTITY:-}"
SPLIT="${SPLIT:-}"

DRY_RUN=0
SKIP_COMPLETED=0
COMPARE_ONLY=0
NO_COMPARE=0
EXTRA_ARGS=()

usage() {
    cat <<'EOF'
Usage: ./train_architectures.sh [options] [-- EXTRA_TRAINER_ARGS...]

Train architecture variants sequentially from one checkpoint and create
comparison.csv and comparison.md in OUTPUT_ROOT.

Suites:
  all   legacy_nll plus the full 2^3 wavelet factorial and voxel_baseline.
  core  legacy_nll, subband, refiner_subband, overlap_subband, full,
        and voxel_baseline.

Named architectures:
  legacy_nll              Historical head/stem with NLL only.
  plain                   No overlap/refiner, global variance, enhanced loss.
  subband                 Per-subband variance only.
  refiner_global          Local refiner with global variance.
  refiner_subband         Local refiner plus per-subband variance.
  overlap_global          Overlapping patch stem with global variance.
  overlap_subband         Overlap plus per-subband variance.
  overlap_refiner_global  Overlap plus refiner with global variance.
  full                    Overlap, refiner, and per-subband variance.
  voxel_baseline          Voxel-space control with patch size 4.

Options:
  --checkpoint PATH          Historical standalone checkpoint.
  --data-root PATH           Directory containing OpenBHB .npy volumes.
  --metadata PATH            OpenBHB metadata TSV.
  --cache-dir PATH           Optional directory containing .pt volumes.
  --output-root PATH         Parent directory for all architecture runs.
  --architectures VALUE      all, core, or comma-separated named variants.
  --total-iters N            Iterations per architecture. Default: 5000.
  --warmup-iters N           Warmup iterations. Default: 500.
  --batch-size N             Per-device batch size. Default: 2.
  --grad-accum N             Gradient accumulation. Default: 4.
  --learning-rate VALUE      Learning rate. Default: 1e-4.
  --num-workers N            DataLoader workers. Default: 4.
  --seed N                   Shared seed. Default: 42.
  --save-interval N          Checkpoint interval. Default: 5000.
  --sample-interval N        Sample interval. Default: 1000.
  --log-interval N           Scalar interval. Default: 10.
  --cuda-visible-devices IDS CUDA device list. Default: 0.
  --logger NAME              tensorboard, wandb, or both.
  --amp-dtype NAME           bfloat16, float16, or none.
  --quick-smoke              Use a tiny two-iteration model/pipeline test.
  --skip-completed           Reuse runs with the expected final checkpoint.
  --compare-only             Do not train; compare existing selected runs.
  --no-compare               Train without creating comparison reports.
  --dry-run                  Print commands without writing or training.
  -h, --help                 Show this help.

Examples:
  ./train_architectures.sh --architectures core --total-iters 10000
  CUDA_VISIBLE_DEVICES=5 BATCH_SIZE=64 ./train_architectures.sh
  ./train_architectures.sh --quick-smoke --architectures full
  ./train_architectures.sh --compare-only \
    --output-root work_dirs/architecture_ablation_20260808_120000
EOF
}

die() {
    printf 'ERROR: %s\n' "$*" >&2
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

is_nonnegative_int() {
    [[ "$1" =~ ^[0-9]+$ ]]
}

print_command() {
    printf '  '
    printf '%q ' "$@"
    printf '\n'
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --checkpoint)
            require_value "$1" "${2-}"
            CHECKPOINT="$2"
            shift 2
            ;;
        --data-root)
            require_value "$1" "${2-}"
            DATA_ROOT="$2"
            shift 2
            ;;
        --metadata)
            require_value "$1" "${2-}"
            METADATA="$2"
            shift 2
            ;;
        --cache-dir)
            require_value "$1" "${2-}"
            CACHE_DIR="$2"
            shift 2
            ;;
        --output-root)
            require_value "$1" "${2-}"
            OUTPUT_ROOT="$2"
            shift 2
            ;;
        --architectures)
            require_value "$1" "${2-}"
            ARCHITECTURES="$2"
            shift 2
            ;;
        --total-iters)
            require_value "$1" "${2-}"
            TOTAL_ITERS="$2"
            shift 2
            ;;
        --warmup-iters)
            require_value "$1" "${2-}"
            WARMUP_ITERS="$2"
            shift 2
            ;;
        --batch-size)
            require_value "$1" "${2-}"
            BATCH_SIZE="$2"
            shift 2
            ;;
        --grad-accum)
            require_value "$1" "${2-}"
            GRAD_ACCUM="$2"
            shift 2
            ;;
        --learning-rate)
            require_value "$1" "${2-}"
            LEARNING_RATE="$2"
            shift 2
            ;;
        --num-workers)
            require_value "$1" "${2-}"
            NUM_WORKERS="$2"
            shift 2
            ;;
        --seed)
            require_value "$1" "${2-}"
            SEED="$2"
            shift 2
            ;;
        --save-interval)
            require_value "$1" "${2-}"
            SAVE_INTERVAL="$2"
            shift 2
            ;;
        --sample-interval)
            require_value "$1" "${2-}"
            SAMPLE_INTERVAL="$2"
            shift 2
            ;;
        --log-interval)
            require_value "$1" "${2-}"
            LOG_INTERVAL="$2"
            shift 2
            ;;
        --cuda-visible-devices)
            require_value "$1" "${2-}"
            CUDA_VISIBLE_DEVICES="$2"
            shift 2
            ;;
        --logger)
            require_value "$1" "${2-}"
            LOGGER="$2"
            shift 2
            ;;
        --amp-dtype)
            require_value "$1" "${2-}"
            AMP_DTYPE="$2"
            shift 2
            ;;
        --quick-smoke)
            TOTAL_ITERS=2
            WARMUP_ITERS=1
            BATCH_SIZE=1
            GRAD_ACCUM=1
            NUM_WORKERS=0
            LOG_INTERVAL=1
            SAVE_INTERVAL=2
            SAMPLE_INTERVAL=2
            VOLUME_SIZE=16
            NUM_HEADS=2
            HEAD_DIM=32
            NUM_LAYERS=2
            SAMPLE_TIMESTEPS=3
            SAMPLE_SUBSTEPS=2
            shift
            ;;
        --skip-completed)
            SKIP_COMPLETED=1
            shift
            ;;
        --compare-only)
            COMPARE_ONLY=1
            shift
            ;;
        --no-compare)
            NO_COMPARE=1
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
        --)
            shift
            EXTRA_ARGS=("$@")
            break
            ;;
        *)
            die "Unknown option: $1"
            ;;
    esac
done

for value in "$TOTAL_ITERS" "$BATCH_SIZE" "$GRAD_ACCUM" "$SAVE_INTERVAL" \
        "$SAMPLE_INTERVAL" "$LOG_INTERVAL" "$SEED" "$VOLUME_SIZE" \
        "$PATCH_SIZE" "$VOXEL_PATCH_SIZE" "$NUM_GAUSSIANS" "$NUM_HEADS" \
        "$HEAD_DIM" "$NUM_LAYERS" "$SAMPLE_TIMESTEPS" "$SAMPLE_SUBSTEPS"; do
    is_positive_int "$value" || die "Expected a positive integer, got: $value"
done
is_nonnegative_int "$WARMUP_ITERS" || die "WARMUP_ITERS must be non-negative"
is_nonnegative_int "$NUM_WORKERS" || die "NUM_WORKERS must be non-negative"
[[ "$SAMPLE_ORDER" == 1 || "$SAMPLE_ORDER" == 2 ]] \
    || die "SAMPLE_ORDER must be 1 or 2"
case "$AMP_DTYPE" in
    bfloat16|float16|none) ;;
    *) die "AMP_DTYPE must be bfloat16, float16, or none" ;;
esac
case "$LOGGER" in
    tensorboard|wandb|both) ;;
    *) die "LOGGER must be tensorboard, wandb, or both" ;;
esac

if [[ "$PYTHON_BIN" == */* ]]; then
    [[ -x "$PYTHON_BIN" ]] || die "Python is not executable: $PYTHON_BIN"
else
    PYTHON_BIN="$(command -v "$PYTHON_BIN")" \
        || die "Python command not found: $PYTHON_BIN"
fi
[[ -f "$TRAINER" ]] || die "Trainer does not exist: $TRAINER"
[[ -f "$COMPARATOR" ]] || die "Comparator does not exist: $COMPARATOR"
[[ -d "$DATA_ROOT" ]] || die "Data root does not exist: $DATA_ROOT"
[[ -f "$METADATA" ]] || die "Metadata does not exist: $METADATA"
if [[ -n "$CACHE_DIR" ]]; then
    [[ -d "$CACHE_DIR" ]] || die "Cache directory does not exist: $CACHE_DIR"
fi
if [[ "$COMPARE_ONLY" -eq 0 ]]; then
    [[ -f "$CHECKPOINT" ]] || die "Checkpoint does not exist: $CHECKPOINT"
elif [[ "$DRY_RUN" -eq 0 ]]; then
    [[ -d "$OUTPUT_ROOT" ]] || die "Output root does not exist: $OUTPUT_ROOT"
fi

ALL_ARCHITECTURES=(
    legacy_nll
    plain
    subband
    refiner_global
    refiner_subband
    overlap_global
    overlap_subband
    overlap_refiner_global
    full
    voxel_baseline
)
CORE_ARCHITECTURES=(
    legacy_nll
    subband
    refiner_subband
    overlap_subband
    full
    voxel_baseline
)

case "$ARCHITECTURES" in
    all)
        SELECTED_ARCHITECTURES=("${ALL_ARCHITECTURES[@]}")
        ;;
    core)
        SELECTED_ARCHITECTURES=("${CORE_ARCHITECTURES[@]}")
        ;;
    *)
        IFS=',' read -r -a SELECTED_ARCHITECTURES <<< "$ARCHITECTURES"
        ;;
esac

is_known_architecture() {
    local candidate="$1"
    local known
    for known in "${ALL_ARCHITECTURES[@]}"; do
        [[ "$candidate" == "$known" ]] && return 0
    done
    return 1
}

for architecture in "${SELECTED_ARCHITECTURES[@]}"; do
    is_known_architecture "$architecture" \
        || die "Unknown architecture: $architecture"
done

LOGGER_ARGS=(tensorboard)
if [[ "$LOGGER" == wandb ]]; then
    LOGGER_ARGS=(wandb)
elif [[ "$LOGGER" == both ]]; then
    LOGGER_ARGS=(tensorboard wandb)
fi

set_variant_args() {
    local architecture="$1"
    local enhanced_loss=(
        --mixture_mean_weight "$MIXTURE_MEAN_WEIGHT"
        --voxel_gradient_weight "$VOXEL_GRADIENT_WEIGHT"
        --boundary_weight "$BOUNDARY_WEIGHT"
    )

    VARIANT_ARGS=(--use_wavelet --patch_size "$PATCH_SIZE")
    case "$architecture" in
        legacy_nll)
            VARIANT_ARGS+=(
                --no_overlap_patch_embed
                --no_local_refinement
                --global_logstd
                --mixture_mean_weight 0
                --voxel_gradient_weight 0
            )
            ;;
        plain)
            VARIANT_ARGS+=(
                --no_overlap_patch_embed
                --no_local_refinement
                --global_logstd
                "${enhanced_loss[@]}"
            )
            ;;
        subband)
            VARIANT_ARGS+=(
                --no_overlap_patch_embed
                --no_local_refinement
                --per_channel_logstd
                "${enhanced_loss[@]}"
            )
            ;;
        refiner_global)
            VARIANT_ARGS+=(
                --no_overlap_patch_embed
                --local_refinement
                --global_logstd
                "${enhanced_loss[@]}"
            )
            ;;
        refiner_subband)
            VARIANT_ARGS+=(
                --no_overlap_patch_embed
                --local_refinement
                --per_channel_logstd
                "${enhanced_loss[@]}"
            )
            ;;
        overlap_global)
            VARIANT_ARGS+=(
                --overlap_patch_embed
                --no_local_refinement
                --global_logstd
                "${enhanced_loss[@]}"
            )
            ;;
        overlap_subband)
            VARIANT_ARGS+=(
                --overlap_patch_embed
                --no_local_refinement
                --per_channel_logstd
                "${enhanced_loss[@]}"
            )
            ;;
        overlap_refiner_global)
            VARIANT_ARGS+=(
                --overlap_patch_embed
                --local_refinement
                --global_logstd
                "${enhanced_loss[@]}"
            )
            ;;
        full)
            VARIANT_ARGS+=(
                --overlap_patch_embed
                --local_refinement
                --per_channel_logstd
                "${enhanced_loss[@]}"
            )
            ;;
        voxel_baseline)
            VARIANT_ARGS=(
                --no_wavelet
                --patch_size "$VOXEL_PATCH_SIZE"
                --no_overlap_patch_embed
                --no_local_refinement
                --global_logstd
                "${enhanced_loss[@]}"
            )
            ;;
    esac
}

build_train_command() {
    local architecture="$1"
    local run_dir="$2"

    set_variant_args "$architecture"
    TRAIN_CMD=(
        "$PYTHON_BIN"
        "$TRAINER"
        --data_root "$DATA_ROOT"
        --metadata "$METADATA"
        --resume "$CHECKPOINT"
        --resume_weights_only
        --work_dir "$run_dir"
        --volume_size "$VOLUME_SIZE"
        --num_gaussians "$NUM_GAUSSIANS"
        --num_heads "$NUM_HEADS"
        --head_dim "$HEAD_DIM"
        --num_layers "$NUM_LAYERS"
        --refinement_hidden_channels "$REFINEMENT_HIDDEN_CHANNELS"
        --refinement_num_layers "$REFINEMENT_NUM_LAYERS"
        --batch_size "$BATCH_SIZE"
        --grad_accum "$GRAD_ACCUM"
        --lr "$LEARNING_RATE"
        --num_workers "$NUM_WORKERS"
        --total_iters "$TOTAL_ITERS"
        --warmup_iters "$WARMUP_ITERS"
        --log_interval "$LOG_INTERVAL"
        --save_interval "$SAVE_INTERVAL"
        --sample_interval "$SAMPLE_INTERVAL"
        --seed "$SEED"
        --sample_timesteps "$SAMPLE_TIMESTEPS"
        --sample_substeps "$SAMPLE_SUBSTEPS"
        --sample_order "$SAMPLE_ORDER"
        --logger "${LOGGER_ARGS[@]}"
        --wandb_project "$WANDB_PROJECT"
        --wandb_name "${architecture}_${RUN_TAG}"
        "${VARIANT_ARGS[@]}"
    )

    if [[ -n "$WANDB_ENTITY" ]]; then
        TRAIN_CMD+=(--wandb_entity "$WANDB_ENTITY")
    fi
    if [[ -n "$CACHE_DIR" ]]; then
        TRAIN_CMD+=(--cache_dir "$CACHE_DIR")
    fi
    if [[ -n "$SPLIT" ]]; then
        TRAIN_CMD+=(--split "$SPLIT")
    fi
    if [[ "$AMP_DTYPE" == none ]]; then
        TRAIN_CMD+=(--no_amp)
    else
        TRAIN_CMD+=(--autocast_dtype "$AMP_DTYPE")
    fi
    TRAIN_CMD+=("${EXTRA_ARGS[@]}")
}

RUN_SPECS=()
for architecture in "${SELECTED_ARCHITECTURES[@]}"; do
    RUN_SPECS+=("${architecture}=${OUTPUT_ROOT}/${architecture}")
done

if [[ "$DRY_RUN" -eq 1 ]]; then
    printf 'Architecture ablation dry run\n'
    printf 'Output root: %s\n' "$OUTPUT_ROOT"
    if [[ "$COMPARE_ONLY" -eq 0 ]]; then
        for architecture in "${SELECTED_ARCHITECTURES[@]}"; do
            build_train_command "$architecture" "$OUTPUT_ROOT/$architecture"
            printf '\n[%s]\n' "$architecture"
            print_command env \
                "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES" \
                PYTHONUNBUFFERED=1 "${TRAIN_CMD[@]}"
        done
    fi
    if [[ "$NO_COMPARE" -eq 0 ]]; then
        COMPARE_CMD=("$PYTHON_BIN" "$COMPARATOR" --output-dir "$OUTPUT_ROOT")
        for spec in "${RUN_SPECS[@]}"; do
            COMPARE_CMD+=(--run "$spec")
        done
        printf '\n[comparison]\n'
        print_command "${COMPARE_CMD[@]}"
    fi
    exit 0
fi

export CUDA_VISIBLE_DEVICES
export PYTHONUNBUFFERED=1

if [[ "$COMPARE_ONLY" -eq 0 ]]; then
    mkdir -p "$OUTPUT_ROOT"
    {
        printf 'run_tag=%s\n' "$RUN_TAG"
        printf 'checkpoint=%s\n' "$CHECKPOINT"
        printf 'data_root=%s\n' "$DATA_ROOT"
        printf 'metadata=%s\n' "$METADATA"
        printf 'architectures=%s\n' "${SELECTED_ARCHITECTURES[*]}"
        printf 'total_iters=%s\n' "$TOTAL_ITERS"
        printf 'seed=%s\n' "$SEED"
        printf 'cuda_visible_devices=%s\n' "$CUDA_VISIBLE_DEVICES"
    } > "$OUTPUT_ROOT/experiment_manifest.txt"
fi

FAILURES=()
for architecture in "${SELECTED_ARCHITECTURES[@]}"; do
    run_dir="$OUTPUT_ROOT/$architecture"
    printf -v final_checkpoint '%s/checkpoints/iter_%07d.pt' \
        "$run_dir" "$TOTAL_ITERS"

    if [[ "$COMPARE_ONLY" -eq 1 ]]; then
        continue
    fi
    if [[ -f "$final_checkpoint" && "$SKIP_COMPLETED" -eq 1 ]]; then
        printf '[%s] already complete; skipping training.\n' "$architecture"
        continue
    fi
    if [[ -f "$run_dir/train_log.jsonl" || -d "$run_dir/checkpoints" ]]; then
        die "Run directory is not empty: $run_dir. Use a new OUTPUT_ROOT or --skip-completed."
    fi

    mkdir -p "$run_dir"
    build_train_command "$architecture" "$run_dir"
    {
        printf '# Architecture: %s\n' "$architecture"
        printf '# Command:\n'
        printf '%q ' env "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES" \
            PYTHONUNBUFFERED=1 "${TRAIN_CMD[@]}"
        printf '\n'
    } > "$run_dir/command.txt"

    printf '\n============================================================\n'
    printf 'Training architecture: %s\n' "$architecture"
    printf 'Output: %s\n' "$run_dir"
    printf '============================================================\n'

    if "${TRAIN_CMD[@]}" 2>&1 | tee "$run_dir/launcher.log"; then
        printf '0\n' > "$run_dir/exit_status.txt"
    else
        status=$?
        printf '%s\n' "$status" > "$run_dir/exit_status.txt"
        FAILURES+=("${architecture}:${status}")
        printf 'Architecture %s failed with status %s; continuing.\n' \
            "$architecture" "$status" >&2
    fi
done

if [[ "$NO_COMPARE" -eq 0 ]]; then
    mkdir -p "$OUTPUT_ROOT"
    COMPARE_CMD=("$PYTHON_BIN" "$COMPARATOR" --output-dir "$OUTPUT_ROOT")
    for spec in "${RUN_SPECS[@]}"; do
        COMPARE_CMD+=(--run "$spec")
    done
    printf '\nGenerating comparison report:\n'
    print_command "${COMPARE_CMD[@]}"
    "${COMPARE_CMD[@]}"
fi

if [[ "${#FAILURES[@]}" -gt 0 ]]; then
    printf 'Failed runs: %s\n' "${FAILURES[*]}" >&2
    exit 1
fi

printf '\nAll selected architecture runs completed.\n'
printf 'Results: %s\n' "$OUTPUT_ROOT"
