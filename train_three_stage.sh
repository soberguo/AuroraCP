#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

PYTHON_BIN="${PYTHON_BIN:-/home/guoyixin/miniconda3/envs/aurora/bin/python}"
GPU_ID="${GPU_ID:-0,1}"
GPU_IDS="${GPU_IDS:-$GPU_ID}"
DATA_FOLDER="${DATA_FOLDER:-/sharefiles1/guoyixin/datasets/weatherbench2_73var}"
PRETRAINED="${PRETRAINED:-ckpt/aurora-0.25-pretrained.ckpt}"
RUN_ROOT="${RUN_ROOT:-weights/three_stage_73var_12h}"
LOG_DIR="${LOG_DIR:-log/three_stage_73var_12h}"
STAGE3_LOG_FILE="${STAGE3_LOG_FILE:-stage3_replay_rollout_from_epoch030_eval040.log}"

TIMESTEP="${TIMESTEP:-12}"
TRAIN_START_YEAR="${TRAIN_START_YEAR:-2010}"
TRAIN_END_YEAR="${TRAIN_END_YEAR:-2018}"
STAGE1_TRAIN_START_YEAR="${STAGE1_TRAIN_START_YEAR:-1979}"
STAGE1_TRAIN_END_YEAR="${STAGE1_TRAIN_END_YEAR:-2018}"
STAGE2_TRAIN_START_YEAR="${STAGE2_TRAIN_START_YEAR:-$TRAIN_START_YEAR}"
STAGE2_TRAIN_END_YEAR="${STAGE2_TRAIN_END_YEAR:-$TRAIN_END_YEAR}"
TEST_START_YEAR="${TEST_START_YEAR:-2020}"
TEST_END_YEAR="${TEST_END_YEAR:-2021}"
LR1="${LR1:-5e-5}"
LR2="${LR2:-5e-5}"
STAGE1_LR1="${STAGE1_LR1:-1e-3}"
STAGE1_LR2="${STAGE1_LR2:-1e-4}"
STAGE2_LR1="${STAGE2_LR1:-$LR1}"
STAGE2_LR2="${STAGE2_LR2:-$LR2}"
MASK_OCEAN_FOR_LAND_VARS="${MASK_OCEAN_FOR_LAND_VARS:-false}"
SAVE_EVERY="${SAVE_EVERY:-1}"
EVAL_EVERY="${EVAL_EVERY:-1}"
STAGE3_EVAL_EVERY="${STAGE3_EVAL_EVERY:-1}"
STAGE3_EVAL_START_EPOCH="${STAGE3_EVAL_START_EPOCH:-40}"

# These are cumulative epoch numbers because each training entrypoint resumes
# its epoch counter from the checkpoint produced by the previous stage.
STAGE1_END_EPOCH="${STAGE1_END_EPOCH:-20}"
STAGE2_END_EPOCH="${STAGE2_END_EPOCH:-30}"
STAGE3_END_EPOCH="${STAGE3_END_EPOCH:-100}"
STAGE1_BATCH_SIZE="${STAGE1_BATCH_SIZE:-4}"
STAGE2_BATCH_SIZE="${STAGE2_BATCH_SIZE:-4}"
STAGE3_BATCH_SIZE="${STAGE3_BATCH_SIZE:-4}"
STAGE3_SHORT_MAX_LEAD="${STAGE3_SHORT_MAX_LEAD:-12}"
STAGE3_LONG_MAX_LEAD="${STAGE3_LONG_MAX_LEAD:-20}"
STAGE3_CURRICULUM_SWITCH_EPOCH="${STAGE3_CURRICULUM_SWITCH_EPOCH:-55}"
REPLAY_STATE_KEEP="${REPLAY_STATE_KEEP:-2}"
DRY_RUN="${DRY_RUN:-0}"

STAGE1_DIR="$RUN_ROOT/stage1_single_step"
STAGE2_DIR="$RUN_ROOT/stage2_two_step"
STAGE3_DIR="${STAGE3_DIR:-$RUN_ROOT/stage3_replay_rollout_from_epoch030_eval040}"
mkdir -p "$STAGE1_DIR" "$STAGE2_DIR" "$STAGE3_DIR" "$LOG_DIR"

if [[ ! -x "$PYTHON_BIN" ]]; then
    echo "Python executable not found: $PYTHON_BIN" >&2
    exit 1
fi

GPU_IDS="${GPU_IDS//[[:space:]]/}"
if [[ ! "$GPU_IDS" =~ ^[0-9]+(,[0-9]+)*$ ]]; then
    echo "GPU_IDS must be a comma-separated list of GPU indices, for example: 0 or 0,1." >&2
    exit 1
fi
IFS=',' read -r -a GPU_ID_ARRAY <<< "$GPU_IDS"
NUM_GPUS="${#GPU_ID_ARRAY[@]}"

if (( NUM_GPUS > 1 )); then
    DISTRIBUTED="true"
    TRAIN_LAUNCHER=(
        "$PYTHON_BIN" -m torch.distributed.run
        --standalone
        --nproc_per_node="$NUM_GPUS"
    )
else
    DISTRIBUTED="false"
    TRAIN_LAUNCHER=("$PYTHON_BIN")
fi

echo "Using GPU_IDS=$GPU_IDS ($NUM_GPUS process(es)); stage batch sizes are per GPU."

for value in "$STAGE1_END_EPOCH" "$STAGE2_END_EPOCH" "$STAGE3_END_EPOCH"; do
    if [[ ! "$value" =~ ^[0-9]+$ ]]; then
        echo "Stage end epochs must be positive integers." >&2
        exit 1
    fi
done
if (( STAGE1_END_EPOCH <= 0 || STAGE2_END_EPOCH <= STAGE1_END_EPOCH || STAGE3_END_EPOCH <= STAGE2_END_EPOCH )); then
    echo "Expected 0 < STAGE1_END_EPOCH < STAGE2_END_EPOCH < STAGE3_END_EPOCH." >&2
    exit 1
fi

checkpoint_for_epoch() {
    local directory="$1"
    local epoch="$2"
    printf '%s/epoch_%03d.pt' "$directory" "$epoch"
}

latest_checkpoint() {
    local directory="$1"
    local filename
    filename="$(find "$directory" -maxdepth 1 -type f -name 'epoch_[0-9][0-9][0-9].pt' -printf '%f\n' | sort | tail -n 1)"
    if [[ -n "$filename" ]]; then
        printf '%s/%s' "$directory" "$filename"
    fi
}

run_command() {
    printf 'Running:'
    printf ' %q' "$@"
    printf '\n'
    if [[ "$DRY_RUN" == "1" ]]; then
        return 0
    fi
    "$@"
}

COMMON_ARGS=(
    --device cuda
    --distributed "$DISTRIBUTED"
    --pretrained "$PRETRAINED"
    --timestep "$TIMESTEP"
    --test_year "$TEST_START_YEAR" "$TEST_END_YEAR"
    --data_folder "$DATA_FOLDER"
    --save_every "$SAVE_EVERY"
    --mask_ocean_for_land_vars "$MASK_OCEAN_FOR_LAND_VARS"
)

run_main_stage() {
    local stage_name="$1"
    local output_dir="$2"
    local end_epoch="$3"
    local train_roll_step="$4"
    local eval_roll_step="$5"
    local batch_size="$6"
    local train_start_year="$7"
    local train_end_year="$8"
    local lr1="$9"
    local lr2="${10}"
    local initial_checkpoint="${11:-}"
    local final_checkpoint
    local resume_checkpoint
    local checkpoint_args=()
    local reset_lr_on_resume="false"

    final_checkpoint="$(checkpoint_for_epoch "$output_dir" "$end_epoch")"
    if [[ -f "$final_checkpoint" ]]; then
        echo "[$stage_name] Complete; using $final_checkpoint"
        return 0
    fi

    resume_checkpoint="$(latest_checkpoint "$output_dir")"
    if [[ -n "$resume_checkpoint" ]]; then
        checkpoint_args=(--ckpt "$resume_checkpoint")
        echo "[$stage_name] Resuming from $resume_checkpoint"
    elif [[ -n "$initial_checkpoint" ]]; then
        if [[ ! -f "$initial_checkpoint" && "$DRY_RUN" != "1" ]]; then
            echo "[$stage_name] Missing input checkpoint: $initial_checkpoint" >&2
            exit 1
        fi
        checkpoint_args=(--ckpt "$initial_checkpoint")
        reset_lr_on_resume="true"
        echo "[$stage_name] Initializing from $initial_checkpoint"
    else
        echo "[$stage_name] Initializing from pretrained weights: $PRETRAINED"
    fi

    run_command env CUDA_VISIBLE_DEVICES="$GPU_IDS" "${TRAIN_LAUNCHER[@]}" main.py \
        "${COMMON_ARGS[@]}" \
        --save_dir "$output_dir" \
        --log_dir "$LOG_DIR" \
        --log_file "${stage_name}.log" \
        --epochs "$end_epoch" \
        --batchsize "$batch_size" \
        --eval_every "$EVAL_EVERY" \
        --train_year "$train_start_year" "$train_end_year" \
        --lr1 "$lr1" \
        --lr2 "$lr2" \
        --reset_lr_on_resume "$reset_lr_on_resume" \
        --train_roll_step "$train_roll_step" \
        --roll_step "$eval_roll_step" \
        "${checkpoint_args[@]}"

    if [[ "$DRY_RUN" != "1" && ! -f "$final_checkpoint" ]]; then
        echo "[$stage_name] Expected checkpoint was not produced: $final_checkpoint" >&2
        exit 1
    fi
}

run_rollout_stage() {
    local initial_checkpoint="$1"
    local final_checkpoint
    local resume_checkpoint

    final_checkpoint="$(checkpoint_for_epoch "$STAGE3_DIR" "$STAGE3_END_EPOCH")"
    if [[ -f "$final_checkpoint" ]]; then
        echo "[stage3_replay_rollout] Complete; using $final_checkpoint"
        return 0
    fi

    resume_checkpoint="$(latest_checkpoint "$STAGE3_DIR")"
    if [[ -z "$resume_checkpoint" ]]; then
        resume_checkpoint="$initial_checkpoint"
    fi
    if [[ ! -f "$resume_checkpoint" && "$DRY_RUN" != "1" ]]; then
        echo "[stage3_replay_rollout] Missing input checkpoint: $resume_checkpoint" >&2
        exit 1
    fi
    echo "[stage3_replay_rollout] Initializing/resuming from $resume_checkpoint"

    run_command env CUDA_VISIBLE_DEVICES="$GPU_IDS" "${TRAIN_LAUNCHER[@]}" rollout_finetune.py \
        "${COMMON_ARGS[@]}" \
        --save_dir "$STAGE3_DIR" \
        --log_dir "$LOG_DIR" \
        --log_file "$STAGE3_LOG_FILE" \
        --epochs "$STAGE3_END_EPOCH" \
        --batchsize "$STAGE3_BATCH_SIZE" \
        --eval_every "$STAGE3_EVAL_EVERY" \
        --eval_start_epoch "$STAGE3_EVAL_START_EPOCH" \
        --replay_state_keep "$REPLAY_STATE_KEEP" \
        --train_year "$TRAIN_START_YEAR" "$TRAIN_END_YEAR" \
        --lr1 "$LR1" \
        --lr2 "$LR2" \
        --train_roll_step 1 \
        --roll_step "$STAGE3_LONG_MAX_LEAD" \
        --rollout_short_max_lead "$STAGE3_SHORT_MAX_LEAD" \
        --rollout_long_max_lead "$STAGE3_LONG_MAX_LEAD" \
        --rollout_curriculum_switch_epoch "$STAGE3_CURRICULUM_SWITCH_EPOCH" \
        --ckpt "$resume_checkpoint"

    if [[ "$DRY_RUN" != "1" && ! -f "$final_checkpoint" ]]; then
        echo "[stage3_replay_rollout] Expected checkpoint was not produced: $final_checkpoint" >&2
        exit 1
    fi
}

STAGE1_FINAL="$(checkpoint_for_epoch "$STAGE1_DIR" "$STAGE1_END_EPOCH")"
STAGE2_FINAL="$(checkpoint_for_epoch "$STAGE2_DIR" "$STAGE2_END_EPOCH")"
STAGE3_FINAL="$(checkpoint_for_epoch "$STAGE3_DIR" "$STAGE3_END_EPOCH")"

run_main_stage stage1_single_step "$STAGE1_DIR" "$STAGE1_END_EPOCH" 1 1 "$STAGE1_BATCH_SIZE" "$STAGE1_TRAIN_START_YEAR" "$STAGE1_TRAIN_END_YEAR" "$STAGE1_LR1" "$STAGE1_LR2"
run_main_stage stage2_two_step "$STAGE2_DIR" "$STAGE2_END_EPOCH" 2 2 "$STAGE2_BATCH_SIZE" "$STAGE2_TRAIN_START_YEAR" "$STAGE2_TRAIN_END_YEAR" "$STAGE2_LR1" "$STAGE2_LR2" "$STAGE1_FINAL"
run_rollout_stage "$STAGE2_FINAL"

echo "Three-stage training complete. Final checkpoint: $STAGE3_FINAL"
