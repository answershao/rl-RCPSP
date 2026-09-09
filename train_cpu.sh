#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${SCRIPT_DIR}"
cd "${PROJECT_ROOT}"

# This host has 52 physical cores across two NUMA nodes. Keep NumPy work in
# each environment single-threaded and reserve CPU capacity for PPO updates.
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export KMP_BLOCKTIME="${KMP_BLOCKTIME:-0}"
export MALLOC_ARENA_MAX="${MALLOC_ARENA_MAX:-4}"
export PYTHONUNBUFFERED=1

RUN_STAMP="${RUN_STAMP:-$(date +%Y%m%d_%H%M%S)}"
RUN_DIR="${RUN_DIR:-outputs/experiments/ppo/cpu_runs/cpu_${RUN_STAMP}}"
SPLITS_PATH="${SPLITS_PATH:-${RUN_DIR}/splits.json}"
LOG_DIR="${LOG_DIR:-${PROJECT_ROOT}/logs/ppo}"
mkdir -p "${LOG_DIR}"
TRAIN_LOG_FILE="${LOG_DIR}/cpu_${RUN_STAMP}.log"

if [[ -f "${RUN_DIR}/final_model.zip" && "${ALLOW_OVERWRITE_BASELINE:-0}" != "1" ]]; then
    echo "CPU baseline already exists: ${RUN_DIR}" >&2
    echo "Set RUN_DIR to a new experiment directory, or explicitly set ALLOW_OVERWRITE_BASELINE=1." >&2
    exit 2
fi

# Keep the total CPU footprint close to the 52 physical cores on this host.
# These can be overridden for benchmarking, for example N_ENVS=40.
N_ENVS="${N_ENVS:-32}"
N_STEPS="${N_STEPS:-384}"
# The retained baseline uses a larger minibatch and three PPO epochs.
BATCH_SIZE="${BATCH_SIZE:-4096}"
N_EPOCHS="${N_EPOCHS:-3}"
TORCH_THREADS="${TORCH_THREADS:-20}"
TOTAL_TIMESTEPS="${TOTAL_TIMESTEPS:-10000000}"
LEARNING_RATE="${LEARNING_RATE:-2e-4}"
ENT_COEF="${ENT_COEF:-0.001}"
VF_COEF="${VF_COEF:-0.1}"
TARGET_KL="${TARGET_KL:-0.01}"
GAMMA="${GAMMA:-0.999}"
GAE_LAMBDA="${GAE_LAMBDA:-0.98}"
EARLY_STOP_PATIENCE="${EARLY_STOP_PATIENCE:-5}"
# Validation runs 225 complete schedules and is intentionally less frequent
# than the training rollouts; override this for tighter early-stop monitoring.
VALIDATION_INTERVAL="${VALIDATION_INTERVAL:-25}"
VALIDATION_MIN_DELTA="${VALIDATION_MIN_DELTA:-0}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-128}"
SEED="${SEED:-17}"

nohup python -m scripts.train_ppo \
    --instances-root data/MPLIB2_train_10_50_5 \
    --n-envs "${N_ENVS}" \
    --total-timesteps "${TOTAL_TIMESTEPS}" \
    --n-steps "${N_STEPS}" \
    --batch-size "${BATCH_SIZE}" \
    --n-epochs "${N_EPOCHS}" \
    --learning-rate "${LEARNING_RATE}" \
    --ent-coef "${ENT_COEF}" \
    --vf-coef "${VF_COEF}" \
    --target-kl "${TARGET_KL}" \
    --gamma "${GAMMA}" \
    --gae-lambda "${GAE_LAMBDA}" \
    --gin-layers 2 \
    --device cpu \
    --mixed-precision none \
    --vec-env subproc \
    --start-method spawn \
    --torch-threads "${TORCH_THREADS}" \
    --torch-interop-threads 1 \
    --early-stop-patience "${EARLY_STOP_PATIENCE}" \
    --validation-interval "${VALIDATION_INTERVAL}" \
    --validation-min-delta "${VALIDATION_MIN_DELTA}" \
    --eval-batch-size "${EVAL_BATCH_SIZE}" \
    --seed "${SEED}" \
    --splits "${SPLITS_PATH}" \
    --baseline-results outputs/baselines_mplib2_10_50_5/makespan_summary.csv \
    --output-dir "${RUN_DIR}" \
    "$@" >"${TRAIN_LOG_FILE}" 2>&1 &

TRAIN_PID=$!
echo "training started in background: PID=${TRAIN_PID}"
echo "log: ${TRAIN_LOG_FILE}"

if [[ "${WAIT_FOR_TRAINING:-0}" == "1" ]]; then
    wait "${TRAIN_PID}"
fi
