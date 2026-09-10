#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${SCRIPT_DIR}"
cd "${PROJECT_ROOT}"

# Environment workers do NumPy/Python scheduling work. Prevent each worker
# from creating its own BLAS thread pool and oversubscribing the host CPUs.
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"
export CUDA_MODULE_LOADING="${CUDA_MODULE_LOADING:-LAZY}"
export PYTHONUNBUFFERED=1

RUN_DIR="${RUN_DIR:-outputs/experiments/ppo/a800_runs/a800_$(date +%Y%m%d_%H%M%S)}"
DATA_ROOT="${DATA_ROOT:-data}"
SPLITS_PATH="${SPLITS_PATH:-${PROJECT_ROOT}/splits.json}"
# Reference-rule makespans for the validation split (checkpoint selection).
# Build it first (docs/EXECUTION_FLOW.md S3):
#   python -m scripts.baselines --data-root data --instance-workers 8 \
#     --output-csv outputs/rules_psp_grid/makespan_summary.csv
REF_RULES="${REF_RULES:-outputs/rules_psp_grid/makespan_summary.csv}"
LOG_DIR="${LOG_DIR:-${PROJECT_ROOT}/logs/ppo}"
mkdir -p "${LOG_DIR}"
TRAIN_LOG_FILE="${LOG_DIR}/a800_$(date +%Y%m%d_%H%M%S).log"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-128}"
VALIDATION_INTERVAL="${VALIDATION_INTERVAL:-10}"
EARLY_STOP_PATIENCE="${EARLY_STOP_PATIENCE:-12}"
GAMMA="${GAMMA:-0.999}"
GAE_LAMBDA="${GAE_LAMBDA:-0.98}"
BATCH_SIZE="${BATCH_SIZE:-4096}"
N_EPOCHS="${N_EPOCHS:-5}"
LEARNING_RATE="${LEARNING_RATE:-2e-4}"
ENT_COEF="${ENT_COEF:-0.005}"
VF_COEF="${VF_COEF:-0.5}"
TARGET_KL="${TARGET_KL:-0.02}"
CRITICAL_PATH_SHAPING="${CRITICAL_PATH_SHAPING:-0.5}"
N_ENVS="${N_ENVS:-48}"
TOTAL_TIMESTEPS="${TOTAL_TIMESTEPS:-6400000}"
EVAL_ALL="${EVAL_ALL:-1}"
EVAL_SUITES="${EVAL_SUITES:-psplib_j30,psplib_j60,psplib_j90,psplib_j120}"

EVAL_ARGS=()
if [[ "${EVAL_ALL}" == "1" ]]; then
    EVAL_ARGS=(--eval-suites "${EVAL_SUITES}")
fi

nohup python -m scripts.train_ppo \
    --data-root "${DATA_ROOT}" \
    --splits "${SPLITS_PATH}" \
    --ref-rules "${REF_RULES}" \
    --max-activities 122 \
    --max-resources 4 \
    --n-envs "${N_ENVS}" \
    --total-timesteps "${TOTAL_TIMESTEPS}" \
    --n-steps 256 \
    --batch-size "${BATCH_SIZE}" \
    --n-epochs "${N_EPOCHS}" \
    --learning-rate "${LEARNING_RATE}" \
    --ent-coef "${ENT_COEF}" \
    --vf-coef "${VF_COEF}" \
    --critical-path-shaping "${CRITICAL_PATH_SHAPING}" \
    --target-kl "${TARGET_KL}" \
    --gamma "${GAMMA}" \
    --gae-lambda "${GAE_LAMBDA}" \
    --gin-layers 2 \
    --device cuda:0 \
    --mixed-precision bf16 \
    --cuda-matmul-precision high \
    --torch-compile \
    --compile-mode reduce-overhead \
    --vec-env subproc \
    --start-method spawn \
    --torch-threads 1 \
    --torch-interop-threads 1 \
    --early-stop-patience "${EARLY_STOP_PATIENCE}" \
    --validation-interval "${VALIDATION_INTERVAL}" \
    --eval-batch-size "${EVAL_BATCH_SIZE}" \
    --seed 17 \
    ${EVAL_ARGS[@]+"${EVAL_ARGS[@]}"} \
    --output-dir "${RUN_DIR}" \
    "$@" >"${TRAIN_LOG_FILE}" 2>&1 &

TRAIN_PID=$!
echo "training started in background: PID=${TRAIN_PID}"
echo "log: ${TRAIN_LOG_FILE}"
echo "run dir: ${RUN_DIR}"
