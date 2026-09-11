#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${SCRIPT_DIR}"
cd "${PROJECT_ROOT}"

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export KMP_BLOCKTIME="${KMP_BLOCKTIME:-0}"
export MALLOC_ARENA_MAX="${MALLOC_ARENA_MAX:-4}"
export PYTHONUNBUFFERED=1

MODELS_ROOT="${MODELS_ROOT:-outputs/experiments/ppo}"
MODEL_FILE="${MODEL_FILE:-final_model.zip}"
OUTPUT_DIR="${OUTPUT_DIR:-${MODELS_ROOT}/inference_search}"
SPLITS="${SPLITS:-splits.json}"
DATA_ROOT="${DATA_ROOT:-data}"
EVAL_GROUPS="${EVAL_GROUPS:-psplib_j30,psplib_j60,psplib_j90,psplib_j120}"
# Test-side reference rules: must cover every group in EVAL_GROUPS.  Build the
# matching CSV first:
#   python -m scripts.baselines --data-root data \
#     --suites psplib_j30,psplib_j60,psplib_j90,psplib_j120 \
#     --instance-workers 8 --output-csv outputs/rules_psplib/makespan_summary.csv
REF_RULES="${REF_RULES:-outputs/rules_psplib/makespan_summary.csv}"
REF_RULE="${REF_RULE:-serial_LST}"
SEEDS="${SEEDS:-17}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-8}"
EVAL_MAX_INSTANCES="${EVAL_MAX_INSTANCES:-0}"
WORKERS="${WORKERS:-48}"
EVALUATION_SEED="${EVALUATION_SEED:-20260910}"
TORCH_THREADS="${TORCH_THREADS:-1}"
TORCH_INTEROP_THREADS="${TORCH_INTEROP_THREADS:-1}"

read -r -a SEED_ARGS <<< "${SEEDS}"

if [[ -z "${MODELS_ROOT}" ]]; then
    echo "set MODELS_ROOT" >&2
    exit 1
fi
for seed in "${SEED_ARGS[@]}"; do
    model_path="${MODELS_ROOT}/seed${seed}/${MODEL_FILE}"
    if [[ ! -f "${model_path}" ]]; then
        echo "PPO model not found: ${model_path}" >&2
        exit 1
    fi
done
if [[ ! -f "${SPLITS}" ]]; then
    echo "protocol splits not found: ${SPLITS}" >&2
    exit 1
fi

LOG_DIR="${LOG_DIR:-${PROJECT_ROOT}/logs/ppo/inference_search}"
mkdir -p "${LOG_DIR}"
SEARCH_LOG_FILE="${LOG_DIR}/search_$(date +%Y%m%d_%H%M%S).log"

nohup python -m scripts.search_ppo \
    --models-root "${MODELS_ROOT}" \
    --model-file "${MODEL_FILE}" \
    --seeds "${SEED_ARGS[@]}" \
    --splits "${SPLITS}" \
    --data-root "${DATA_ROOT}" \
    --eval-groups "${EVAL_GROUPS}" \
    --ref-rules "${REF_RULES}" \
    --ref-rule "${REF_RULE}" \
    --output-dir "${OUTPUT_DIR}" \
    --device cpu \
    --eval-batch-size "${EVAL_BATCH_SIZE}" \
    --eval-max-instances "${EVAL_MAX_INSTANCES}" \
    --workers "${WORKERS}" \
    --torch-threads "${TORCH_THREADS}" \
    --torch-interop-threads "${TORCH_INTEROP_THREADS}" \
    --evaluation-seed "${EVALUATION_SEED}" \
    "$@" >"${SEARCH_LOG_FILE}" 2>&1 &

SEARCH_PID=$!
echo "inference search started in background: PID=${SEARCH_PID}"
echo "log: ${SEARCH_LOG_FILE}"
echo "results: ${OUTPUT_DIR}"

if [[ "${WAIT_FOR_SEARCH:-0}" == "1" ]]; then
    wait "${SEARCH_PID}"
fi
