#!/usr/bin/env bash
# Step 2 (+ step 4 merge) of the agreed pipeline, WITHOUT PPO training:
#   solve the test suites (PSPLIB j30-j120) with priority rules / GA
#   (GPHH optional, off by default -- enable with RUN_GPHH=1),
#   then merge everything into one comparison CSV (gap vs BKS).
#
# Step 0 (instance diagnostics) runs first: scripts/instance_stats writes
#   outputs/instance_stats/instances.csv, which the aggregate stage needs to
#   emit summary_by_regime.csv (RF level x RS quartile -- the main line reports
#   per regime, pooled means hide the tight/loose difference).  Disable with
#   RUN_STATS=0; the aggregate stage then drops --params and only writes
#   merged_detail.csv + summary_by_suite.csv.
#
# Step 3 (PPO) runs separately via train_a800.sh / train_cpu.sh.  After a
# trained run exists, re-run this script with WITH_PPO=1 to fold the
# ppo_makespan column into the same merged table.
#
# Usage:
#   bash run_test_baselines.sh                       # full run, front
#   nohup bash run_test_baselines.sh > /dev/null 2>&1 &   # background
#   RUN_RULES=0 RUN_GA=1 RUN_GPHH=0 bash run_test_baselines.sh  # selective rerun
#   SMOKE_MAX_INSTANCES=2 GA_POPULATION=20 GA_GENERATIONS=30 bash run_test_baselines.sh  # smoke
#
# Re-run safety: a stage whose output CSV already exists is SKIPPED unless
# ALLOW_OVERWRITE=1 is set -- the GA stage can take hours, do not redo it
# accidentally.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${SCRIPT_DIR}"
cd "${PROJECT_ROOT}"

# ---------------- configuration (env-overridable) ----------------
PYTHON="${PYTHON:-python}"
DATA_ROOT="${DATA_ROOT:-data}"
SPLITS_PATH="${SPLITS_PATH:-${PROJECT_ROOT}/splits.json}"
BKS_JSON="${BKS_JSON:-data/bks/bks_psplib.json}"
SEED="${SEED:-17}"
# test suites per the agreed protocol: PSPLIB only (rg300/patterson are optional
# generalization references, not part of the main test)
SUITES="${SUITES:-psplib_j30,psplib_j60,psplib_j90,psplib_j120}"

RUN_STATS="${RUN_STATS:-1}"
RUN_RULES="${RUN_RULES:-1}"
RUN_GA="${RUN_GA:-1}"
# GPHH disabled by default: GP evolution is slow (train 40 trees x N generations
# before any evaluation).  Enable explicitly with RUN_GPHH=1.
RUN_GPHH="${RUN_GPHH:-0}"
RUN_AGGREGATE="${RUN_AGGREGATE:-1}"
ALLOW_OVERWRITE="${ALLOW_OVERWRITE:-0}"
WITH_PPO="${WITH_PPO:-0}"
PPO_SUMMARY="${PPO_SUMMARY:-outputs/experiments/ppo/ppo_rcpsp/ppo_eval_summary.csv}"

# 0 = full suite; set e.g. SMOKE_MAX_INSTANCES=2 for a smoke run
MAX_INSTANCES="${SMOKE_MAX_INSTANCES:-0}"

RULES_WORKERS="${RULES_WORKERS:-8}"
GA_WORKERS="${GA_WORKERS:-8}"
GPHH_WORKERS="${GPHH_WORKERS:-6}"
GA_POPULATION="${GA_POPULATION:-50}"
GA_GENERATIONS="${GA_GENERATIONS:-200}"
GPHH_TRAIN_INSTANCES="${GPHH_TRAIN_INSTANCES:-40}"

OUT_STATS="${OUT_STATS:-outputs/instance_stats}"
OUT_RULES="${OUT_RULES:-outputs/rules_psplib/makespan_summary.csv}"
OUT_GA="${OUT_GA:-outputs/ga_psplib/ga.csv}"
OUT_GPHH_DIR="${OUT_GPHH_DIR:-outputs/gphh_psplib/v1}"
OUT_AGG="${OUT_AGG:-outputs/aggregate_psplib}"
LOG_DIR="${LOG_DIR:-logs/baselines}"
RUN_STAMP="${RUN_STAMP:-$(date +%Y%m%d_%H%M%S)}"
mkdir -p "${LOG_DIR}"

MAX_ARGS=()
if [[ "${MAX_INSTANCES}" != "0" ]]; then
    MAX_ARGS=(--max-instances "${MAX_INSTANCES}")
    echo "SMOKE mode: at most ${MAX_INSTANCES} instance(s) per suite"
fi
# NOTE: macOS ships bash 3.2 where `set -u` + expanding an EMPTY array
# ("${arr[@]}") aborts with "unbound variable".  Every array expansion below
# therefore uses the ${arr[@]+...} guard, which is safe on bash 3.2.

# stage <name> <output-file> -- skips when the output exists (unless
# ALLOW_OVERWRITE=1); always runs the aggregate stage (it is cheap).
stage_ready() {
    local name="$1" output="$2"
    if [[ -f "${output}" && "${ALLOW_OVERWRITE}" != "1" ]]; then
        echo "[skip] ${name}: output already exists -> ${output}"
        echo "       (set ALLOW_OVERWRITE=1 to recompute this stage)"
        return 1
    fi
    return 0
}

echo "== step 2/4: test-suite baselines (seed ${SEED}, suites ${SUITES}) =="

# ---------------- stage 0: instance diagnostics (feeds --params) -------------
if [[ "${RUN_STATS}" == "1" ]] && stage_ready "instance_stats" "${OUT_STATS}/instances.csv"; then
    echo "[run ] instance_stats -> ${OUT_STATS}/instances.csv"
    "${PYTHON}" -m scripts.instance_stats \
        --data-root "${DATA_ROOT}" \
        --splits "${SPLITS_PATH}" \
        --workers "${RULES_WORKERS}" \
        --output-dir "${OUT_STATS}" 2>&1 | tee "${LOG_DIR}/stats_${RUN_STAMP}.log"
fi

# ---------------- stage 1: priority rules (fast) ----------------
if [[ "${RUN_RULES}" == "1" ]] && stage_ready "rules" "${OUT_RULES}"; then
    echo "[run ] rules -> ${OUT_RULES}"
    "${PYTHON}" -m scripts.baselines \
        --data-root "${DATA_ROOT}" \
        --suites "${SUITES}" \
        --seed "${SEED}" \
        --instance-workers "${RULES_WORKERS}" \
        ${MAX_ARGS[@]+"${MAX_ARGS[@]}"} \
        --output-csv "${OUT_RULES}" 2>&1 | tee "${LOG_DIR}/rules_${RUN_STAMP}.log"
fi

# ---------------- stage 2: GA random keys (slowest; pop x gen x instances) ----
if [[ "${RUN_GA}" == "1" ]] && stage_ready "ga" "${OUT_GA}"; then
    echo "[run ] ga (pop=${GA_POPULATION} x gen=${GA_GENERATIONS}) -> ${OUT_GA}"
    "${PYTHON}" -m scripts.run_ga \
        --data-root "${DATA_ROOT}" \
        --suites "${SUITES}" \
        --seed "${SEED}" \
        --instance-workers "${GA_WORKERS}" \
        --population "${GA_POPULATION}" \
        --generations "${GA_GENERATIONS}" \
        ${MAX_ARGS[@]+"${MAX_ARGS[@]}"} \
        --output-csv "${OUT_GA}" 2>&1 | tee "${LOG_DIR}/ga_${RUN_STAMP}.log"
fi

# ---------------- stage 3: GPHH (train on the protocol's train split) --------
if [[ "${RUN_GPHH}" == "1" ]]; then
    if stage_ready "gphh" "${OUT_GPHH_DIR}/eval_summary.csv"; then
        echo "[run ] gphh -> ${OUT_GPHH_DIR}"
        "${PYTHON}" -m scripts.run_gphh \
            --data-root "${DATA_ROOT}" \
            --splits "${SPLITS_PATH}" \
            --train-instances "${GPHH_TRAIN_INSTANCES}" \
            --seed "${SEED}" \
            --eval-suites "${SUITES}" \
            --eval-workers "${GPHH_WORKERS}" \
            ${MAX_ARGS[@]+"${MAX_ARGS[@]}"} \
            --output-dir "${OUT_GPHH_DIR}" 2>&1 | tee "${LOG_DIR}/gphh_${RUN_STAMP}.log"
    fi
fi

# ---------------- step 4: merge everything into one comparison table ----------
if [[ "${RUN_AGGREGATE}" == "1" ]]; then
    AGG_GPHH_ARGS=()
    if [[ -f "${OUT_GPHH_DIR}/eval_summary.csv" ]]; then
        AGG_GPHH_ARGS=(--gphh "${OUT_GPHH_DIR}/eval_summary.csv")
    else
        echo "[warn] gphh eval_summary not found -> merged table will have no gphh column"
        echo "       (run with RUN_GPHH=1 first to produce it)"
    fi
    AGG_PPO_ARGS=()
    if [[ "${WITH_PPO}" == "1" ]]; then
        if [[ ! -f "${PPO_SUMMARY}" ]]; then
            echo "WITH_PPO=1 but summary not found: ${PPO_SUMMARY}" >&2
            echo "run step 3 first (train_a800.sh / train_cpu.sh) or fix PPO_SUMMARY" >&2
            exit 1
        fi
        AGG_PPO_ARGS=(--ppo "${PPO_SUMMARY}")
        echo "[run ] aggregate (rules+ga+gphh+ppo) -> ${OUT_AGG}"
    else
        echo "[run ] aggregate (rules+ga+gphh, no PPO yet) -> ${OUT_AGG}"
    fi
    AGG_PARAMS_ARGS=()
    if [[ -f "${OUT_STATS}/instances.csv" ]]; then
        AGG_PARAMS_ARGS=(--params "${OUT_STATS}/instances.csv")
    else
        echo "[warn] instance_stats output missing -> summary_by_regime.csv will be skipped"
        echo "       (stage 0 writes it; enable with RUN_STATS=1 and rerun)"
    fi
    "${PYTHON}" -m scripts.aggregate_results \
        --rules "${OUT_RULES}" \
        --ga "${OUT_GA}" \
        --bks "${BKS_JSON}" \
        --out-dir "${OUT_AGG}" \
        ${AGG_GPHH_ARGS[@]+"${AGG_GPHH_ARGS[@]}"} \
        ${AGG_PPO_ARGS[@]+"${AGG_PPO_ARGS[@]}"} \
        ${AGG_PARAMS_ARGS[@]+"${AGG_PARAMS_ARGS[@]}"} 2>&1 | tee "${LOG_DIR}/aggregate_${RUN_STAMP}.log"

    echo "== done =="
    echo "merged table : ${OUT_AGG}/merged_detail.csv"
    echo "suite summary: ${OUT_AGG}/summary_by_suite.csv"
    if [[ -f "${OUT_AGG}/summary_by_regime.csv" ]]; then
        echo "regime table : ${OUT_AGG}/summary_by_regime.csv (suite x RF x RS quartile)"
    fi
    echo "re-run with WITH_PPO=1 after PPO training to add the ppo_makespan column"
fi
