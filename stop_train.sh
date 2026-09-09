#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${SCRIPT_DIR}"
SHUTDOWN_TIMEOUT="${SHUTDOWN_TIMEOUT:-30}"

if [[ ! "${SHUTDOWN_TIMEOUT}" =~ ^[0-9]+$ ]]; then
    echo "SHUTDOWN_TIMEOUT must be a non-negative integer" >&2
    exit 2
fi

mapfile -t candidates < <(
    pgrep -u "$(id -u)" -f 'python.*-m scripts\.train_ppo' || true
)

train_pids=()
for pid in "${candidates[@]}"; do
    [[ -r "/proc/${pid}/cmdline" ]] || continue
    process_root="$(readlink -f "/proc/${pid}/cwd" 2>/dev/null || true)"
    [[ "${process_root}" == "${PROJECT_ROOT}" ]] || continue
    train_pids+=("${pid}")
done

if ((${#train_pids[@]} == 0)); then
    echo "no background training process found for ${PROJECT_ROOT}"
    exit 0
fi

descendants_of() {
    local parent="$1"
    local child
    while read -r child; do
        [[ -n "${child}" ]] || continue
        descendants_of "${child}"
        echo "${child}"
    done < <(pgrep -P "${parent}" || true)
}

all_pids=("${train_pids[@]}")
for pid in "${train_pids[@]}"; do
    while read -r child; do
        [[ -n "${child}" ]] && all_pids+=("${child}")
    done < <(descendants_of "${pid}")
done
mapfile -t all_pids < <(printf '%s\n' "${all_pids[@]}" | sort -nu)

echo "stopping training PID(s): ${train_pids[*]}"
kill -TERM "${train_pids[@]}" 2>/dev/null || true

deadline=$((SECONDS + SHUTDOWN_TIMEOUT))
while ((SECONDS < deadline)); do
    survivors=()
    for pid in "${all_pids[@]}"; do
        kill -0 "${pid}" 2>/dev/null && survivors+=("${pid}")
    done
    if ((${#survivors[@]} == 0)); then
        echo "training stopped"
        exit 0
    fi
    sleep 1
done

survivors=()
for pid in "${all_pids[@]}"; do
    kill -0 "${pid}" 2>/dev/null && survivors+=("${pid}")
done
if ((${#survivors[@]} > 0)); then
    echo "forcing remaining PID(s) to stop: ${survivors[*]}"
    kill -KILL "${survivors[@]}" 2>/dev/null || true
fi
echo "training stopped"
