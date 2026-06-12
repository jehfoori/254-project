#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VORTEX_HOME="${VORTEX_HOME:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"
BUILD_DIR="${BUILD_DIR:-${VORTEX_HOME}/build}"
LOG_DIR="${LOG_DIR:-${VORTEX_HOME}/logs/rtl}"

TEST=""
M=32
N=32
K=32
COOP_N_TILES=1
PANEL_K_TILES=1
LOCAL_BASELINE=0
PROFILE=0
NO_BUILD=0
CLEAN=0
CONFIGS="-DEXT_TCU_ENABLE -DNUM_CORES=4 -DGBAR_ENABLE"

usage() {
  cat <<'USAGE'
Usage: scripts/rtl/run-rtl-test.sh --test NAME [options]

Tests:
  coop_lmem
  coop_dxa_csr
  coop_dxa_skeleton
  coop_dxa_lmem_write
  sgemm_tcu
  coop_sgemm_tcu

Options:
  --m N                       Matrix M dimension (default: 32)
  --n N                       Matrix N dimension (default: 32)
  --k N                       Matrix K dimension (default: 32)
  --coop-n-tiles N            COOP_N_TILES for coop_sgemm_tcu (default: 1)
  --panel-k-tiles N           PANEL_K_TILES for coop_sgemm_tcu (default: 1)
  --local-baseline            Use local-memory staging for sgemm_tcu
  --profile                   Enable compile-time cooperative profile stats
  --no-build                  Skip RTLSIM build step
  --clean                     Clean this regression before building/running
  --configs FLAGS             Override base CONFIGS
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --test)
      TEST="$2"
      shift 2
      ;;
    --m)
      M="$2"
      shift 2
      ;;
    --n)
      N="$2"
      shift 2
      ;;
    --k)
      K="$2"
      shift 2
      ;;
    --coop-n-tiles)
      COOP_N_TILES="$2"
      shift 2
      ;;
    --panel-k-tiles)
      PANEL_K_TILES="$2"
      shift 2
      ;;
    --local-baseline)
      LOCAL_BASELINE=1
      shift
      ;;
    --profile)
      PROFILE=1
      shift
      ;;
    --no-build)
      NO_BUILD=1
      shift
      ;;
    --clean)
      CLEAN=1
      shift
      ;;
    --configs)
      CONFIGS="$2"
      shift 2
      ;;
    --help|-h)
      usage
      exit 0
      ;;
    *)
      echo "error: unknown argument: $1" >&2
      usage >&2
      exit 1
      ;;
  esac
done

if [[ -z "${TEST}" ]]; then
  echo "error: --test is required" >&2
  usage >&2
  exit 1
fi

case "${TEST}" in
  coop_lmem|coop_dxa_csr|coop_dxa_skeleton|coop_dxa_lmem_write|sgemm_tcu|coop_sgemm_tcu)
    ;;
  *)
    echo "error: unsupported test: ${TEST}" >&2
    usage >&2
    exit 1
    ;;
esac

if [[ "${NO_BUILD}" -eq 0 ]]; then
  "${SCRIPT_DIR}/build-rtlsim.sh" --configs "${CONFIGS}"
else
  "${SCRIPT_DIR}/ensure-build.sh"
fi

test_dir="${BUILD_DIR}/tests/regression/${TEST}"
if [[ ! -d "${test_dir}" ]]; then
  echo "error: missing regression build directory: ${test_dir}" >&2
  exit 1
fi

run_configs="${CONFIGS}"
opts=""

case "${TEST}" in
  coop_lmem|coop_dxa_csr|coop_dxa_skeleton|coop_dxa_lmem_write)
    opts=""
    ;;
  sgemm_tcu)
    opts="-m${M} -n${N} -k${K}"
    if [[ "${LOCAL_BASELINE}" -eq 1 ]]; then
      run_configs="${run_configs} -DSGEMM_TCU_USE_LMEM=1"
    fi
    ;;
  coop_sgemm_tcu)
    opts="-m${M} -n${N} -k${K}"
    run_configs="${run_configs} -DCOOP_N_TILES=${COOP_N_TILES} -DPANEL_K_TILES=${PANEL_K_TILES}"
    if [[ "${PROFILE}" -eq 1 ]]; then
      run_configs="${run_configs} -DCOOP_PROFILE_STATS=1"
    fi
    ;;
esac

mkdir -p "${LOG_DIR}"
timestamp="$(date +%Y%m%d-%H%M%S)"
name="${TEST}-m${M}-n${N}-k${K}"
log_file="${LOG_DIR}/run-${name}-${timestamp}.log"

cd "${test_dir}"

config_stamp="${test_dir}/.rtl-config"
previous_configs=""
if [[ -f "${config_stamp}" ]]; then
  previous_configs="$(<"${config_stamp}")"
fi

if [[ "${CLEAN}" -eq 1 || "${previous_configs}" != "${run_configs}" ]]; then
  if [[ "${CLEAN}" -eq 1 ]]; then
    echo "cleaning ${TEST}"
  else
    echo "CONFIGS changed for ${TEST}; cleaning stale regression artifacts"
  fi
  make clean CONFIGS="${run_configs}"
fi

echo "running ${TEST}"
echo "CONFIGS=\"${run_configs}\""
if [[ -n "${opts}" ]]; then
  echo "OPTS=\"${opts}\""
fi

if make run-rtlsim CONFIGS="${run_configs}" OPTS="${opts}" 2>&1 | tee "${log_file}"; then
  printf '%s\n' "${run_configs}" > "${config_stamp}"
  echo "test complete"
  echo "log: ${log_file}"
else
  status=$?
  echo "test failed; log: ${log_file}" >&2
  exit "${status}"
fi
