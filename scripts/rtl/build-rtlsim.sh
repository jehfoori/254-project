#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VORTEX_HOME="${VORTEX_HOME:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"
BUILD_DIR="${BUILD_DIR:-${VORTEX_HOME}/build}"
LOG_DIR="${LOG_DIR:-${VORTEX_HOME}/logs/rtl}"
JOBS=1
CLEAN=0
CONFIGS="-DEXT_TCU_ENABLE -DNUM_CORES=4 -DGBAR_ENABLE"

usage() {
  cat <<'USAGE'
Usage: scripts/rtl/build-rtlsim.sh [--jobs N] [--clean] [--configs FLAGS]

Builds build/runtime/rtlsim while preserving incremental artifacts by default.
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --jobs|-j)
      JOBS="$2"
      shift 2
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

missing_deps=()
for dep in \
  "${VORTEX_HOME}/third_party/hardfloat/source/fNToRecFN.v" \
  "${VORTEX_HOME}/third_party/softfloat/source/include/softfloat.h" \
  "${VORTEX_HOME}/third_party/ramulator/src/base/base.h"; do
  if [[ ! -f "${dep}" ]]; then
    missing_deps+=("${dep}")
  fi
done

if [[ "${#missing_deps[@]}" -ne 0 ]]; then
  echo "error: missing Vortex third-party dependencies:" >&2
  printf '  %s\n' "${missing_deps[@]}" >&2
  echo "hint: run 'git submodule update --init --recursive' from the repo root" >&2
  exit 1
fi

echo "building third-party runtime dependencies"
make -C "${VORTEX_HOME}/third_party"

"${SCRIPT_DIR}/ensure-build.sh"

mkdir -p "${LOG_DIR}"
timestamp="$(date +%Y%m%d-%H%M%S)"
log_file="${LOG_DIR}/build-rtlsim-${timestamp}.log"

cd "${BUILD_DIR}/runtime/rtlsim"

if [[ "${CLEAN}" -eq 1 ]]; then
  echo "cleaning RTLSIM build artifacts"
  make clean CONFIGS="${CONFIGS}"
fi

echo "building RTLSIM with CONFIGS=\"${CONFIGS}\" jobs=${JOBS}"
if make -j"${JOBS}" CONFIGS="${CONFIGS}" 2>&1 | tee "${log_file}"; then
  echo "RTLSIM build complete"
  echo "log: ${log_file}"
else
  status=$?
  echo "RTLSIM build failed; log: ${log_file}" >&2
  exit "${status}"
fi
