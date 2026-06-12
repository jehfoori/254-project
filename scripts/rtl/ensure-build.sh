#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VORTEX_HOME="${VORTEX_HOME:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"
TOOLDIR="${TOOLDIR:-/root/254-Project/vortex-tools}"
BUILD_DIR="${BUILD_DIR:-${VORTEX_HOME}/build}"

required_paths=(
  "${BUILD_DIR}/config.mk"
  "${BUILD_DIR}/runtime/rtlsim/Makefile"
  "${BUILD_DIR}/tests/regression/coop_dxa_csr/Makefile"
  "${BUILD_DIR}/tests/regression/coop_dxa_panel_stream/Makefile"
  "${BUILD_DIR}/tests/regression/coop_lmem/Makefile"
  "${BUILD_DIR}/tests/regression/sgemm_tcu/Makefile"
  "${BUILD_DIR}/tests/regression/coop_sgemm_tcu/Makefile"
)

needs_configure=0
for path in "${required_paths[@]}"; do
  if [[ ! -e "${path}" ]]; then
    needs_configure=1
    break
  fi
done

if [[ "${needs_configure}" -eq 0 ]]; then
  echo "build tree is ready: ${BUILD_DIR}"
  exit 0
fi

mkdir -p "${BUILD_DIR}"
cd "${BUILD_DIR}"

echo "configuring Vortex build tree at ${BUILD_DIR}"
../configure \
  --xlen=64 \
  --tooldir="${TOOLDIR}" \
  --prefix="${BUILD_DIR}"
