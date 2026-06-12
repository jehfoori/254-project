#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

"${SCRIPT_DIR}/build-rtlsim.sh"

"${SCRIPT_DIR}/run-rtl-test.sh" --test sgemm_tcu --m 32 --n 32 --k 32 --no-build
"${SCRIPT_DIR}/run-rtl-test.sh" --test sgemm_tcu --m 32 --n 32 --k 32 --local-baseline --no-build --clean
