#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/Users/jef/Desktop/CS254-project/254-Project}"
IMAGE="${VORTEX_DOCKER_IMAGE:-vortex-dev:latest}"
CONTAINER_ROOT="${CONTAINER_ROOT:-/root/254-Project}"
CONTAINER_VORTEX="${CONTAINER_ROOT}/vortex"
CONTAINER_TOOLS="${CONTAINER_ROOT}/vortex-tools"
CONTAINER_PATH="${CONTAINER_TOOLS}/verilator/bin:${CONTAINER_TOOLS}/llvm-vortex/bin:${CONTAINER_TOOLS}/riscv64-gnu-toolchain/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"

if [[ ! -d "${PROJECT_ROOT}/vortex" ]]; then
  echo "error: expected vortex repo at ${PROJECT_ROOT}/vortex" >&2
  exit 1
fi

if [[ $# -eq 0 ]]; then
  set -- bash
fi

tty_flags=(-i)
if [[ -t 0 && -t 1 ]]; then
  tty_flags=(-it)
fi

exec docker run --rm "${tty_flags[@]}" \
  -v "${PROJECT_ROOT}:${CONTAINER_ROOT}" \
  -w "${CONTAINER_VORTEX}" \
  -e VORTEX_HOME="${CONTAINER_VORTEX}" \
  -e TOOLDIR="${CONTAINER_TOOLS}" \
  -e XLEN=64 \
  -e PATH="${CONTAINER_PATH}" \
  "${IMAGE}" "$@"
