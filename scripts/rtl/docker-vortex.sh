#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VORTEX_REPO="$(cd "${SCRIPT_DIR}/../.." && pwd)"
if [[ -z "${PROJECT_ROOT:-}" ]]; then
  PROJECT_ROOT="$(cd "${VORTEX_REPO}/.." && pwd)"
fi
IMAGE="${VORTEX_DOCKER_IMAGE:-vortex-dev:latest}"
PLATFORM="${VORTEX_DOCKER_PLATFORM:-linux/amd64}"
CONTAINER_ROOT="${CONTAINER_ROOT:-/root/254-Project}"
CONTAINER_VORTEX="${CONTAINER_ROOT}/vortex"
CONTAINER_TOOLS="${CONTAINER_ROOT}/vortex-tools"
CONTAINER_PATH="${CONTAINER_TOOLS}/verilator/bin:${CONTAINER_TOOLS}/llvm-vortex/bin:${CONTAINER_TOOLS}/riscv64-gnu-toolchain/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"

if [[ ! -d "${PROJECT_ROOT}/vortex" ]]; then
  echo "error: expected vortex repo at ${PROJECT_ROOT}/vortex" >&2
  exit 1
fi
if [[ ! -d "${PROJECT_ROOT}/vortex-tools" ]]; then
  echo "error: expected vortex-tools at ${PROJECT_ROOT}/vortex-tools" >&2
  echo "hint: keep vortex/ and vortex-tools/ as sibling directories, or set PROJECT_ROOT explicitly" >&2
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
  --platform "${PLATFORM}" \
  -v "${PROJECT_ROOT}:${CONTAINER_ROOT}" \
  -w "${CONTAINER_VORTEX}" \
  -e VORTEX_HOME="${CONTAINER_VORTEX}" \
  -e TOOLDIR="${CONTAINER_TOOLS}" \
  -e XLEN=64 \
  -e PATH="${CONTAINER_PATH}" \
  "${IMAGE}" "$@"
