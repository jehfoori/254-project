#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VORTEX_REPO="$(cd "${SCRIPT_DIR}/../.." && pwd)"
if [[ -z "${PROJECT_ROOT:-}" ]]; then
  PROJECT_ROOT="$(cd "${VORTEX_REPO}/.." && pwd)"
fi
if [[ -n "${VORTEX_TOOLS:-}" ]]; then
  TOOLS_DIR="${VORTEX_TOOLS}"
elif [[ -n "${TOOLDIR:-}" ]]; then
  TOOLS_DIR="${TOOLDIR}"
elif [[ -d "${PROJECT_ROOT}/vortex-tools" ]]; then
  TOOLS_DIR="${PROJECT_ROOT}/vortex-tools"
else
  TOOLS_DIR="${HOME}/tools"
fi
IMAGE="${VORTEX_DOCKER_IMAGE:-vortex-dev:latest}"
PLATFORM="${VORTEX_DOCKER_PLATFORM:-linux/amd64}"
CONTAINER_ROOT="${CONTAINER_ROOT:-/root/254-Project}"
CONTAINER_VORTEX="${CONTAINER_ROOT}/vortex"
CONTAINER_TOOLS="${CONTAINER_ROOT}/vortex-tools"
CONTAINER_PATH="${CONTAINER_TOOLS}/verilator/bin:${CONTAINER_TOOLS}/llvm-vortex/bin:${CONTAINER_TOOLS}/riscv64-gnu-toolchain/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"

if [[ ! -d "${VORTEX_REPO}/.git" ]]; then
  echo "error: expected Vortex git repo at ${VORTEX_REPO}" >&2
  exit 1
fi
if [[ ! -d "${TOOLS_DIR}" ]]; then
  echo "error: expected Vortex toolchain directory at ${TOOLS_DIR}" >&2
  echo "hint: set VORTEX_TOOLS or TOOLDIR to the directory containing llvm-vortex/ and riscv64-gnu-toolchain/" >&2
  exit 1
fi
if [[ ! -x "${TOOLS_DIR}/llvm-vortex/bin/clang++" || ! -d "${TOOLS_DIR}/riscv64-gnu-toolchain" ]]; then
  echo "error: ${TOOLS_DIR} does not look like a Vortex toolchain directory" >&2
  echo "hint: it should contain llvm-vortex/ and riscv64-gnu-toolchain/" >&2
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
  -v "${VORTEX_REPO}:${CONTAINER_VORTEX}" \
  -v "${TOOLS_DIR}:${CONTAINER_TOOLS}" \
  -w "${CONTAINER_VORTEX}" \
  -e VORTEX_HOME="${CONTAINER_VORTEX}" \
  -e TOOLDIR="${CONTAINER_TOOLS}" \
  -e XLEN=64 \
  -e PATH="${CONTAINER_PATH}" \
  "${IMAGE}" "$@"
