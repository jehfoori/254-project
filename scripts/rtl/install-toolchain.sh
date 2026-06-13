#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VORTEX_REPO="$(cd "${SCRIPT_DIR}/../.." && pwd)"
PROJECT_ROOT="$(cd "${VORTEX_REPO}/.." && pwd)"
TOOLS_DIR="${VORTEX_TOOLS:-${TOOLDIR:-${PROJECT_ROOT}/vortex-tools}}"
IMAGE="${VORTEX_DOCKER_IMAGE:-vortex-dev:latest}"
PLATFORM="${VORTEX_DOCKER_PLATFORM:-linux/amd64}"
CONTAINER_ROOT="${CONTAINER_ROOT:-/root/254-Project}"
CONTAINER_VORTEX="${CONTAINER_ROOT}/vortex"
CONTAINER_TOOLS="${CONTAINER_ROOT}/vortex-tools"

usage() {
  cat <<'USAGE'
Usage: scripts/rtl/install-toolchain.sh [--tooldir PATH]

Installs the standard Vortex prebuilt toolchain into PATH by running Vortex's
toolchain installer inside the Linux Docker environment. This is needed on macOS
because Vortex's configure/toolchain installer only supports Linux OS targets.
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --tooldir)
      TOOLS_DIR="$2"
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

if [[ ! -d "${VORTEX_REPO}/.git" ]]; then
  echo "error: expected Vortex git repo at ${VORTEX_REPO}" >&2
  exit 1
fi

mkdir -p "${TOOLS_DIR}"

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
  "${IMAGE}" \
  bash -lc './configure --xlen=64 --tooldir="${TOOLDIR}" && ./ci/toolchain_install.sh --all'
