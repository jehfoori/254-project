#!/usr/bin/env bash
set -euo pipefail

BASELINE_COMMIT="${BASELINE_COMMIT:-2f8d0ce26}"
DXA_COMMIT="${DXA_COMMIT:-069249d51}"
PROJECT_ROOT="${PROJECT_ROOT:-/Users/jef/Desktop/CS254-project/254-Project}"
VORTEX_REPO="${VORTEX_REPO:-${PROJECT_ROOT}/vortex}"
DOCKER_IMAGE="${YOSYS_DOCKER_IMAGE:-ubuntu:24.04}"
TMP_PARENT="${TMP_PARENT:-/private/tmp}"
RUN_ID="${RUN_ID:-$(date +%Y%m%d-%H%M%S)}"
TMP_ROOT="${TMP_ROOT:-$(mktemp -d "${TMP_PARENT}/vortex-yosys-area-${RUN_ID}.XXXXXX")}"
LOG_ROOT="${LOG_ROOT:-${VORTEX_REPO}/logs/yosys-area/${RUN_ID}}"

SYN_CONFIGS='-DDPI_DISABLE -DEXT_F_DISABLE -DNUM_WARPS=2 -DNUM_THREADS=2 -DGBAR_ENABLE'
SYN_PREFIX='build_yosys_area'
TOP='Vortex'

usage() {
  cat <<EOF
Usage: $(basename "$0") [--keep-worktrees] [--stream]

Runs a matched Yosys area proxy for:
  baseline: ${BASELINE_COMMIT}
  dxa:      ${DXA_COMMIT}

Environment overrides:
  BASELINE_COMMIT, DXA_COMMIT, PROJECT_ROOT, VORTEX_REPO,
  YOSYS_DOCKER_IMAGE, TMP_PARENT, TMP_ROOT, LOG_ROOT, RUN_ID
EOF
}

KEEP_WORKTREES=0
STREAM_LOG=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --keep-worktrees)
      KEEP_WORKTREES=1
      shift
      ;;
    --stream)
      STREAM_LOG=1
      shift
      ;;
    -h|--help)
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

mkdir -p "${LOG_ROOT}"
BASELINE_WT="${TMP_ROOT}/baseline"
DXA_WT="${TMP_ROOT}/dxa"
PROJECT_MOUNT="/root/254-Project"
TMP_MOUNT="/work"
CONTAINER_TOOLS="${PROJECT_MOUNT}/vortex-tools"

cleanup() {
  if [[ "${KEEP_WORKTREES}" -eq 0 ]]; then
    git -C "${VORTEX_REPO}" worktree remove --force "${BASELINE_WT}" >/dev/null 2>&1 || true
    git -C "${VORTEX_REPO}" worktree remove --force "${DXA_WT}" >/dev/null 2>&1 || true
  fi
}
trap cleanup EXIT

echo "==> Creating isolated worktrees under ${TMP_ROOT}"
git -C "${VORTEX_REPO}" worktree add --detach "${BASELINE_WT}" "${BASELINE_COMMIT}" >/dev/null
git -C "${VORTEX_REPO}" worktree add --detach "${DXA_WT}" "${DXA_COMMIT}" >/dev/null

cat > "${LOG_ROOT}/config.txt" <<EOF
baseline_commit=${BASELINE_COMMIT}
dxa_commit=${DXA_COMMIT}
docker_image=${DOCKER_IMAGE}
tmp_root=${TMP_ROOT}
project_root=${PROJECT_ROOT}
configs=${SYN_CONFIGS}
prefix=${SYN_PREFIX}
num_cores=4
EOF

docker_cmd=$(cat <<'EOF'
set -euo pipefail

export DEBIAN_FRONTEND=noninteractive
apt-get update
apt-get install -y --no-install-recommends \
  ca-certificates \
  build-essential \
  git \
  make \
  python3 \
  libffi8

TOOLS="/root/254-Project/vortex-tools"
BUNDLED_PATH="${TOOLS}/yosys/bin:${TOOLS}/sv2v/bin:${TOOLS}/verilator/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
FALLBACK_PATH="${TOOLS}/sv2v/bin:${TOOLS}/verilator/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"

export PATH="${BUNDLED_PATH}"
if yosys -V >/work/toolcheck-bundled-yosys.txt 2>&1; then
  echo "toolchain= bundled-yosys" | tee /work/toolchain.txt
else
  echo "bundled_yosys_failed:" > /work/toolcheck-bundled-yosys.txt
  "${TOOLS}/yosys/bin/yosys" -V >> /work/toolcheck-bundled-yosys.txt 2>&1 || true
  apt-get install -y --no-install-recommends yosys
  export PATH="${FALLBACK_PATH}"
  echo "toolchain= apt-yosys" | tee /work/toolchain.txt
fi

yosys -V | tee /work/yosys-version.txt
sv2v --version | head -n 1 | tee /work/sv2v-version.txt
verilator --version | tee /work/verilator-version.txt

run_one() {
  local name="$1"
  local dir="$2"
  echo "==> Running ${name} synthesis in ${dir}"
  cd "${dir}"
  ./configure --xlen=64 --tooldir=/root/254-Project/vortex-tools --prefix="${dir}/build"
  PREFIX=build_yosys_area make -C hw/syn/yosys clean
  PREFIX=build_yosys_area \
    NUM_CORES=4 \
    CONFIGS="-DDPI_DISABLE -DEXT_F_DISABLE -DNUM_WARPS=2 -DNUM_THREADS=2 -DGBAR_ENABLE" \
    make -C hw/syn/yosys synthesis
}

run_one baseline /work/baseline
run_one dxa /work/dxa
EOF
)

echo "==> Running synthesis in ${DOCKER_IMAGE}"
set +e
if [[ "${STREAM_LOG}" -eq 1 ]]; then
  docker run --rm -i \
    --platform linux/amd64 \
    -v "${PROJECT_ROOT}:${PROJECT_MOUNT}" \
    -v "${TMP_ROOT}:${TMP_MOUNT}" \
    -w "${TMP_MOUNT}" \
    "${DOCKER_IMAGE}" \
    bash -lc "${docker_cmd}" 2>&1 | tee "${LOG_ROOT}/docker-run.log"
  docker_status=${PIPESTATUS[0]}
else
  docker run --rm -i \
    --platform linux/amd64 \
    -v "${PROJECT_ROOT}:${PROJECT_MOUNT}" \
    -v "${TMP_ROOT}:${TMP_MOUNT}" \
    -w "${TMP_MOUNT}" \
    "${DOCKER_IMAGE}" \
    bash -lc "${docker_cmd}" >"${LOG_ROOT}/docker-run.log" 2>&1
  docker_status=$?
fi
set -e

cp -f "${TMP_ROOT}/toolchain.txt" "${LOG_ROOT}/toolchain.txt" 2>/dev/null || true
cp -f "${TMP_ROOT}/toolcheck-bundled-yosys.txt" "${LOG_ROOT}/toolcheck-bundled-yosys.txt" 2>/dev/null || true
cp -f "${TMP_ROOT}/yosys-version.txt" "${LOG_ROOT}/yosys-version.txt" 2>/dev/null || true
cp -f "${TMP_ROOT}/sv2v-version.txt" "${LOG_ROOT}/sv2v-version.txt" 2>/dev/null || true
cp -f "${TMP_ROOT}/verilator-version.txt" "${LOG_ROOT}/verilator-version.txt" 2>/dev/null || true

for name in baseline dxa; do
  src="${TMP_ROOT}/${name}/hw/syn/yosys/${SYN_PREFIX}_${TOP}/yosys.log"
  if [[ -f "${src}" ]]; then
    cp -f "${src}" "${LOG_ROOT}/${name}-yosys.log"
  fi
done

if [[ "${docker_status}" -ne 0 ]]; then
  echo "error: synthesis flow failed; logs saved under ${LOG_ROOT}" >&2
  echo "==> Last 80 lines of docker-run.log:" >&2
  tail -n 80 "${LOG_ROOT}/docker-run.log" >&2 || true
  exit "${docker_status}"
fi

python3 - "${LOG_ROOT}" <<'PY'
import pathlib
import re
import sys

log_root = pathlib.Path(sys.argv[1])

metrics = [
    "Number of wires",
    "Number of wire bits",
    "Number of public wires",
    "Number of public wire bits",
    "Number of memories",
    "Number of memory bits",
    "Number of processes",
    "Number of cells",
]

def parse(path):
    text = path.read_text(errors="replace")
    result = {}
    for metric in metrics:
        matches = re.findall(rf"{re.escape(metric)}:\s+([0-9]+)", text)
        if matches:
            result[metric] = int(matches[-1])
    cell_counts = {}
    in_cells = False
    for line in text.splitlines():
        if line.strip() == "Number of cells:" or re.match(r"\s*Number of cells:\s+\d+", line):
            in_cells = True
            continue
        if in_cells:
            m = re.match(r"\s+(\S+)\s+([0-9]+)\s*$", line)
            if m:
                cell_counts[m.group(1)] = int(m.group(2))
            elif line.startswith("===") or line.strip() == "":
                if cell_counts:
                    break
    result["cell_counts"] = cell_counts
    return result

baseline = parse(log_root / "baseline-yosys.log")
dxa = parse(log_root / "dxa-yosys.log")

def delta_line(metric):
    b = baseline.get(metric)
    d = dxa.get(metric)
    if b is None or d is None:
        return f"| {metric} | n/a | n/a | n/a | n/a |"
    diff = d - b
    pct = "n/a" if b == 0 else f"{(diff / b) * 100:.2f}%"
    return f"| {metric} | {b:,} | {d:,} | {diff:+,} | {pct} |"

cell_keys = sorted(set(baseline["cell_counts"]) | set(dxa["cell_counts"]))
top_cells = sorted(
    cell_keys,
    key=lambda k: max(baseline["cell_counts"].get(k, 0), dxa["cell_counts"].get(k, 0)),
    reverse=True,
)[:12]

summary = []
summary.append("# Yosys Area Proxy Summary\n")
summary.append("## Totals\n")
summary.append("| Metric | Baseline | DXA | Delta | Delta % |")
summary.append("|---|---:|---:|---:|---:|")
for metric in metrics:
    summary.append(delta_line(metric))
summary.append("\n## Major Cell Categories\n")
summary.append("| Cell | Baseline | DXA | Delta | Delta % |")
summary.append("|---|---:|---:|---:|---:|")
for cell in top_cells:
    b = baseline["cell_counts"].get(cell, 0)
    d = dxa["cell_counts"].get(cell, 0)
    diff = d - b
    pct = "n/a" if b == 0 else f"{(diff / b) * 100:.2f}%"
    summary.append(f"| `{cell}` | {b:,} | {d:,} | {diff:+,} | {pct} |")
summary.append("")
(log_root / "summary.md").write_text("\n".join(summary))
print("\n".join(summary))
PY

echo "==> Logs and summary saved under ${LOG_ROOT}"
