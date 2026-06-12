# Reproducing the DXA/TMA Results

This document describes how to reproduce the final hardware-owned DXA/TMA
results from this branch. The implementation keeps GEMM computation on the
existing tensor-core path and adds a DXA-style data movement engine that moves
panels from global memory into core-local panel storage.

## Repository State

Use the final merged branch:

```bash
git clone https://github.com/jehfoori/254-project.git vortex
cd vortex
git checkout master
```

If the final branch has not been merged yet, use:

```bash
git checkout true-dxa-tma-redo
```

The final implementation commit before merge was:

```text
069249d51 Add RTL DXA diagnostics and timing evidence
```

The final implementation includes:

- SimX hardware-owned DXA stream modeling.
- RTL DXA CSR descriptor plumbing.
- RTL `VX_team_dxa_engine.sv`.
- A dedicated DXA D-cache read requester.
- DXA-owned local-memory write ports into each core.
- Ping-pong panel slots.
- A 4-entry tagged DXA global-read window.
- Regressions for CSR, control, local-memory write, panel stream, and SGEMM.

## Required Layout

The workflow assumes the Vortex repo and `vortex-tools` are siblings:

```text
254-Project/
  vortex/
  vortex-tools/
```

From inside `vortex/`, the helper scripts auto-detect this parent directory.
No local path variable is needed for the normal case.

If you use a different layout, set `PROJECT_ROOT` to the directory containing
both `vortex/` and `vortex-tools/`:

```bash
export PROJECT_ROOT=/absolute/path/to/254-Project
```

## Required Tools

You need:

- Docker with the daemon running.
- The course/Vortex Docker image:

```text
vortex-dev:latest
```

- The Vortex tool bundle in:

```text
../vortex-tools
```

The helper scripts run commands inside Docker and set:

```text
VORTEX_HOME=/root/254-Project/vortex
TOOLDIR=/root/254-Project/vortex-tools
XLEN=64
```

## Enter The Docker Environment

From the Vortex repo:

```bash
./scripts/rtl/docker-vortex.sh bash
```

Most commands below can also be run directly from the host by prefixing them
with `./scripts/rtl/docker-vortex.sh`.

## Build RTLSIM

Build the RTL simulator:

```bash
./scripts/rtl/docker-vortex.sh ./scripts/rtl/build-rtlsim.sh --jobs 1
```

The default RTL config is:

```text
-DEXT_TCU_ENABLE -DNUM_CORES=4 -DGBAR_ENABLE
```

`--jobs 1` is intentional. It was the most stable setting for the Verilator/PCH
workflow on the development machine. The script preserves incremental build
artifacts unless `--clean` is passed.

Logs are written to:

```text
logs/rtl/
```

## RTL Correctness Smoke Tests

Run the DXA control/data-movement smoke tests:

```bash
./scripts/rtl/docker-vortex.sh ./scripts/rtl/run-rtl-test.sh --test coop_dxa_csr --no-build

./scripts/rtl/docker-vortex.sh ./scripts/rtl/run-rtl-test.sh --test coop_dxa_skeleton --no-build

./scripts/rtl/docker-vortex.sh ./scripts/rtl/run-rtl-test.sh --test coop_dxa_lmem_write --no-build

./scripts/rtl/docker-vortex.sh ./scripts/rtl/run-rtl-test.sh --test coop_dxa_panel_stream --no-build
```

These tests validate:

- Team/DXA CSR descriptor round-trip.
- DXA start/wait command handling.
- DXA-owned local-memory writes.
- Panel-scale global-memory reads into ping-pong local panel slots.

## RTL SGEMM Comparison Runs

The final RTL comparison uses three axes:

1. Stock direct-fragment `sgemm_tcu`.
2. Local-memory staged `sgemm_tcu`.
3. Cooperative RTL DXA `coop_sgemm_tcu`.

Use stats-off runs for performance numbers.

### K=16

```bash
./scripts/rtl/docker-vortex.sh ./scripts/rtl/run-rtl-test.sh \
  --test sgemm_tcu --m 32 --n 32 --k 16 --no-build --clean

./scripts/rtl/docker-vortex.sh ./scripts/rtl/run-rtl-test.sh \
  --test sgemm_tcu --m 32 --n 32 --k 16 --local-baseline --no-build --clean

./scripts/rtl/docker-vortex.sh ./scripts/rtl/run-rtl-test.sh \
  --test coop_sgemm_tcu --m 32 --n 32 --k 16 \
  --coop-n-tiles 1 --panel-k-tiles 1 --no-build --clean
```

Expected development-machine results:

| Path | Cycles | Instrs | Result |
|---|---:|---:|---|
| stock `sgemm_tcu` | 53,087 | 51,085 | PASSED |
| local-memory `sgemm_tcu` | 76,930 | 161,036 | PASSED |
| RTL DXA `coop_sgemm_tcu` | 75,806 | 59,392 | PASSED |

### K=32

```bash
./scripts/rtl/docker-vortex.sh ./scripts/rtl/run-rtl-test.sh \
  --test sgemm_tcu --m 32 --n 32 --k 32 --no-build --clean

./scripts/rtl/docker-vortex.sh ./scripts/rtl/run-rtl-test.sh \
  --test sgemm_tcu --m 32 --n 32 --k 32 --local-baseline --no-build --clean

./scripts/rtl/docker-vortex.sh ./scripts/rtl/run-rtl-test.sh \
  --test coop_sgemm_tcu --m 32 --n 32 --k 32 \
  --coop-n-tiles 1 --panel-k-tiles 1 --no-build --clean
```

Expected development-machine results:

| Path | Cycles | Instrs | Result |
|---|---:|---:|---|
| stock `sgemm_tcu` | 56,742 | 71,310 | PASSED |
| local-memory `sgemm_tcu` | 108,369 | 291,596 | PASSED |
| RTL DXA `coop_sgemm_tcu` | 83,951 | 79,360 | PASSED |

### K=64

```bash
./scripts/rtl/docker-vortex.sh ./scripts/rtl/run-rtl-test.sh \
  --test sgemm_tcu --m 32 --n 32 --k 64 --no-build --clean

./scripts/rtl/docker-vortex.sh ./scripts/rtl/run-rtl-test.sh \
  --test sgemm_tcu --m 32 --n 32 --k 64 --local-baseline --no-build --clean

./scripts/rtl/docker-vortex.sh ./scripts/rtl/run-rtl-test.sh \
  --test coop_sgemm_tcu --m 32 --n 32 --k 64 \
  --coop-n-tiles 1 --panel-k-tiles 2 --no-build --clean
```

Expected development-machine results:

| Path | Cycles | Instrs | Result |
|---|---:|---:|---|
| stock `sgemm_tcu` | 66,017 | 111,757 | FP drift: 1 tiny mismatch |
| local-memory `sgemm_tcu` | 172,138 | 552,716 | FP drift: 1 tiny mismatch |
| RTL DXA `coop_sgemm_tcu` | 146,873 | 178,856 | FP drift: max ULP 7 |

### K=128

```bash
./scripts/rtl/docker-vortex.sh ./scripts/rtl/run-rtl-test.sh \
  --test sgemm_tcu --m 32 --n 32 --k 128 --no-build --clean

./scripts/rtl/docker-vortex.sh ./scripts/rtl/run-rtl-test.sh \
  --test sgemm_tcu --m 32 --n 32 --k 128 --local-baseline --no-build --clean

./scripts/rtl/docker-vortex.sh ./scripts/rtl/run-rtl-test.sh \
  --test coop_sgemm_tcu --m 32 --n 32 --k 128 \
  --coop-n-tiles 1 --panel-k-tiles 4 --no-build --clean
```

Expected development-machine results:

| Path | Cycles | Instrs | Result |
|---|---:|---:|---|
| stock `sgemm_tcu` | 83,422 | 192,653 | FP drift: 10 tiny mismatches |
| local-memory `sgemm_tcu` | 299,547 | 1,074,956 | FP drift: 10 tiny mismatches |
| RTL DXA `coop_sgemm_tcu` | 194,764 | 284,336 | FP drift: max ULP 9 |

For K=64 and K=128, small floating-point drift appears in stock,
local-memory, and DXA runs. We treated this as RTLSIM tensor floating-point
precision sensitivity rather than DXA data corruption.

## Larger-M Scaling Point

```bash
./scripts/rtl/docker-vortex.sh ./scripts/rtl/run-rtl-test.sh \
  --test sgemm_tcu --m 64 --n 32 --k 32 --no-build --clean

./scripts/rtl/docker-vortex.sh ./scripts/rtl/run-rtl-test.sh \
  --test sgemm_tcu --m 64 --n 32 --k 32 --local-baseline --no-build --clean

./scripts/rtl/docker-vortex.sh ./scripts/rtl/run-rtl-test.sh \
  --test coop_sgemm_tcu --m 64 --n 32 --k 32 \
  --coop-n-tiles 1 --panel-k-tiles 1 --no-build --clean
```

Expected development-machine results:

| Shape | Path | Cycles | Instrs | Result |
|---|---|---:|---:|---|
| `m64 n32 k32` | stock `sgemm_tcu` | 75,638 | 124,942 | PASSED |
| `m64 n32 k32` | local-memory `sgemm_tcu` | 179,508 | 565,516 | PASSED |
| `m64 n32 k32` | RTL DXA `coop_sgemm_tcu` | 141,953 | 144,640 | PASSED |

## Timing Diagnostics

Timing runs add CSR reads and should not be used as clean performance results.
They are useful for attributing time.

Local-memory baseline timing:

```bash
./scripts/rtl/docker-vortex.sh ./scripts/rtl/run-rtl-test.sh \
  --test sgemm_tcu --m 32 --n 32 --k 32 \
  --local-baseline --timing --no-build --clean
```

Cooperative DXA timing:

```bash
./scripts/rtl/docker-vortex.sh ./scripts/rtl/run-rtl-test.sh \
  --test coop_sgemm_tcu --m 32 --n 32 --k 32 \
  --coop-n-tiles 1 --panel-k-tiles 1 \
  --timing --no-build --clean
```

Development-machine timing bucket results:

| Path | Stage / DXA wait cycles | Fragment-load cycles | MMA cycles | Store cycles |
|---|---:|---:|---:|---:|
| local-memory `sgemm_tcu` | 925,748 | 88,133 | 7,695 | 38,645 |
| cooperative DXA `coop_sgemm_tcu` | 25,387 | 39,078 | 4,993 | 11,968 |

## DXA Hardware Counter Diagnostic

Run:

```bash
./scripts/rtl/docker-vortex.sh ./scripts/rtl/run-rtl-test.sh \
  --test coop_dxa_panel_stream --profile --no-build --clean
```

Representative development-machine counters:

| Counter | Value |
|---|---:|
| commands | 1 |
| panels completed | 4 |
| D-cache read requests | 64 |
| D-cache read responses | 64 |
| local-memory writes | 128 |
| busy cycles | 3,350 |
| wait-slot cycles | 2,725 |
| stream cycles | 426 |
| overwrite-block cycles | 2,721 |
| D-cache request stall cycles | 6 |
| no-free-window cycles | 294 |
| response-wait cycles | 304 |
| local-memory fanout cycles | 256 |
| local-memory stall cycles | 0 |
| drain cycles | 58 |
| max pending reads | 4 |
| max ready backlog | 4 |

## SimX Final Results

The final SimX architecture was the hardware-owned DXA stream model. It is useful
as architectural evidence, while the RTL results above are the more concrete
implementation evidence.

Run the same regressions in the normal Vortex build tree with:

```text
CONFIGS="-DEXT_TCU_ENABLE -DNUM_CORES=4"
```

The final SimX speedups over matching local-memory SGEMM were:

| K | N tiles | Local-memory cycles | DXA stream cycles | Speedup |
|---:|---:|---:|---:|---:|
| 32 | 1 | 109,404 | 92,540 | 1.18x |
| 64 | 1 | 174,355 | 120,227 | 1.45x |
| 128 | 1 | 305,209 | 171,811 | 1.78x |
| 32 | 2 | 114,949 | 84,019 | 1.37x |
| 64 | 2 | 166,333 | 108,380 | 1.53x |
| 128 | 2 | 269,852 | 159,407 | 1.69x |
| 32 | 4 | 162,235 | 88,798 | 1.83x |
| 64 | 4 | 271,151 | 128,849 | 2.10x |
| 128 | 4 | 477,905 | 209,323 | 2.28x |

## Yosys Area Proxy Attempt

A helper script is included:

```bash
./scripts/rtl/run-yosys-area-proxy.sh
```

This attempts a matched Yosys generic synthesis comparison between:

```text
baseline: 2f8d0ce26
DXA:      069249d51
```

using:

```text
NUM_CORES=4
CONFIGS="-DDPI_DISABLE -DEXT_F_DISABLE -DNUM_WARPS=2 -DNUM_THREADS=2 -DGBAR_ENABLE"
```

On the development laptop, this did not produce final area numbers. The modern
Ubuntu 24.04/Yosys 0.33 flow elaborated the pre-DXA baseline and progressed deep
into synthesis, but was killed during ABC mapping after expanding a large
inferred memory netlist:

```text
Extracted 1181395 gates and 2230047 wires ...
Executing ABC.
Killed yosys -l yosys.log -s synth.ys
make: *** [Makefile:94: synthesis] Error 137
```

This is consistent with local/container resource exhaustion, not a DXA RTL
failure. Since the pre-DXA baseline failed before the DXA design was synthesized,
we do not report area, timing, or power numbers.

## Known Limitations

- The RTL validated path is `COOP_N_TILES=1`.
- `COOP_N_TILES=2` is not a final RTL result because it produced NaNs during
  widened-N testing.
- K=64 and K=128 have tiny FP drift in stock, local-memory, and DXA RTLSIM runs.
- The DXA engine is a data-movement engine only; tensor cores still own GEMM.
- Stock direct-fragment `sgemm_tcu` remains faster than the DXA path because it
  avoids local-memory staging entirely. The fairest DXA comparison is against
  local-memory staged SGEMM.
- Full area/timing/power evaluation remains future work.
