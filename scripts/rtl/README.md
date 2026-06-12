# RTL Workflow Helpers

This directory contains small wrappers for rebuilding RTLSIM and running the
SGEMM/DXA regressions used by the true DXA/TMA redo branch.

The current branch contains the SimX DXA path and the first RTL DXA data
movement engine. For the full TA-facing reproduction recipe, see
`REPRODUCTION.md` at the repo root.

## Docker Entry Point

Run commands inside the project Docker image:

```bash
./scripts/rtl/docker-vortex.sh bash
```

By default this mounts:

- host: `/Users/jef/Desktop/CS254-project/254-Project`
- container: `/root/254-Project`

and sets:

- `VORTEX_HOME=/root/254-Project/vortex`
- `TOOLDIR=/root/254-Project/vortex-tools`
- `XLEN=64`

## Build RTLSIM

Inside Docker:

```bash
./scripts/rtl/build-rtlsim.sh
```

From the host:

```bash
./scripts/rtl/docker-vortex.sh ./scripts/rtl/build-rtlsim.sh
```

The default config is:

```text
-DEXT_TCU_ENABLE -DNUM_CORES=4 -DGBAR_ENABLE
```

The script defaults to `--jobs 1` because that was the stable path during prior
RTLSIM/Verilator PCH issues. It does not clean by default, so generated objects
and PCH artifacts are preserved across iterations.

## Run Tests

Run the local-memory smoke test:

```bash
./scripts/rtl/docker-vortex.sh ./scripts/rtl/run-rtl-test.sh --test coop_lmem --no-build
```

Run the DXA panel-stream smoke test:

```bash
./scripts/rtl/docker-vortex.sh ./scripts/rtl/run-rtl-test.sh --test coop_dxa_panel_stream --no-build
```

Run stock tensor SGEMM:

```bash
./scripts/rtl/docker-vortex.sh ./scripts/rtl/run-rtl-test.sh \
  --test sgemm_tcu --m 32 --n 32 --k 32 --no-build
```

Run the local-memory tensor SGEMM baseline:

```bash
./scripts/rtl/docker-vortex.sh ./scripts/rtl/run-rtl-test.sh \
  --test sgemm_tcu --m 32 --n 32 --k 32 --local-baseline --no-build --clean
```

Run the cooperative DXA SGEMM path:

```bash
./scripts/rtl/docker-vortex.sh ./scripts/rtl/run-rtl-test.sh \
  --test coop_sgemm_tcu --m 32 --n 32 --k 32 \
  --coop-n-tiles 1 --panel-k-tiles 1 --no-build --clean
```

Use `--profile` to enable compile-time cooperative stats for
`coop_sgemm_tcu`. Stock `sgemm_tcu` does not have a runtime counter flag.

## Smoke Test

```bash
./scripts/rtl/docker-vortex.sh ./scripts/rtl/smoke-rtl.sh
```

This runs a short stock/local/cooperative sanity set. For final result tables,
use the commands in `REPRODUCTION.md`.

Logs are written under `logs/rtl/`, which is intentionally ignored by git.
