# RTL Workflow Helpers

This directory contains small wrappers for rebuilding RTLSIM and running the
SGEMM regressions used by the true DXA/TMA redo branch.

The current branch contains the clean SimX DXA path. These scripts are only
workflow scaffolding for the next RTL implementation pass; they do not add RTL
DXA engine logic by themselves.

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

`coop_lmem` uses the cooperative team CSRs beginning at `VX_CSR_TEAM_ID`.
On the clean SimX-only branch, stock RTL does not implement those CSRs yet, so
this test is expected to fail until the first RTL team/DXA CSR chunk lands.

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
  --coop-n-tiles 2 --panel-k-tiles 1 --no-build --clean
```

Use `--profile` to enable compile-time cooperative stats for
`coop_sgemm_tcu`. Stock `sgemm_tcu` does not have a runtime counter flag.

## Smoke Test

```bash
./scripts/rtl/docker-vortex.sh ./scripts/rtl/smoke-rtl.sh
```

This runs:

- stock `sgemm_tcu m32 n32 k32`
- local-memory `sgemm_tcu m32 n32 k32`

After RTL team/DXA CSR support is implemented, add `coop_lmem` and
`coop_sgemm_tcu` back to the smoke set as cooperative correctness checks.

Logs are written under `logs/rtl/`, which is intentionally ignored by git.
