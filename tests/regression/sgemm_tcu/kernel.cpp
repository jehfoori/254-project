#include "common.h"
#include <vx_spawn.h>
#include <vx_tensor.h>

namespace vt = vortex::tensor;
using ctx = vt::wmma_context<NUM_THREADS, vt::ITYPE, vt::OTYPE>;

void kernel_body(kernel_arg_t *__UNIFORM__ arg) {
  auto pA = reinterpret_cast<ctx::input_t *>(arg->A_addr);
  auto pB = reinterpret_cast<ctx::input_t *>(arg->B_addr);
  auto pC = reinterpret_cast<ctx::output_t *>(arg->C_addr);
#if SGEMM_TCU_TIMING_STATS
  auto pStats = reinterpret_cast<timing_stats_t *>(arg->stats_addr);
  uint64_t timing_total_start = csr_read(VX_CSR_MCYCLE);
  uint64_t timing_stage_cycles = 0;
  uint64_t timing_operand_load_cycles = 0;
  uint64_t timing_mma_cycles = 0;
  uint64_t timing_post_iter_sync_cycles = 0;
  uint64_t timing_store_cycles = 0;
#endif

  uint32_t M = arg->M;
  uint32_t N = arg->N;
  uint32_t K = arg->K;

  ctx::fragment_a   fragA;
  ctx::fragment_b   fragB[SGEMM_TCU_N_TILES];
  ctx::fragment_acc fragC[SGEMM_TCU_N_TILES];

#if SGEMM_TCU_USE_LMEM
  static_assert(vt::ITYPE::bits >= 8, "SGEMM_TCU_USE_LMEM supports byte-or-larger input types");
#endif
  static_assert(SGEMM_TCU_N_TILES >= 1, "SGEMM_TCU_N_TILES must be at least one");
  static_assert(vt::ITYPE::bits >= 8 || SGEMM_TCU_N_TILES == 1,
                "SGEMM_TCU_N_TILES > 1 is only supported for byte-or-larger input types");
  constexpr uint32_t a_tile_elems = ctx::tileM * ctx::tileK;
  constexpr uint32_t b_strip_elems = ctx::tileK * ctx::tileN * SGEMM_TCU_N_TILES;
#if SGEMM_TCU_USE_LMEM
  auto local_A = reinterpret_cast<ctx::input_t *>(__local_mem((a_tile_elems + b_strip_elems) * sizeof(ctx::input_t)));
  auto local_B = local_A + a_tile_elems;
#endif

  // calculate tile row & column based on block index
  uint32_t tile_row = blockIdx.y * ctx::tileM;
  uint32_t tile_col = blockIdx.x * ctx::tileN * SGEMM_TCU_N_TILES;

  // Initialize accumulator tile to zero
  for (uint32_t n_tile = 0; n_tile < SGEMM_TCU_N_TILES; ++n_tile) {
    ctx::fill_fragment(fragC[n_tile], 0);
  }

  for (int i = 0; i < K; i += ctx::tileK) {
#if SGEMM_TCU_USE_LMEM
#if SGEMM_TCU_TIMING_STATS
    uint64_t timing_start = csr_read(VX_CSR_MCYCLE);
#endif
    for (uint32_t elem = threadIdx.x; elem < a_tile_elems; elem += blockDim.x) {
      uint32_t row = elem / ctx::tileK;
      uint32_t col = elem % ctx::tileK;
      local_A[elem] = pA[(tile_row + row) * K + i + col];
    }
    for (uint32_t elem = threadIdx.x; elem < b_strip_elems; elem += blockDim.x) {
      uint32_t row = elem / (ctx::tileN * SGEMM_TCU_N_TILES);
      uint32_t col = elem % (ctx::tileN * SGEMM_TCU_N_TILES);
      local_B[elem] = pB[(i + row) * N + tile_col + col];
    }
    __syncthreads();
#if SGEMM_TCU_TIMING_STATS
    timing_stage_cycles += csr_read(VX_CSR_MCYCLE) - timing_start;
    timing_start = csr_read(VX_CSR_MCYCLE);
#endif

    ctx::load_matrix_sync(fragA, local_A, ctx::tileK);
    for (uint32_t n_tile = 0; n_tile < SGEMM_TCU_N_TILES; ++n_tile) {
      ctx::load_matrix_sync(fragB[n_tile],
                            local_B + n_tile * ctx::tileN,
                            ctx::tileN * SGEMM_TCU_N_TILES);
    }
#if SGEMM_TCU_TIMING_STATS
    timing_operand_load_cycles += csr_read(VX_CSR_MCYCLE) - timing_start;
#endif
#else
    auto pTileA = pA + tile_row * K + i;

    // Load A tile
#if SGEMM_TCU_TIMING_STATS
    uint64_t timing_start = csr_read(VX_CSR_MCYCLE);
#endif
    ctx::load_matrix_sync(fragA, pTileA, K);

    for (uint32_t n_tile = 0; n_tile < SGEMM_TCU_N_TILES; ++n_tile) {
      uint32_t n_col = tile_col + n_tile * ctx::tileN;
      if constexpr (vt::ITYPE::bits < 8) {
        // For sub-byte matrix B must be in col-major format.
        auto pTileB = pB + n_col * K + i;
        ctx::load_matrix_sync<vt::col_major>(fragB[n_tile], pTileB, K);
      } else {
        auto pTileB = pB + i * N + n_col;
        ctx::load_matrix_sync(fragB[n_tile], pTileB, N);
      }
    }
#if SGEMM_TCU_TIMING_STATS
    timing_operand_load_cycles += csr_read(VX_CSR_MCYCLE) - timing_start;
#endif
#endif

    // Matrix multiply-accumulate: c += a * b
#if SGEMM_TCU_TIMING_STATS
    timing_start = csr_read(VX_CSR_MCYCLE);
#endif
    for (uint32_t n_tile = 0; n_tile < SGEMM_TCU_N_TILES; ++n_tile) {
      ctx::mma_sync(fragC[n_tile], fragA, fragB[n_tile], fragC[n_tile]);
    }
#if SGEMM_TCU_TIMING_STATS
    timing_mma_cycles += csr_read(VX_CSR_MCYCLE) - timing_start;
#endif
#if SGEMM_TCU_USE_LMEM
#if SGEMM_TCU_TIMING_STATS
    timing_start = csr_read(VX_CSR_MCYCLE);
#endif
    __syncthreads();
#if SGEMM_TCU_TIMING_STATS
    timing_post_iter_sync_cycles += csr_read(VX_CSR_MCYCLE) - timing_start;
#endif
#endif
  }

  // Store the computed C tile
  for (uint32_t n_tile = 0; n_tile < SGEMM_TCU_N_TILES; ++n_tile) {
#if SGEMM_TCU_TIMING_STATS
    uint64_t timing_start = csr_read(VX_CSR_MCYCLE);
#endif
    auto pTileC = pC + tile_row * N + tile_col + n_tile * ctx::tileN;
    ctx::store_matrix_sync(pTileC, fragC[n_tile], N);
#if SGEMM_TCU_TIMING_STATS
    timing_store_cycles += csr_read(VX_CSR_MCYCLE) - timing_start;
#endif
  }

#if SGEMM_TCU_TIMING_STATS
  uint32_t block_idx = blockIdx.x + blockIdx.y * gridDim.x;
  pStats[block_idx] = {
    csr_read(VX_CSR_MCYCLE) - timing_total_start,
    timing_stage_cycles,
    timing_operand_load_cycles,
    timing_mma_cycles,
    timing_post_iter_sync_cycles,
    timing_store_cycles,
  };
#endif
}

int main() {
  auto arg = (kernel_arg_t *)csr_read(VX_CSR_MSCRATCH);
  return vx_spawn_threads(2, arg->grid_dim, arg->block_dim, (vx_kernel_func_cb)kernel_body, arg);
}
