#include "common.h"
#include <vx_spawn.h>
#include <vx_tensor.h>

namespace vt = vortex::tensor;
using ctx = vt::wmma_context<NUM_THREADS, vt::ITYPE, vt::OTYPE>;

static_assert(vt::ITYPE::bits == 16, "coop_sgemm_tcu currently assumes 16-bit tensor inputs");
static_assert(vt::OTYPE::bits == 32, "coop_sgemm_tcu currently assumes 32-bit tensor outputs");
static_assert(ctx::tileM == ctx::tileK, "coop_sgemm_tcu assumes tileM == tileK for shared multicast shape");

static inline uint32_t block_linear_id() {
  return blockIdx.x + blockIdx.y * gridDim.x;
}

void kernel_body(kernel_arg_t* __UNIFORM__ arg) {
  auto pA = reinterpret_cast<ctx::input_t*>(arg->A_addr);
  auto pB = reinterpret_cast<ctx::input_t*>(arg->B_addr);
  auto pC = reinterpret_cast<ctx::output_t*>(arg->C_addr);
  auto stats_ptr = reinterpret_cast<block_stats_t*>(arg->stats_addr);

  constexpr uint32_t a_tile_elems = ctx::tileM * ctx::tileK;
  constexpr uint32_t b_tile_elems = ctx::tileK * ctx::tileN;
  constexpr uint32_t a_tile_bytes = a_tile_elems * sizeof(ctx::input_t);
  constexpr uint32_t b_tile_bytes = b_tile_elems * sizeof(ctx::input_t);
  constexpr uint32_t tile_bytes = a_tile_bytes + b_tile_bytes;

  bool async_mode = (arg->mode != 0);
  auto local_ptr = reinterpret_cast<ctx::input_t*>(__local_mem((async_mode ? 2 : 1) * tile_bytes));
  ctx::input_t* a_tiles[2] = {
    local_ptr,
    local_ptr + (tile_bytes / sizeof(ctx::input_t)),
  };
  ctx::input_t* b_tiles[2] = {
    local_ptr + a_tile_elems,
    local_ptr + (tile_bytes / sizeof(ctx::input_t)) + a_tile_elems,
  };

  uint32_t N = arg->N;
  uint32_t K = arg->K;
  uint32_t team_rank = vx_team_rank();
  uint32_t tile_row = blockIdx.y * ctx::tileM;
  uint32_t tile_col = blockIdx.x * ctx::tileN;

  ctx::fragment_a   fragA;
  ctx::fragment_b   fragB;
  ctx::fragment_acc fragC;

  ctx::fill_fragment(fragC, 0);

  uint32_t a_tile_loads = 0;
  uint32_t b_tile_loads = 0;
  uint32_t team_barriers = 0;
  uint32_t team_arrives = 0;
  uint32_t team_waits = 0;

  if (threadIdx.x == 0) {
    vx_team_set_multicast_shape(ctx::tileK, K * sizeof(ctx::input_t));
  }
  __syncthreads();

  auto program_tiles = [&](uint32_t k_base, uint32_t buffer_slot) {
    uint32_t base_offset = async_mode ? (buffer_slot * tile_bytes) : 0;
    vx_team_clear_copy();
    if (__team_rank_x == 0) {
      uint64_t global_addr = arg->A_addr + sizeof(ctx::input_t) * (tile_row * K + k_base);
      vx_team_set_multicast_slot(0, global_addr, base_offset, a_tile_bytes,
                                 (1u << team_rank) | (1u << (team_rank + 1)));
      ++a_tile_loads;
    }
    if (__team_rank_y == 0) {
      uint32_t slot = (__team_rank_x == 0) ? 1 : 0;
      uint64_t global_addr = arg->B_addr + sizeof(ctx::input_t) * (k_base * N + tile_col);
      vx_team_set_multicast_slot(slot, global_addr, base_offset + a_tile_bytes, b_tile_bytes,
                                 (1u << team_rank) | (1u << (team_rank + __team_size_x)));
      ++b_tile_loads;
    }
  };

  if (async_mode) {
    if (threadIdx.x == 0) {
      program_tiles(0, 0);
    }
    __syncthreads();
    vx_team_barrier();
    ++team_barriers;
  }

  for (uint32_t k = 0; k < K; k += ctx::tileK) {
    uint32_t tile_index = k / ctx::tileK;
    uint32_t current_slot = async_mode ? (tile_index & 1u) : 0u;
    auto local_A = a_tiles[current_slot];
    auto local_B = b_tiles[current_slot];

    if (!async_mode) {
      if (threadIdx.x == 0) {
        program_tiles(k, 0);
      }
      __syncthreads();
      vx_team_barrier();
      ++team_barriers;
    } else {
      uint32_t next_k = k + ctx::tileK;
      if (next_k < K) {
        uint32_t next_slot = current_slot ^ 1u;
        if (threadIdx.x == 0) {
          program_tiles(next_k, next_slot);
        }
        __syncthreads();
        vx_team_arrive();
        ++team_arrives;
      }
    }

    ctx::load_matrix_sync(fragA, local_A, ctx::tileK);
    ctx::load_matrix_sync(fragB, local_B, ctx::tileN);
    ctx::mma_sync(fragC, fragA, fragB, fragC);

    if (async_mode && (k + ctx::tileK) < K) {
      vx_team_wait();
      ++team_waits;
    }
  }

  auto pTileC = pC + tile_row * N + tile_col;
  ctx::store_matrix_sync(pTileC, fragC, N);

  if (threadIdx.x == 0) {
    stats_ptr[block_linear_id()] = {
      __team_id,
      __team_rank_x,
      __team_rank_y,
      a_tile_loads,
      b_tile_loads,
      team_barriers,
      team_arrives,
      team_waits,
    };
  }
}

int main() {
  auto* arg = (kernel_arg_t*)csr_read(VX_CSR_MSCRATCH);
  return vx_spawn_cooperative_groups(2,
                                     arg->grid_dim,
                                     arg->block_dim,
                                     arg->team_dim[0],
                                     arg->team_dim[1],
                                     (vx_kernel_func_cb)kernel_body,
                                     arg);
}
