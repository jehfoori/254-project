#include "common.h"
#include <vx_spawn.h>
#include <vx_tensor.h>

namespace vt = vortex::tensor;
using ctx = vt::wmma_context<NUM_THREADS, vt::ITYPE, vt::OTYPE>;

static_assert(vt::ITYPE::bits == 16, "coop_sgemm_tcu currently assumes 16-bit tensor inputs");
static_assert(vt::OTYPE::bits == 32, "coop_sgemm_tcu currently assumes 32-bit tensor outputs");
static_assert(ctx::tileM == ctx::tileK, "coop_sgemm_tcu assumes tileM == tileK for shared multicast shape");
static_assert(PANEL_K_TILES >= 1, "PANEL_K_TILES must be at least 1");
static_assert(COOP_N_TILES >= 1, "COOP_N_TILES must be at least 1");
static_assert(COOP_M_TILES >= 1, "COOP_M_TILES must be at least 1");

#if COOP_PROFILE_STATS
#define COOP_STAT(stmt) do { stmt; } while (false)
#else
#define COOP_STAT(stmt) do {} while (false)
#endif

static inline uint32_t block_linear_id() {
  return blockIdx.x + blockIdx.y * gridDim.x;
}

void kernel_body(kernel_arg_t* __UNIFORM__ arg) {
  auto pC = reinterpret_cast<ctx::output_t*>(arg->C_addr);
#if COOP_PROFILE_STATS
  auto stats_ptr = reinterpret_cast<block_stats_t*>(arg->stats_addr);
#endif

  constexpr uint32_t a_tile_elems = ctx::tileM * ctx::tileK;
  constexpr uint32_t b_tile_elems = ctx::tileK * ctx::tileN;
  constexpr uint32_t a_tile_bytes = a_tile_elems * sizeof(ctx::input_t);
  constexpr uint32_t b_tile_bytes = b_tile_elems * sizeof(ctx::input_t);
  constexpr uint32_t n_strip_cols = COOP_N_TILES * ctx::tileN;
  constexpr uint32_t a_panel_elems = COOP_M_TILES * PANEL_K_TILES * a_tile_elems;
  constexpr uint32_t b_panel_elems = PANEL_K_TILES * ctx::tileK * n_strip_cols;
  constexpr uint32_t a_panel_bytes = a_panel_elems * sizeof(ctx::input_t);
  constexpr uint32_t b_panel_bytes = b_panel_elems * sizeof(ctx::input_t);
  constexpr uint32_t panel_slot_bytes = 2 * a_panel_bytes + 2 * b_panel_bytes;

  bool oracle_panel_mode = (arg->mode == 1);
  uint32_t N = arg->N;
  uint32_t K = arg->K;
  uint32_t tile_row = blockIdx.y * COOP_M_TILES * ctx::tileM;
  uint32_t tile_col = blockIdx.x * n_strip_cols;

#if !COOP_PROFILE_STATS && PANEL_K_TILES == 16 && COOP_M_TILES == 1 && COOP_N_TILES == 2
  if (!oracle_panel_mode && K == PANEL_K_TILES * ctx::tileK) {
    ctx::fragment_a fragA;
#if COOP_DIAG_SERIAL_N2
    ctx::fragment_b fragB;
    ctx::fragment_acc fragC;
#else
    ctx::fragment_b fragB0;
    ctx::fragment_b fragB1;
    ctx::fragment_acc fragC0;
    ctx::fragment_acc fragC1;

    ctx::fill_fragment(fragC0, 0);
    ctx::fill_fragment(fragC1, 0);
#endif

    if (threadIdx.x == 0) {
      vx_team_enable_panel();
    }
    __syncthreads();

    if (threadIdx.x == 0) {
      vx_team_clear_copy();
      if (__team_rank_x == 0) {
        uint64_t global_addr = arg->A_addr + sizeof(ctx::input_t) * (tile_row * K);
        uint32_t panel_offset = __team_rank_y * a_panel_bytes;
        vx_team_set_panel_slot_2d(0, global_addr, panel_offset, a_panel_bytes,
                                  ctx::tileM, K * sizeof(ctx::input_t));
      }
      if (__team_rank_y == 0) {
        uint32_t slot = (__team_rank_x == 0) ? 1 : 0;
        uint64_t global_addr = arg->B_addr + sizeof(ctx::input_t) * tile_col;
        uint32_t panel_offset = 2 * a_panel_bytes + __team_rank_x * b_panel_bytes;
        vx_team_set_panel_slot_2d(slot, global_addr, panel_offset, b_panel_bytes,
                                  PANEL_K_TILES * ctx::tileK, N * sizeof(ctx::input_t));
      }
    }
    __syncthreads();
    vx_team_barrier();

    auto panel_base = reinterpret_cast<ctx::input_t*>(csr_read(VX_CSR_LOCAL_MEM_BASE)
                                                     + VX_TEAM_PANEL_OFFSET);
    auto panel_A = panel_base + __team_rank_y * a_panel_elems;
    auto panel_B = panel_base + 2 * a_panel_elems + __team_rank_x * b_panel_elems;

#if COOP_DIAG_SERIAL_N2
    auto pTileC = pC + tile_row * N + tile_col;
    for (uint32_t n_tile = 0; n_tile < COOP_N_TILES; ++n_tile) {
      ctx::fill_fragment(fragC, 0);
      for (uint32_t panel_tile = 0; panel_tile < PANEL_K_TILES; ++panel_tile) {
        auto local_B = panel_B + panel_tile * ctx::tileK * n_strip_cols
                     + n_tile * ctx::tileN;
        auto local_A = panel_A + panel_tile * ctx::tileK;
        ctx::load_matrix_sync(fragB, local_B, n_strip_cols);
        ctx::load_matrix_sync(fragA, local_A, PANEL_K_TILES * ctx::tileK);
        ctx::mma_sync(fragC, fragA, fragB, fragC);
      }
      ctx::store_matrix_sync(pTileC + n_tile * ctx::tileN, fragC, N);
    }
#else
    for (uint32_t panel_tile = 0; panel_tile < PANEL_K_TILES; ++panel_tile) {
      auto local_B = panel_B + panel_tile * ctx::tileK * n_strip_cols;
      ctx::load_matrix_sync(fragB0, local_B, n_strip_cols);
      ctx::load_matrix_sync(fragB1, local_B + ctx::tileN, n_strip_cols);
      auto local_A = panel_A + panel_tile * ctx::tileK;
      ctx::load_matrix_sync(fragA, local_A, PANEL_K_TILES * ctx::tileK);
      ctx::mma_sync(fragC0, fragA, fragB0, fragC0);
      ctx::mma_sync(fragC1, fragA, fragB1, fragC1);
    }

    auto pTileC = pC + tile_row * N + tile_col;
    ctx::store_matrix_sync(pTileC, fragC0, N);
    ctx::store_matrix_sync(pTileC + ctx::tileN, fragC1, N);
#endif
    return;
  }
#endif

#if !COOP_PROFILE_STATS && PANEL_K_TILES == 16 && COOP_M_TILES == 1 && COOP_N_TILES == 1
  if (!oracle_panel_mode && K == PANEL_K_TILES * ctx::tileK) {
    ctx::fragment_a fragA;
    ctx::fragment_b fragB;
    ctx::fragment_acc fragC;

    ctx::fill_fragment(fragC, 0);

    if (threadIdx.x == 0) {
      vx_team_enable_panel();
    }
    __syncthreads();

    if (threadIdx.x == 0) {
      vx_team_clear_copy();
      if (__team_rank_x == 0) {
        uint64_t global_addr = arg->A_addr + sizeof(ctx::input_t) * (tile_row * K);
        uint32_t panel_offset = __team_rank_y * a_panel_bytes;
        vx_team_set_panel_slot_2d(0, global_addr, panel_offset, a_panel_bytes,
                                  ctx::tileM, K * sizeof(ctx::input_t));
      }
      if (__team_rank_y == 0) {
        uint32_t slot = (__team_rank_x == 0) ? 1 : 0;
        uint64_t global_addr = arg->B_addr + sizeof(ctx::input_t) * tile_col;
        uint32_t panel_offset = 2 * a_panel_bytes + __team_rank_x * b_panel_bytes;
        vx_team_set_panel_slot_2d(slot, global_addr, panel_offset, b_panel_bytes,
                                  PANEL_K_TILES * ctx::tileK, N * sizeof(ctx::input_t));
      }
    }
    __syncthreads();
    vx_team_barrier();

    auto panel_base = reinterpret_cast<ctx::input_t*>(csr_read(VX_CSR_LOCAL_MEM_BASE)
                                                     + VX_TEAM_PANEL_OFFSET);
    auto panel_A = panel_base + __team_rank_y * a_panel_elems;
    auto panel_B = panel_base + 2 * a_panel_elems + __team_rank_x * b_panel_elems;

    for (uint32_t panel_tile = 0; panel_tile < PANEL_K_TILES; ++panel_tile) {
      auto local_B = panel_B + panel_tile * ctx::tileK * n_strip_cols;
      auto local_A = panel_A + panel_tile * ctx::tileK;
      ctx::load_matrix_sync(fragB, local_B, n_strip_cols);
      ctx::load_matrix_sync(fragA, local_A, PANEL_K_TILES * ctx::tileK);
      ctx::mma_sync(fragC, fragA, fragB, fragC);
    }

    auto pTileC = pC + tile_row * N + tile_col;
    ctx::store_matrix_sync(pTileC, fragC, N);
    return;
  }
#endif

  ctx::fragment_a fragA;
  ctx::fragment_b fragB[COOP_N_TILES];
  ctx::fragment_acc fragC[COOP_M_TILES][COOP_N_TILES];

  for (uint32_t m_tile = 0; m_tile < COOP_M_TILES; ++m_tile) {
    for (uint32_t n_tile = 0; n_tile < COOP_N_TILES; ++n_tile) {
      ctx::fill_fragment(fragC[m_tile][n_tile], 0);
    }
  }

#if COOP_PROFILE_STATS
  uint32_t a_tile_loads = 0;
  uint32_t b_tile_loads = 0;
  uint32_t team_barriers = 0;
  uint32_t team_arrives = 0;
  uint32_t team_waits = 0;
  uint32_t transfer_events = 0;
  uint32_t global_bytes = 0;
  uint32_t fanout_bytes = 0;
  uint32_t panel_write_bytes = 0;
  uint32_t panel_read_bytes = 0;
  uint32_t panel_transfers = 0;
  uint32_t mma_steps = 0;
  uint32_t partial_panels = 0;
  uint32_t slot_reuse_barriers = 0;
#endif

  if (threadIdx.x == 0) {
    vx_team_enable_panel();
  }
  __syncthreads();

  auto program_panel = [&](uint32_t k_base, uint32_t buffer_slot, uint32_t panel_tiles) {
    uint32_t panel_base_offset = buffer_slot * panel_slot_bytes;
    vx_team_clear_copy();
    if (__team_rank_x == 0) {
      uint64_t global_addr = arg->A_addr + sizeof(ctx::input_t) * (tile_row * K + k_base);
      uint32_t copy_bytes = panel_tiles * COOP_M_TILES * a_tile_bytes;
      uint32_t panel_offset = panel_base_offset + __team_rank_y * a_panel_bytes;
      if (oracle_panel_mode) {
        vx_team_set_panel_oracle_slot_2d(0, global_addr, panel_offset, copy_bytes,
                                         COOP_M_TILES * ctx::tileM, K * sizeof(ctx::input_t));
      } else {
        vx_team_set_panel_slot_2d(0, global_addr, panel_offset, copy_bytes,
                                  COOP_M_TILES * ctx::tileM, K * sizeof(ctx::input_t));
      }
      COOP_STAT(a_tile_loads += panel_tiles * COOP_M_TILES);
      COOP_STAT(++transfer_events);
      COOP_STAT(++panel_transfers);
      COOP_STAT(global_bytes += copy_bytes);
      COOP_STAT(panel_write_bytes += copy_bytes);
    }
    if (__team_rank_y == 0) {
      uint32_t slot = (__team_rank_x == 0) ? 1 : 0;
      uint64_t global_addr = arg->B_addr + sizeof(ctx::input_t) * (k_base * N + tile_col);
      uint32_t copy_bytes = panel_tiles * ctx::tileK * n_strip_cols * sizeof(ctx::input_t);
      uint32_t panel_offset = panel_base_offset + 2 * a_panel_bytes + __team_rank_x * b_panel_bytes;
      if (oracle_panel_mode) {
        vx_team_set_panel_oracle_slot_2d(slot, global_addr, panel_offset, copy_bytes,
                                         panel_tiles * ctx::tileK, N * sizeof(ctx::input_t));
      } else {
        vx_team_set_panel_slot_2d(slot, global_addr, panel_offset, copy_bytes,
                                  panel_tiles * ctx::tileK, N * sizeof(ctx::input_t));
      }
      COOP_STAT(b_tile_loads += panel_tiles * COOP_N_TILES);
      COOP_STAT(++transfer_events);
      COOP_STAT(++panel_transfers);
      COOP_STAT(global_bytes += copy_bytes);
      COOP_STAT(panel_write_bytes += copy_bytes);
    }
  };

  uint32_t num_k_tiles = K / ctx::tileK;
  if (threadIdx.x == 0) {
    uint32_t panel_tiles = (num_k_tiles < PANEL_K_TILES) ? num_k_tiles : PANEL_K_TILES;
    program_panel(0, 0, panel_tiles);
  }
  __syncthreads();
  vx_team_barrier();
  COOP_STAT(++team_barriers);

  for (uint32_t panel_base_tile = 0; panel_base_tile < num_k_tiles; panel_base_tile += PANEL_K_TILES) {
    uint32_t panel_index = panel_base_tile / PANEL_K_TILES;
    uint32_t current_slot = panel_index & 1u;
    uint32_t panel_tiles = num_k_tiles - panel_base_tile;
    if (panel_tiles > PANEL_K_TILES)
      panel_tiles = PANEL_K_TILES;
    if (panel_tiles != PANEL_K_TILES)
      COOP_STAT(++partial_panels);

    uint32_t next_panel_tile = panel_base_tile + PANEL_K_TILES;
    if (next_panel_tile < num_k_tiles) {
      uint32_t next_slot = current_slot ^ 1u;
      uint32_t next_panel_tiles = num_k_tiles - next_panel_tile;
      if (next_panel_tiles > PANEL_K_TILES)
        next_panel_tiles = PANEL_K_TILES;
      if (next_panel_tile >= (2 * PANEL_K_TILES)) {
        vx_team_barrier();
        COOP_STAT(++team_barriers);
        COOP_STAT(++slot_reuse_barriers);
      }
      if (threadIdx.x == 0) {
        program_panel(next_panel_tile * ctx::tileK, next_slot, next_panel_tiles);
      }
      __syncthreads();
      vx_team_arrive();
      COOP_STAT(++team_arrives);
    }

    auto panel_base = reinterpret_cast<ctx::input_t*>(csr_read(VX_CSR_LOCAL_MEM_BASE)
                                                     + VX_TEAM_PANEL_OFFSET
                                                     + current_slot * panel_slot_bytes);
    auto panel_A = panel_base + __team_rank_y * a_panel_elems;
    auto panel_B = panel_base + 2 * a_panel_elems + __team_rank_x * b_panel_elems;
    uint32_t panel_a_ldm = panel_tiles * ctx::tileK;
    for (uint32_t panel_tile = 0; panel_tile < panel_tiles; ++panel_tile) {
      for (uint32_t n_tile = 0; n_tile < COOP_N_TILES; ++n_tile) {
        auto local_B = panel_B + panel_tile * ctx::tileK * n_strip_cols + n_tile * ctx::tileN;
        ctx::load_matrix_sync(fragB[n_tile], local_B, n_strip_cols);
        COOP_STAT(panel_read_bytes += b_tile_bytes);
      }
      for (uint32_t m_tile = 0; m_tile < COOP_M_TILES; ++m_tile) {
        auto local_A = panel_A + m_tile * ctx::tileM * panel_tiles * ctx::tileK
                             + panel_tile * ctx::tileK;
        ctx::load_matrix_sync(fragA, local_A, panel_a_ldm);
        COOP_STAT(panel_read_bytes += a_tile_bytes);
        for (uint32_t n_tile = 0; n_tile < COOP_N_TILES; ++n_tile) {
          ctx::mma_sync(fragC[m_tile][n_tile], fragA, fragB[n_tile], fragC[m_tile][n_tile]);
          COOP_STAT(++mma_steps);
        }
      }
    }

    if (next_panel_tile < num_k_tiles) {
      vx_team_wait();
      COOP_STAT(++team_waits);
    }
  }

  for (uint32_t m_tile = 0; m_tile < COOP_M_TILES; ++m_tile) {
    for (uint32_t n_tile = 0; n_tile < COOP_N_TILES; ++n_tile) {
      auto pTileC = pC + (tile_row + m_tile * ctx::tileM) * N
                       + tile_col + n_tile * ctx::tileN;
      ctx::store_matrix_sync(pTileC, fragC[m_tile][n_tile], N);
    }
  }

#if COOP_PROFILE_STATS
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
      transfer_events,
      global_bytes,
      fanout_bytes,
      panel_write_bytes,
      panel_read_bytes,
      panel_transfers,
      mma_steps,
      partial_panels,
      slot_reuse_barriers,
    };
  }
#endif
}

#undef COOP_STAT

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
