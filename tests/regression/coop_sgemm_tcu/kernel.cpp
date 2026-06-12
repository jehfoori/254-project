#include "common.h"
#include <vx_spawn.h>
#include <vx_tensor.h>

namespace vt = vortex::tensor;
using ctx = vt::wmma_context<NUM_THREADS, vt::ITYPE, vt::OTYPE>;

static_assert(vt::ITYPE::bits == 16, "coop_sgemm_tcu currently assumes 16-bit tensor inputs");
static_assert(vt::OTYPE::bits == 32, "coop_sgemm_tcu currently assumes 32-bit tensor outputs");
static_assert(ctx::tileM == ctx::tileK, "coop_sgemm_tcu assumes tileM == tileK for panel layout");
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

  uint32_t N = arg->N;
  uint32_t K = arg->K;
  uint32_t tile_row = blockIdx.y * COOP_M_TILES * ctx::tileM;
  uint32_t tile_col = blockIdx.x * n_strip_cols;
  uint32_t team_base_row = tile_row - __team_rank_y * COOP_M_TILES * ctx::tileM;
  uint32_t team_base_col = tile_col - __team_rank_x * n_strip_cols;
  uint32_t num_k_tiles = K / ctx::tileK;

  ctx::fragment_a fragA;
  ctx::fragment_b fragB[COOP_N_TILES];
  ctx::fragment_acc fragC[COOP_M_TILES][COOP_N_TILES];

  for (uint32_t m_tile = 0; m_tile < COOP_M_TILES; ++m_tile) {
    for (uint32_t n_tile = 0; n_tile < COOP_N_TILES; ++n_tile) {
      ctx::fill_fragment(fragC[m_tile][n_tile], 0);
    }
  }

#if COOP_PROFILE_STATS
  uint32_t dxa_commands = 0;
  uint32_t dxa_a_panels = 0;
  uint32_t dxa_b_panels = 0;
  uint32_t team_barriers = 0;
  uint32_t dxa_stream_commands = 0;
  uint32_t dxa_slot_waits = 0;
  uint32_t dxa_global_bytes = 0;
  uint32_t baseline_global_bytes = 0;
  uint32_t avoided_global_bytes = 0;
  uint32_t panel_read_bytes = 0;
  uint32_t mma_steps = 0;
  uint32_t partial_panels = 0;
#endif

  if (threadIdx.x == 0) {
    vx_team_enable_panel();
  }
  __syncthreads();

  if (threadIdx.x == 0 && vx_team_rank() == 0) {
    uint32_t panel_count = num_k_tiles / PANEL_K_TILES;
    uint32_t a_copy_bytes = PANEL_K_TILES * COOP_M_TILES * a_tile_bytes;
    uint32_t b_copy_bytes = PANEL_K_TILES * ctx::tileK * n_strip_cols * sizeof(ctx::input_t);
    vx_team_clear_copy();
    vx_team_set_dxa_stream_slot_2d(0,
                                   arg->A_addr + sizeof(ctx::input_t) * (team_base_row * K),
                                   0,
                                   a_copy_bytes,
                                   COOP_M_TILES * ctx::tileM,
                                   K * sizeof(ctx::input_t),
                                   panel_count);
    vx_team_set_dxa_stream_slot_2d(1,
                                   arg->B_addr + sizeof(ctx::input_t) * team_base_col,
                                   2 * a_panel_bytes,
                                   b_copy_bytes,
                                   PANEL_K_TILES * ctx::tileK,
                                   N * sizeof(ctx::input_t),
                                   panel_count);
  }
  __syncthreads();
  vx_team_dxa_start();
  COOP_STAT(dxa_stream_commands += (vx_team_rank() == 0) ? 1 : 0);

  for (uint32_t panel_base_tile = 0; panel_base_tile < num_k_tiles; panel_base_tile += PANEL_K_TILES) {
    uint32_t panel_index = panel_base_tile / PANEL_K_TILES;
    uint32_t current_slot = panel_index & 1u;
    uint32_t panel_tiles = num_k_tiles - panel_base_tile;
    if (panel_tiles > PANEL_K_TILES)
      panel_tiles = PANEL_K_TILES;
    if (panel_tiles != PANEL_K_TILES)
      COOP_STAT(++partial_panels);

    vx_team_dxa_wait_slot(current_slot);
    COOP_STAT(++dxa_slot_waits);

    uint32_t a_copy_bytes = panel_tiles * COOP_M_TILES * a_tile_bytes;
    uint32_t b_copy_bytes = panel_tiles * ctx::tileK * n_strip_cols * sizeof(ctx::input_t);

    COOP_STAT(baseline_global_bytes += COOP_M_TILES * a_copy_bytes
                                     + COOP_N_TILES * panel_tiles * b_tile_bytes);
    if (vx_team_rank() == 0) {
      uint32_t team_dxa_bytes = __team_size_y * a_copy_bytes + __team_size_x * b_copy_bytes;
      uint32_t team_baseline_bytes = __team_size_x * __team_size_y
                                   * (COOP_M_TILES * a_copy_bytes
                                    + COOP_N_TILES * panel_tiles * b_tile_bytes);
      COOP_STAT(dxa_commands += (panel_index == 0) ? 1 : 0);
      COOP_STAT(dxa_a_panels += __team_size_y);
      COOP_STAT(dxa_b_panels += __team_size_x);
      COOP_STAT(dxa_global_bytes += team_dxa_bytes);
      COOP_STAT(avoided_global_bytes += team_baseline_bytes - team_dxa_bytes);
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
      dxa_commands,
      dxa_a_panels,
      dxa_b_panels,
      team_barriers,
      dxa_stream_commands,
      dxa_slot_waits,
      dxa_global_bytes,
      baseline_global_bytes,
      avoided_global_bytes,
      panel_read_bytes,
      mma_steps,
      partial_panels,
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
