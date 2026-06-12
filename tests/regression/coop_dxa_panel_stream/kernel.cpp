#include <vx_spawn.h>
#include "common.h"

static inline uint32_t block_linear_id() {
  return blockIdx.x + blockIdx.y * gridDim.x + blockIdx.z * gridDim.x * gridDim.y;
}

static inline uint64_t panel_word(uint32_t slot, uint32_t offset, uint32_t word) {
  auto base = csr_read(VX_CSR_LOCAL_MEM_BASE) + VX_TEAM_PANEL_OFFSET
            + slot * kPanelSlotBytes + offset;
  auto* ptr = reinterpret_cast<volatile uint64_t*>(base);
  return ptr[word];
}

void kernel_body(kernel_arg_t* __UNIFORM__ arg) {
  auto* dst_ptr = reinterpret_cast<result_t*>(arg->dst_addr);

  if (threadIdx.x == 0 && vx_team_rank() == 0) {
    vx_team_clear_copy();
    vx_team_set_dxa_stream_slot_2d(0,
                                   arg->src_a_addr,
                                   kAOffset,
                                   kCopyBytes,
                                   kTileRows,
                                   kGlobalStride,
                                   kPanelCount);
    vx_team_set_dxa_stream_slot_2d(1,
                                   arg->src_b_addr,
                                   kBOffset,
                                   kCopyBytes,
                                   kTileRows,
                                   kGlobalStride,
                                   kPanelCount);
  }
  __syncthreads();

  vx_team_dxa_start();

  uint32_t errors = 0;
  uint64_t first_actual = 0;
  uint64_t first_expected = 0;

  for (uint32_t panel = 0; panel < kPanelCount; ++panel) {
    uint32_t slot = panel & 1u;
    vx_team_dxa_wait_slot(slot);

    uint32_t a_offset = kAOffset + __team_rank_y * kCopyBytes;
    uint32_t b_offset = kBOffset + __team_rank_x * kCopyBytes;

    for (uint32_t row = 0; row < kTileRows; ++row) {
      for (uint32_t beat = 0; beat < (kRowBytes / sizeof(uint64_t)); ++beat) {
        uint32_t word = row * (kRowBytes / sizeof(uint64_t)) + beat;
        uint64_t actual_a = panel_word(slot, a_offset, word);
        uint64_t expected_a = expected_value(0, panel, __team_rank_y, row, beat);
        if (actual_a != expected_a) {
          if (errors == 0) {
            first_actual = actual_a;
            first_expected = expected_a;
          }
          ++errors;
        }

        uint64_t actual_b = panel_word(slot, b_offset, word);
        uint64_t expected_b = expected_value(1, panel, __team_rank_x, row, beat);
        if (actual_b != expected_b) {
          if (errors == 0) {
            first_actual = actual_b;
            first_expected = expected_b;
          }
          ++errors;
        }
      }
    }
  }

  dst_ptr[block_linear_id()] = {
    __team_rank_x,
    __team_rank_y,
    errors,
    uint32_t(errors == 0),
    first_actual,
    first_expected,
  };
}

int main() {
  auto* arg = (kernel_arg_t*)csr_read(VX_CSR_MSCRATCH);
  return vx_spawn_cooperative_groups(3,
                                     arg->grid_dim,
                                     arg->block_dim,
                                     arg->team_dim[0],
                                     arg->team_dim[1],
                                     (vx_kernel_func_cb)kernel_body,
                                     arg);
}
