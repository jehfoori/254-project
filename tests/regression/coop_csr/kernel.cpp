#include <vx_spawn.h>
#include "common.h"

static inline uint32_t block_linear_id() {
  return blockIdx.x + blockIdx.y * gridDim.x + blockIdx.z * gridDim.x * gridDim.y;
}

static inline uint32_t expected_accum(uint32_t a_offset, uint32_t b_offset,
                                      uint32_t k_tiles, uint32_t n_tiles) {
  uint32_t acc = 0;
  uint32_t k_idx = 0;
  uint32_t n_idx = 0;
  uint32_t base = (a_offset >> 2) + (b_offset >> 2);
  uint32_t total = k_tiles * n_tiles;
  for (uint32_t step = 0; step < total; ++step) {
    acc += base + k_idx + n_idx;
    if (k_idx + 1 < k_tiles) {
      ++k_idx;
    } else {
      k_idx = 0;
      ++n_idx;
    }
  }
  return acc;
}

void kernel_body(kernel_arg_t* __UNIFORM__ arg) {
  auto* dst_ptr = reinterpret_cast<result_t*>(arg->dst_addr);

  uint32_t block_id = block_linear_id();
  uint32_t rank_x = __team_rank_x;
  uint32_t rank_y = __team_rank_y;
  uint32_t rank = vx_team_rank();

  uint32_t a_offset = 0x100 + 0x20 * rank;
  uint32_t b_offset = 0x200 + 0x20 * rank;
  uint32_t c_addr_lo = 0x3000 + 0x40 * rank;
  uint32_t k_tiles = 3;
  uint32_t n_tiles = rank + 1;

  vx_team_tensor_status_select(0);
  uint32_t idle_status = vx_team_tensor_status();

  vx_team_tensor_mma_panel(a_offset,
                           b_offset,
                           c_addr_lo,
                           16,
                           8,
                           8,
                           k_tiles,
                           n_tiles);

  uint32_t done_status = 0;
  for (uint32_t spin = 0; spin < 1024; ++spin) {
    vx_team_tensor_status_select(0);
    done_status = vx_team_tensor_status();
    if ((done_status & 0x1u) != 0) {
      break;
    }
  }

  vx_team_tensor_status_select(1);
  uint32_t remaining_cycles = vx_team_tensor_status();
  vx_team_tensor_status_select(2);
  uint32_t observed_k_tiles = vx_team_tensor_status();
  vx_team_tensor_status_select(3);
  uint32_t observed_n_tiles = vx_team_tensor_status();
  vx_team_tensor_status_select(4);
  uint32_t observed_a_offset = vx_team_tensor_status();
  vx_team_tensor_status_select(5);
  uint32_t observed_b_offset = vx_team_tensor_status();
  vx_team_tensor_status_select(6);
  uint32_t observed_c_addr_lo = vx_team_tensor_status();
  vx_team_tensor_status_select(7);
  uint32_t observed_state = vx_team_tensor_status();
  vx_team_tensor_status_select(8);
  uint32_t observed_compute_steps_done = vx_team_tensor_status();
  vx_team_tensor_status_select(9);
  uint32_t observed_compute_steps_total = vx_team_tensor_status();
  vx_team_tensor_status_select(10);
  uint32_t observed_writeback_count = vx_team_tensor_status();
  vx_team_tensor_status_select(11);
  uint32_t observed_writeback_signature = vx_team_tensor_status();
  vx_team_tensor_status_select(12);
  uint32_t observed_accum = vx_team_tensor_status();
  vx_team_tensor_status_select(13);
  uint32_t observed_k_idx = vx_team_tensor_status();
  vx_team_tensor_status_select(14);
  uint32_t observed_n_idx = vx_team_tensor_status();
  vx_team_tensor_status_select(15);
  uint32_t observed_write_addr = vx_team_tensor_status();
  vx_team_tensor_status_select(16);
  uint32_t observed_write_data = vx_team_tensor_status();

  uint32_t expected_compute_steps = k_tiles * n_tiles;
  uint32_t expected_acc = expected_accum(a_offset, b_offset, k_tiles, n_tiles);
  uint32_t expected_write = expected_acc ^ expected_compute_steps ^ n_tiles;

  uint32_t ok = 1;
  ok &= (idle_status == 0);
  ok &= ((done_status & 0x1u) != 0);
  ok &= ((done_status & 0x2u) == 0);
  ok &= (remaining_cycles == 0);
  ok &= (observed_k_tiles == k_tiles);
  ok &= (observed_n_tiles == n_tiles);
  ok &= (observed_a_offset == a_offset);
  ok &= (observed_b_offset == b_offset);
  ok &= (observed_c_addr_lo == c_addr_lo);
  ok &= (observed_state == 4);
  ok &= (observed_compute_steps_done == expected_compute_steps);
  ok &= (observed_compute_steps_total == expected_compute_steps);
  ok &= (observed_writeback_count == 1);
  ok &= (observed_writeback_signature == expected_write);
  ok &= (observed_accum == expected_acc);
  ok &= (observed_k_idx == 0);
  ok &= (observed_n_idx == n_tiles - 1);
  ok &= (observed_write_addr == c_addr_lo);
  ok &= (observed_write_data == expected_write);

  result_t result = {
    __team_id,
    rank_x,
    rank_y,
    idle_status,
    done_status,
    remaining_cycles,
    observed_k_tiles,
    observed_n_tiles,
    observed_a_offset,
    observed_b_offset,
    observed_c_addr_lo,
    observed_state,
    observed_compute_steps_done,
    observed_compute_steps_total,
    observed_writeback_count,
    observed_writeback_signature,
    observed_accum,
    observed_k_idx,
    observed_n_idx,
    observed_write_addr,
    observed_write_data,
    ok,
  };

  dst_ptr[block_id] = result;
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
