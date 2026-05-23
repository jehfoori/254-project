#include <vx_spawn.h>
#include "common.h"

enum : uint32_t {
  kLocalWords = 4,
  kRowSlot = 0,
  kColSlot = 1,
};

static inline uint32_t block_linear_id() {
  return blockIdx.x + blockIdx.y * gridDim.x + blockIdx.z * gridDim.x * gridDim.y;
}

void kernel_body(kernel_arg_t* __UNIFORM__ arg) {
  auto* dst_ptr = reinterpret_cast<result_t*>(arg->dst_addr);
  auto* scratch = reinterpret_cast<volatile uint32_t*>(__local_mem(kLocalWords * sizeof(uint32_t)));

  uint32_t team_rank = vx_team_rank();
  uint32_t team_stride = __team_size_x;

  scratch[kRowSlot] = 0;
  scratch[kColSlot] = 0;

  if (__team_rank_x == 0) {
    scratch[kRowSlot] = 0x100 + team_rank;
    vx_team_set_copy(kRowSlot * sizeof(uint32_t), sizeof(uint32_t), 1u << (team_rank + 1));
  } else {
    vx_team_clear_copy();
  }
  vx_team_barrier();

  uint32_t row_value = scratch[kRowSlot];

  if (__team_rank_y == 0) {
    scratch[kColSlot] = 0x200 + team_rank;
    vx_team_set_copy(kColSlot * sizeof(uint32_t), sizeof(uint32_t), 1u << (team_rank + team_stride));
  } else {
    vx_team_clear_copy();
  }
  vx_team_barrier();

  uint32_t col_value = scratch[kColSlot];

  uint32_t expected_row = (__team_rank_x == 0) ? (0x100 + team_rank) : (0x100 + team_rank - 1);
  uint32_t expected_col = (__team_rank_y == 0) ? (0x200 + team_rank) : (0x200 + team_rank - team_stride);

  result_t result = {
    __team_id,
    __team_rank_x,
    __team_rank_y,
    row_value,
    col_value,
    uint32_t(row_value == expected_row && col_value == expected_col),
  };

  dst_ptr[block_linear_id()] = result;
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
