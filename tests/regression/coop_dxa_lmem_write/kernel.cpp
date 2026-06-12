#include <vx_spawn.h>
#include "common.h"

enum : uint32_t {
  kScratchBytes = 16,
  kSlot0Word = 0,
  kSlot1Word = 2,
  kPanelCount = 1,
  kCopyBytes = sizeof(uint32_t),
};

static inline uint32_t block_linear_id() {
  return blockIdx.x + blockIdx.y * gridDim.x + blockIdx.z * gridDim.x * gridDim.y;
}

void kernel_body(kernel_arg_t* __UNIFORM__ arg) {
  auto* dst_ptr = reinterpret_cast<result_t*>(arg->dst_addr);
  auto* scratch = reinterpret_cast<volatile uint32_t*>(__local_mem(kScratchBytes));

  uint32_t slot0_offset = uint32_t(reinterpret_cast<uintptr_t>(&scratch[kSlot0Word])
                                - uintptr_t(csr_read(VX_CSR_LOCAL_MEM_BASE)));
  uint32_t slot1_offset = uint32_t(reinterpret_cast<uintptr_t>(&scratch[kSlot1Word])
                                - uintptr_t(csr_read(VX_CSR_LOCAL_MEM_BASE)));

  scratch[kSlot0Word] = 0;
  scratch[kSlot1Word] = 0;

  vx_team_set_dxa_stream_slot_2d(0,
                                 arg->src0_addr,
                                 slot0_offset,
                                 kCopyBytes,
                                 1,
                                 kCopyBytes,
                                 kPanelCount);
  vx_team_set_dxa_stream_slot_2d(1,
                                 arg->src1_addr,
                                 slot1_offset,
                                 kCopyBytes,
                                 1,
                                 kCopyBytes,
                                 kPanelCount);

  vx_team_dxa_start();
  vx_team_dxa_wait_slot(0);
  uint32_t slot0_value = scratch[kSlot0Word];
  vx_team_dxa_wait_slot(1);
  uint32_t slot1_value = scratch[kSlot1Word];

  uint32_t expected_slot0 = arg->expected_slot0;
  uint32_t expected_slot1 = arg->expected_slot1;

  dst_ptr[block_linear_id()] = {
    __team_rank_x,
    __team_rank_y,
    slot0_value,
    slot1_value,
    expected_slot0,
    expected_slot1,
    uint32_t(slot0_value == expected_slot0 && slot1_value == expected_slot1),
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
