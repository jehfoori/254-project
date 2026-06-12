#include <vx_spawn.h>
#include "common.h"

enum : uint32_t {
  kTileRows = 8,
  kGlobalStride = 512,
  kSlot0Offset = 0x100,
  kSlot0Size = 0x40,
  kSlot0Panels = 2,
  kSlot1Offset = 0x180,
  kSlot1Size = 0x40,
  kSlot1Panels = 2,
};

static inline uint32_t block_linear_id() {
  return blockIdx.x + blockIdx.y * gridDim.x + blockIdx.z * gridDim.x * gridDim.y;
}

void kernel_body(kernel_arg_t* __UNIFORM__ arg) {
  auto* dst_ptr = reinterpret_cast<result_t*>(arg->dst_addr);

  uint32_t team_rank = vx_team_rank();
  uint64_t slot0_global = 0x100000000ull + uint64_t(team_rank) * 0x1000ull;
  uint64_t slot1_global = 0x200000000ull + uint64_t(team_rank) * 0x1000ull;

  vx_team_set_dxa_stream_slot_2d(0,
                                 slot0_global,
                                 kSlot0Offset + team_rank * 4,
                                 kSlot0Size,
                                 kTileRows,
                                 kGlobalStride,
                                 kSlot0Panels);
  vx_team_set_dxa_stream_slot_2d(1,
                                 slot1_global,
                                 kSlot1Offset + team_rank * 4,
                                 kSlot1Size,
                                 kTileRows,
                                 kGlobalStride,
                                 kSlot1Panels);

  vx_team_dxa_start();
  vx_team_dxa_wait_slot(0);
  vx_team_dxa_wait_slot(1);

  uint32_t slot0_mode = csr_read(VX_CSR_TEAM_COPY_MODE);
  uint32_t slot1_mode = csr_read(VX_CSR_TEAM_COPY_MODE_1);
  uint32_t slot0_panels = csr_read(VX_CSR_TEAM_DST_MASK);
  uint32_t slot1_panels = csr_read(VX_CSR_TEAM_DST_MASK_1);

  uint32_t status = 1;
  status &= (slot0_mode == VX_TEAM_COPY_MODE_DXA_STREAM);
  status &= (slot1_mode == VX_TEAM_COPY_MODE_DXA_STREAM);
  status &= (slot0_panels == kSlot0Panels);
  status &= (slot1_panels == kSlot1Panels);

  dst_ptr[block_linear_id()] = {
    __team_rank_x,
    __team_rank_y,
    slot0_mode,
    slot1_mode,
    slot0_panels,
    slot1_panels,
    status,
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
