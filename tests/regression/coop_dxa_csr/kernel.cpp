#include <vx_spawn.h>
#include "common.h"

enum : uint32_t {
  kTileRows = 8,
  kGlobalStride = 512,
  kSlot0Offset = 0x80,
  kSlot0Size = 0x40,
  kSlot0Panels = 3,
  kSlot1Offset = 0x180,
  kSlot1Size = 0x60,
  kSlot1Panels = 5,
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
                                 kSlot0Size + team_rank * 8,
                                 kTileRows,
                                 kGlobalStride,
                                 kSlot0Panels);
  vx_team_set_dxa_stream_slot_2d(1,
                                 slot1_global,
                                 kSlot1Offset + team_rank * 4,
                                 kSlot1Size + team_rank * 8,
                                 kTileRows,
                                 kGlobalStride,
                                 kSlot1Panels);

  uint32_t csr_team_id = csr_read(VX_CSR_TEAM_ID);
  uint32_t csr_team_rank = csr_read(VX_CSR_TEAM_RANK);
  uint32_t csr_team_size = csr_read(VX_CSR_TEAM_SIZE);
  uint32_t csr_tile_rows = csr_read(VX_CSR_TEAM_TILE_ROWS);
  uint32_t csr_global_stride = csr_read(VX_CSR_TEAM_GLOBAL_STRIDE);
  uint32_t csr_slot0_offset = csr_read(VX_CSR_TEAM_SRC_OFFSET);
  uint32_t csr_slot0_size = csr_read(VX_CSR_TEAM_COPY_SIZE);
  uint32_t csr_slot0_mask = csr_read(VX_CSR_TEAM_DST_MASK);
  uint32_t csr_slot0_mode = csr_read(VX_CSR_TEAM_COPY_MODE);
  uint64_t csr_slot0_global = csr_read(VX_CSR_TEAM_GLOBAL_ADDR);
  uint32_t csr_slot1_offset = csr_read(VX_CSR_TEAM_SRC_OFFSET_1);
  uint32_t csr_slot1_size = csr_read(VX_CSR_TEAM_COPY_SIZE_1);
  uint32_t csr_slot1_mask = csr_read(VX_CSR_TEAM_DST_MASK_1);
  uint32_t csr_slot1_mode = csr_read(VX_CSR_TEAM_COPY_MODE_1);
  uint64_t csr_slot1_global = csr_read(VX_CSR_TEAM_GLOBAL_ADDR_1);

  uint32_t status = 1;
  status &= (csr_team_id == __team_id);
  status &= ((csr_team_rank & 0xffff) == __team_rank_x);
  status &= (((csr_team_rank >> 16) & 0xffff) == __team_rank_y);
  status &= ((csr_team_size & 0xffff) == __team_size_x);
  status &= (((csr_team_size >> 16) & 0xffff) == __team_size_y);
  status &= (csr_tile_rows == kTileRows);
  status &= (csr_global_stride == kGlobalStride);
  status &= (csr_slot0_offset == kSlot0Offset + team_rank * 4);
  status &= (csr_slot0_size == kSlot0Size + team_rank * 8);
  status &= (csr_slot0_mask == kSlot0Panels);
  status &= (csr_slot0_mode == VX_TEAM_COPY_MODE_DXA_STREAM);
  status &= (csr_slot0_global == slot0_global);
  status &= (csr_slot1_offset == kSlot1Offset + team_rank * 4);
  status &= (csr_slot1_size == kSlot1Size + team_rank * 8);
  status &= (csr_slot1_mask == kSlot1Panels);
  status &= (csr_slot1_mode == VX_TEAM_COPY_MODE_DXA_STREAM);
  status &= (csr_slot1_global == slot1_global);

  dst_ptr[block_linear_id()] = {
    csr_team_id,
    csr_team_rank & 0xffff,
    (csr_team_rank >> 16) & 0xffff,
    csr_team_size & 0xffff,
    (csr_team_size >> 16) & 0xffff,
    csr_tile_rows,
    csr_global_stride,
    csr_slot0_offset,
    csr_slot0_size,
    csr_slot0_mask,
    csr_slot0_mode,
    csr_slot0_global,
    csr_slot1_offset,
    csr_slot1_size,
    csr_slot1_mask,
    csr_slot1_mode,
    csr_slot1_global,
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
