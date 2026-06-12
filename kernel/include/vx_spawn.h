// Copyright © 2019-2023
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef __VX_SPAWN_H__
#define __VX_SPAWN_H__

#include <vx_intrinsics.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef union {
  struct {
    uint32_t x;
    uint32_t y;
    uint32_t z;
  };
  uint32_t m[3];
} dim3_t;

extern __thread dim3_t blockIdx;
extern __thread dim3_t threadIdx;
extern dim3_t gridDim;
extern dim3_t blockDim;

extern __thread uint32_t __local_group_id;
extern uint32_t __warps_per_group;
extern __thread uint32_t __team_id;
extern __thread uint32_t __team_rank_x;
extern __thread uint32_t __team_rank_y;
extern __thread uint32_t __team_size_x;
extern __thread uint32_t __team_size_y;

typedef void (*vx_kernel_func_cb)(void *arg);

typedef void (*vx_serial_cb)(void *arg);

#define __local_mem(size) \
  (void*)((int8_t*)csr_read(VX_CSR_LOCAL_MEM_BASE) + __local_group_id * size)

#define __syncthreads() \
  vx_barrier(__local_group_id, __warps_per_group)

#define VX_TEAM_BARRIER_ID 0xC0000000u
#define VX_TEAM_ARRIVE_ID  0xC0000001u
#define VX_TEAM_WAIT_ID    0xC0000002u
#define VX_TEAM_DXA_START_ID      0xC0000003u
#define VX_TEAM_DXA_WAIT_SLOT0_ID 0xC0000004u
#define VX_TEAM_DXA_WAIT_SLOT1_ID 0xC0000005u
#define VX_TEAM_COPY_MODE_LOCAL 0u
#define VX_TEAM_COPY_MODE_GLOBAL 1u
#define VX_TEAM_COPY_MODE_PANEL 2u
#define VX_TEAM_COPY_MODE_DXA_STREAM 4u
#ifndef VX_CSR_TEAM_DXA_STAT_SEL
#define VX_CSR_TEAM_DXA_STAT_SEL 0xFD3
#endif
#ifndef VX_CSR_TEAM_DXA_STAT
#define VX_CSR_TEAM_DXA_STAT 0xFD4
#endif
// SimX reserves this local-memory window for team-shared panel reads.
#ifndef VX_TEAM_PANEL_OFFSET
#define VX_TEAM_PANEL_OFFSET 0x2000u
#endif
#ifndef VX_TEAM_PANEL_WINDOW
#define VX_TEAM_PANEL_WINDOW 0x10000u
#endif

#ifndef VX_CSR_TEAM_TENSOR_A_OFFSET
#define VX_CSR_TEAM_TENSOR_A_OFFSET 0xFD4
#endif
#ifndef VX_CSR_TEAM_TENSOR_B_OFFSET
#define VX_CSR_TEAM_TENSOR_B_OFFSET 0xFD5
#endif
#ifndef VX_CSR_TEAM_TENSOR_C_ADDR
#define VX_CSR_TEAM_TENSOR_C_ADDR 0xFD6
#endif
#ifndef VX_CSR_TEAM_TENSOR_C_STRIDE
#define VX_CSR_TEAM_TENSOR_C_STRIDE 0xFD7
#endif
#ifndef VX_CSR_TEAM_TENSOR_A_STRIDE
#define VX_CSR_TEAM_TENSOR_A_STRIDE 0xFD8
#endif
#ifndef VX_CSR_TEAM_TENSOR_B_STRIDE
#define VX_CSR_TEAM_TENSOR_B_STRIDE 0xFD9
#endif
#ifndef VX_CSR_TEAM_TENSOR_K_TILES
#define VX_CSR_TEAM_TENSOR_K_TILES 0xFDA
#endif
#ifndef VX_CSR_TEAM_TENSOR_RUN
#define VX_CSR_TEAM_TENSOR_RUN 0xFDB
#endif

inline uint32_t vx_team_rank() {
  return __team_rank_y * __team_size_x + __team_rank_x;
}

inline uint32_t vx_team_size() {
  return __team_size_x * __team_size_y;
}

inline void vx_team_set_copy_slot(uint32_t slot, uint32_t src_offset, uint32_t copy_size, uint32_t dst_mask) {
  if (slot == 0) {
    csr_write(VX_CSR_TEAM_SRC_OFFSET, src_offset);
    csr_write(VX_CSR_TEAM_COPY_SIZE, copy_size);
    csr_write(VX_CSR_TEAM_DST_MASK, dst_mask);
    csr_write(VX_CSR_TEAM_COPY_MODE, VX_TEAM_COPY_MODE_LOCAL);
    csr_write(VX_CSR_TEAM_GLOBAL_ADDR, 0);
  } else {
    csr_write(VX_CSR_TEAM_SRC_OFFSET_1, src_offset);
    csr_write(VX_CSR_TEAM_COPY_SIZE_1, copy_size);
    csr_write(VX_CSR_TEAM_DST_MASK_1, dst_mask);
    csr_write(VX_CSR_TEAM_COPY_MODE_1, VX_TEAM_COPY_MODE_LOCAL);
    csr_write(VX_CSR_TEAM_GLOBAL_ADDR_1, 0);
  }
}

inline void vx_team_set_multicast_shape(uint32_t tile_rows, uint32_t global_stride) {
  csr_write(VX_CSR_TEAM_TILE_ROWS, tile_rows);
  csr_write(VX_CSR_TEAM_GLOBAL_STRIDE, global_stride);
}

inline void vx_team_enable_panel() {
  csr_write(VX_CSR_TEAM_COPY_MODE, VX_TEAM_COPY_MODE_PANEL);
  csr_write(VX_CSR_TEAM_COPY_SIZE, 0);
}

inline void vx_team_set_multicast_slot(uint32_t slot,
                                       uint64_t global_addr,
                                       uint32_t dst_offset,
                                       uint32_t copy_size,
                                       uint32_t dst_mask) {
  if (slot == 0) {
    csr_write(VX_CSR_TEAM_SRC_OFFSET, dst_offset);
    csr_write(VX_CSR_TEAM_COPY_SIZE, copy_size);
    csr_write(VX_CSR_TEAM_DST_MASK, dst_mask);
    csr_write(VX_CSR_TEAM_COPY_MODE, VX_TEAM_COPY_MODE_GLOBAL);
    csr_write(VX_CSR_TEAM_GLOBAL_ADDR, global_addr);
  } else {
    csr_write(VX_CSR_TEAM_SRC_OFFSET_1, dst_offset);
    csr_write(VX_CSR_TEAM_COPY_SIZE_1, copy_size);
    csr_write(VX_CSR_TEAM_DST_MASK_1, dst_mask);
    csr_write(VX_CSR_TEAM_COPY_MODE_1, VX_TEAM_COPY_MODE_GLOBAL);
    csr_write(VX_CSR_TEAM_GLOBAL_ADDR_1, global_addr);
  }
}

inline void vx_team_set_panel_slot(uint32_t slot,
                                   uint64_t global_addr,
                                   uint32_t panel_offset,
                                   uint32_t copy_size) {
  if (slot == 0) {
    csr_write(VX_CSR_TEAM_SRC_OFFSET, panel_offset);
    csr_write(VX_CSR_TEAM_COPY_SIZE, copy_size);
    csr_write(VX_CSR_TEAM_DST_MASK, 1);
    csr_write(VX_CSR_TEAM_COPY_MODE, VX_TEAM_COPY_MODE_PANEL);
    csr_write(VX_CSR_TEAM_GLOBAL_ADDR, global_addr);
  } else {
    csr_write(VX_CSR_TEAM_SRC_OFFSET_1, panel_offset);
    csr_write(VX_CSR_TEAM_COPY_SIZE_1, copy_size);
    csr_write(VX_CSR_TEAM_DST_MASK_1, 1);
    csr_write(VX_CSR_TEAM_COPY_MODE_1, VX_TEAM_COPY_MODE_PANEL);
    csr_write(VX_CSR_TEAM_GLOBAL_ADDR_1, global_addr);
  }
}

inline void vx_team_set_panel_slot_2d(uint32_t slot,
                                      uint64_t global_addr,
                                      uint32_t panel_offset,
                                      uint32_t copy_size,
                                      uint32_t tile_rows,
                                      uint32_t global_stride) {
  vx_team_set_multicast_shape(tile_rows, global_stride);
  vx_team_set_panel_slot(slot, global_addr, panel_offset, copy_size);
}

inline void vx_team_set_dxa_stream_slot_2d(uint32_t slot,
                                           uint64_t global_addr,
                                           uint32_t panel_offset,
                                           uint32_t copy_size,
                                           uint32_t tile_rows,
                                           uint32_t global_stride,
                                           uint32_t panel_count) {
  vx_team_set_multicast_shape(tile_rows, global_stride);
  vx_team_set_panel_slot(slot, global_addr, panel_offset, copy_size);
  if (slot == 0) {
    csr_write(VX_CSR_TEAM_DST_MASK, panel_count);
    csr_write(VX_CSR_TEAM_COPY_MODE, VX_TEAM_COPY_MODE_DXA_STREAM);
  } else {
    csr_write(VX_CSR_TEAM_DST_MASK_1, panel_count);
    csr_write(VX_CSR_TEAM_COPY_MODE_1, VX_TEAM_COPY_MODE_DXA_STREAM);
  }
}

inline void vx_team_tensor_mma_panel(uint32_t a_panel_offset,
                                     uint32_t b_panel_offset,
                                     uint64_t c_addr,
                                     uint32_t c_stride,
                                     uint32_t a_stride,
                                     uint32_t b_stride,
                                     uint32_t k_tiles,
                                     uint32_t n_tiles) {
  csr_write(VX_CSR_TEAM_TENSOR_A_OFFSET, a_panel_offset);
  csr_write(VX_CSR_TEAM_TENSOR_B_OFFSET, b_panel_offset);
  csr_write(VX_CSR_TEAM_TENSOR_C_ADDR, c_addr);
  csr_write(VX_CSR_TEAM_TENSOR_C_STRIDE, c_stride);
  csr_write(VX_CSR_TEAM_TENSOR_A_STRIDE, a_stride);
  csr_write(VX_CSR_TEAM_TENSOR_B_STRIDE, b_stride);
  csr_write(VX_CSR_TEAM_TENSOR_K_TILES, k_tiles);
  csr_write(VX_CSR_TEAM_TENSOR_RUN, n_tiles);
}

inline void vx_team_set_copy(uint32_t src_offset, uint32_t copy_size, uint32_t dst_mask) {
  vx_team_set_copy_slot(0, src_offset, copy_size, dst_mask);
}

inline void vx_team_clear_copy() {
  vx_team_set_copy_slot(0, 0, 0, 0);
  vx_team_set_copy_slot(1, 0, 0, 0);
}

inline void vx_team_barrier() {
  vx_barrier(VX_TEAM_BARRIER_ID, vx_team_size());
}

inline void vx_team_arrive() {
  vx_barrier(VX_TEAM_ARRIVE_ID, vx_team_size());
}

inline void vx_team_wait() {
  vx_barrier(VX_TEAM_WAIT_ID, vx_team_size());
}

inline void vx_team_dxa_start() {
  vx_barrier(VX_TEAM_DXA_START_ID, vx_team_size());
}

inline void vx_team_dxa_wait_slot(uint32_t slot) {
  vx_barrier(slot == 0 ? VX_TEAM_DXA_WAIT_SLOT0_ID : VX_TEAM_DXA_WAIT_SLOT1_ID,
             vx_team_size());
}

enum {
  VX_TEAM_DXA_STAT_COMMANDS = 0,
  VX_TEAM_DXA_STAT_PANELS_COMPLETED = 1,
  VX_TEAM_DXA_STAT_DCACHE_READ_REQS = 2,
  VX_TEAM_DXA_STAT_DCACHE_READ_RSPS = 3,
  VX_TEAM_DXA_STAT_LMEM_WRITES = 4,
  VX_TEAM_DXA_STAT_BUSY_CYCLES = 5,
  VX_TEAM_DXA_STAT_WAIT_SLOT_CYCLES = 6,
  VX_TEAM_DXA_STAT_STREAM_CYCLES = 7,
  VX_TEAM_DXA_STAT_OVERWRITE_BLOCK_CYCLES = 8,
  VX_TEAM_DXA_STAT_DCACHE_REQ_STALL_CYCLES = 9,
  VX_TEAM_DXA_STAT_NO_FREE_WINDOW_CYCLES = 10,
  VX_TEAM_DXA_STAT_RESPONSE_WAIT_CYCLES = 11,
  VX_TEAM_DXA_STAT_LMEM_FANOUT_CYCLES = 12,
  VX_TEAM_DXA_STAT_LMEM_STALL_CYCLES = 13,
  VX_TEAM_DXA_STAT_DRAIN_CYCLES = 14,
  VX_TEAM_DXA_STAT_MAX_PENDING_READS = 15,
  VX_TEAM_DXA_STAT_MAX_READY_BACKLOG = 16,
  VX_TEAM_DXA_STAT_COUNT = 17,
};

inline uint64_t vx_team_dxa_stat_read(uint32_t selector) {
  csr_write(VX_CSR_TEAM_DXA_STAT_SEL, selector);
  return csr_read(VX_CSR_TEAM_DXA_STAT);
}

// launch a kernel function with a grid of blocks and block of threads
int vx_spawn_threads(uint32_t dimension,
                     const uint32_t* grid_dim,
                     const uint32_t* block_dim,
                     vx_kernel_func_cb kernel_func,
                     const void* arg);

int vx_spawn_cooperative_groups(uint32_t dimension,
                                const uint32_t* grid_dim,
                                const uint32_t* block_dim,
                                uint32_t team_dim_x,
                                uint32_t team_dim_y,
                                vx_kernel_func_cb kernel_func,
                                const void* arg);

// function call serialization
void vx_serial(vx_serial_cb callback, const void * arg);

#ifdef __cplusplus
}
#endif

#endif // __VX_SPAWN_H__
