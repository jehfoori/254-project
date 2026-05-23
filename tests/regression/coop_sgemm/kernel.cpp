#include <vx_spawn.h>
#include "common.h"

static inline uint32_t block_linear_id() {
  return blockIdx.x + blockIdx.y * gridDim.x;
}

void kernel_body(kernel_arg_t* __UNIFORM__ arg) {
  auto A_ptr = reinterpret_cast<TYPE*>(arg->A_addr);
  auto B_ptr = reinterpret_cast<TYPE*>(arg->B_addr);
  auto C_ptr = reinterpret_cast<TYPE*>(arg->C_addr);
  auto stats_ptr = reinterpret_cast<block_stats_t*>(arg->stats_addr);

  uint32_t size = arg->size;
  uint32_t tile_size = arg->tile_size;
  uint32_t a_tile_bytes = tile_size * tile_size * sizeof(TYPE);
  bool split_mode = (arg->mode == 0);
  bool fused_local_mode = (arg->mode == 1);
  bool global_sync_mode = (arg->mode == 2);
  bool global_async_mode = (arg->mode == 3);
  bool fused_mode = !split_mode;
  bool global_mode = global_sync_mode || global_async_mode;
  bool async_mode = global_async_mode;

  uint32_t tile_elems = blockDim.x * blockDim.y;
  auto local_ptr = reinterpret_cast<TYPE*>(__local_mem((async_mode ? 4 : 2) * tile_elems * sizeof(TYPE)));
  TYPE* a_tiles[2] = {
    local_ptr,
    async_mode ? (local_ptr + 2 * tile_elems) : local_ptr,
  };
  TYPE* b_tiles[2] = {
    local_ptr + tile_elems,
    async_mode ? (local_ptr + 3 * tile_elems) : (local_ptr + tile_elems),
  };

  uint32_t g_row = blockIdx.y * blockDim.y + threadIdx.y;
  uint32_t g_col = blockIdx.x * blockDim.x + threadIdx.x;
  uint32_t l_row = threadIdx.y;
  uint32_t l_col = threadIdx.x;
  uint32_t team_rank = vx_team_rank();

  TYPE sum(0);
  uint32_t a_tile_loads = 0;
  uint32_t b_tile_loads = 0;
  uint32_t team_barriers = 0;
  uint32_t team_arrives = 0;
  uint32_t team_waits = 0;

  if (threadIdx.x == 0 && threadIdx.y == 0) {
    vx_team_set_multicast_shape(tile_size, size * sizeof(TYPE));
  }
  __syncthreads();

  auto program_team_tiles = [&](uint32_t k_base, uint32_t buffer_slot) {
    uint32_t base_offset = async_mode ? (buffer_slot * 2 * a_tile_bytes) : 0;
    vx_team_clear_copy();
    if (__team_rank_x == 0) {
      if (global_mode) {
        uint64_t global_addr = arg->A_addr + sizeof(TYPE) * ((blockIdx.y * tile_size) * size + k_base);
        vx_team_set_multicast_slot(0, global_addr, base_offset, a_tile_bytes,
                                   (1u << team_rank) | (1u << (team_rank + 1)));
      } else {
        vx_team_set_copy_slot(0, 0, a_tile_bytes, 1u << (team_rank + 1));
      }
      ++a_tile_loads;
    }
    if (__team_rank_y == 0) {
      uint32_t slot = (__team_rank_x == 0) ? 1 : 0;
      if (global_mode) {
        uint64_t global_addr = arg->B_addr + sizeof(TYPE) * (k_base * size + blockIdx.x * tile_size);
        vx_team_set_multicast_slot(slot, global_addr, base_offset + a_tile_bytes, a_tile_bytes,
                                   (1u << team_rank) | (1u << (team_rank + __team_size_x)));
      } else {
        vx_team_set_copy_slot(slot, a_tile_bytes, a_tile_bytes, 1u << (team_rank + __team_size_x));
      }
      ++b_tile_loads;
    }
  };

  if (async_mode) {
    if (threadIdx.x == 0 && threadIdx.y == 0) {
      program_team_tiles(0, 0);
    }
    __syncthreads();
    vx_team_barrier();
    ++team_barriers;
  }

  for (uint32_t k = 0; k < size; k += tile_size) {
    uint32_t tile_index = k / tile_size;
    uint32_t current_slot = async_mode ? (tile_index & 1u) : 0u;
    auto local_A = a_tiles[current_slot];
    auto local_B = b_tiles[current_slot];

    if (__team_rank_x == 0 && !global_mode) {
      local_A[l_row * tile_size + l_col] = A_ptr[g_row * size + (k + l_col)];
    }

    if (__team_rank_y == 0 && !global_mode) {
      local_B[l_row * tile_size + l_col] = B_ptr[(k + l_row) * size + g_col];
    }
    __syncthreads();

    if (!async_mode) {
      if (threadIdx.x == 0 && threadIdx.y == 0) {
        if (fused_mode) {
          program_team_tiles(k, 0);
        } else {
          if (__team_rank_x == 0) {
            vx_team_set_copy(0, a_tile_bytes, 1u << (team_rank + 1));
            ++a_tile_loads;
          } else {
            vx_team_clear_copy();
          }
        }
      }
      __syncthreads();
      vx_team_barrier();
      ++team_barriers;

      if (split_mode) {
        if (threadIdx.x == 0 && threadIdx.y == 0) {
          if (__team_rank_y == 0) {
            vx_team_set_copy(a_tile_bytes, a_tile_bytes, 1u << (team_rank + __team_size_x));
            ++b_tile_loads;
          } else {
            vx_team_clear_copy();
          }
        }
        __syncthreads();
        vx_team_barrier();
        ++team_barriers;
      }
    } else {
      uint32_t next_k = k + tile_size;
      if (next_k < size) {
        uint32_t next_slot = current_slot ^ 1u;
        if (threadIdx.x == 0 && threadIdx.y == 0) {
          program_team_tiles(next_k, next_slot);
        }
        __syncthreads();
        vx_team_arrive();
        ++team_arrives;
      }
    }

    for (uint32_t j = 0; j < tile_size; ++j) {
      sum += local_A[l_row * tile_size + j] * local_B[j * tile_size + l_col];
    }
    __syncthreads();

    if (async_mode && (k + tile_size) < size) {
      vx_team_wait();
      ++team_waits;
    }
  }

  C_ptr[g_row * size + g_col] = sum;

  if (threadIdx.x == 0 && threadIdx.y == 0) {
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
