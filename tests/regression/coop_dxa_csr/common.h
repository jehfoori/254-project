#ifndef _COMMON_H_
#define _COMMON_H_

#include <stdint.h>

typedef struct {
  uint32_t grid_dim[3];
  uint32_t block_dim[3];
  uint32_t team_dim[2];
  uint64_t dst_addr;
} kernel_arg_t;

typedef struct {
  uint32_t team_id;
  uint32_t team_rank_x;
  uint32_t team_rank_y;
  uint32_t team_size_x;
  uint32_t team_size_y;
  uint32_t tile_rows;
  uint32_t global_stride;
  uint32_t slot0_offset;
  uint32_t slot0_size;
  uint32_t slot0_mask;
  uint32_t slot0_mode;
  uint64_t slot0_global;
  uint32_t slot1_offset;
  uint32_t slot1_size;
  uint32_t slot1_mask;
  uint32_t slot1_mode;
  uint64_t slot1_global;
  uint32_t status;
} result_t;

#endif
