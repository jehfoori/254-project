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
  uint32_t team_rank_x;
  uint32_t team_rank_y;
  uint32_t slot0_mode;
  uint32_t slot1_mode;
  uint32_t slot0_panels;
  uint32_t slot1_panels;
  uint32_t status;
} result_t;

#endif
