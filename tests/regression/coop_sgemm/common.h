#ifndef _COMMON_H_
#define _COMMON_H_

#include <stdint.h>

#ifndef TYPE
#define TYPE float
#endif

typedef struct {
  uint32_t team_id;
  uint32_t team_rank_x;
  uint32_t team_rank_y;
  uint32_t a_tile_loads;
  uint32_t b_tile_loads;
  uint32_t team_barriers;
  uint32_t team_arrives;
  uint32_t team_waits;
} block_stats_t;

typedef struct {
  uint32_t grid_dim[2];
  uint32_t block_dim[2];
  uint32_t team_dim[2];
  uint32_t size;
  uint32_t tile_size;
  uint32_t mode;
  uint64_t A_addr;
  uint64_t B_addr;
  uint64_t C_addr;
  uint64_t stats_addr;
} kernel_arg_t;

#endif
