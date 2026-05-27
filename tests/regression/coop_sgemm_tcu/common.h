#ifndef _COMMON_H_
#define _COMMON_H_

#include <stdint.h>

#ifndef NUM_THREADS
#define NUM_THREADS 4
#endif

#ifndef ITYPE
#define ITYPE fp16
#endif

#ifndef OTYPE
#define OTYPE fp32
#endif

#ifndef PANEL_K_TILES
#define PANEL_K_TILES 1
#endif

#ifndef COOP_N_TILES
#define COOP_N_TILES 1
#endif

#ifndef COOP_M_TILES
#define COOP_M_TILES 1
#endif

#ifndef COOP_PROFILE_STATS
#define COOP_PROFILE_STATS 1
#endif

#ifndef COOP_DIAG_SERIAL_N2
#define COOP_DIAG_SERIAL_N2 0
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
  uint32_t transfer_events;
  uint32_t global_bytes;
  uint32_t fanout_bytes;
  uint32_t panel_write_bytes;
  uint32_t panel_read_bytes;
  uint32_t panel_transfers;
  uint32_t mma_steps;
  uint32_t partial_panels;
  uint32_t slot_reuse_barriers;
} block_stats_t;

typedef struct {
  uint32_t grid_dim[2];
  uint32_t block_dim[2];
  uint32_t team_dim[2];
  uint32_t mode;
  uint32_t M, N, K;
  uint64_t A_addr;
  uint64_t B_addr;
  uint64_t C_addr;
  uint64_t stats_addr;
} kernel_arg_t;

#endif
