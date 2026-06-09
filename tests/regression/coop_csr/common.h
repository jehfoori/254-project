#ifndef _COMMON_H_
#define _COMMON_H_

#include <stdint.h>

typedef struct {
  uint32_t team_id;
  uint32_t team_rank_x;
  uint32_t team_rank_y;
  uint32_t idle_status;
  uint32_t done_status;
  uint32_t remaining_cycles;
  uint32_t k_tiles;
  uint32_t n_tiles;
  uint32_t a_offset;
  uint32_t b_offset;
  uint32_t c_addr_lo;
  uint32_t state;
  uint32_t compute_steps_done;
  uint32_t compute_steps_total;
  uint32_t writeback_count;
  uint32_t writeback_signature;
  uint32_t accum;
  uint32_t k_idx;
  uint32_t n_idx;
  uint32_t write_addr;
  uint32_t write_data;
  uint32_t ok;
} result_t;

typedef struct {
  uint32_t grid_dim[3];
  uint32_t block_dim[3];
  uint32_t team_dim[2];
  uint64_t dst_addr;
} kernel_arg_t;

#endif
