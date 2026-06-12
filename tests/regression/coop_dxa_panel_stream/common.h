#ifndef _COMMON_H_
#define _COMMON_H_

#include <stdint.h>

enum : uint32_t {
  kPanelCount = 4,
  kTeamX = 2,
  kTeamY = 2,
  kTileRows = 2,
  kRowBytes = 16,
  kCopyBytes = kTileRows * kRowBytes,
  kGlobalStride = 64,
  kAOffset = 0,
  kBOffset = kTeamY * kCopyBytes,
  kPanelSlotBytes = kTeamY * kCopyBytes + kTeamX * kCopyBytes,
  kWordsPerCopy = kCopyBytes / sizeof(uint64_t),
};

typedef struct {
  uint32_t grid_dim[3];
  uint32_t block_dim[3];
  uint32_t team_dim[2];
  uint64_t src_a_addr;
  uint64_t src_b_addr;
  uint64_t dst_addr;
} kernel_arg_t;

typedef struct {
  uint32_t team_rank_x;
  uint32_t team_rank_y;
  uint32_t errors;
  uint32_t status;
  uint64_t first_actual;
  uint64_t first_expected;
} result_t;

static inline uint64_t expected_value(uint32_t copy, uint32_t panel, uint32_t group, uint32_t row, uint32_t beat) {
  return (uint64_t(copy + 1) << 60)
       | (uint64_t(panel) << 44)
       | (uint64_t(group) << 32)
       | (uint64_t(row) << 16)
       | uint64_t(beat);
}

#endif
