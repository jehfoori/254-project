#include <iostream>
#include <vector>
#include <vortex.h>
#include "common.h"

#define RT_CHECK(_expr)                                         \
  do {                                                          \
    int _ret = _expr;                                           \
    if (0 == _ret)                                              \
      break;                                                    \
    std::cout << "Error: " << #_expr << " returned " << _ret    \
              << "!" << std::endl;                              \
    cleanup();                                                  \
    return -1;                                                  \
  } while (false)

namespace {

const char* kernel_file = "kernel.vxbin";
constexpr uint32_t kGridX = 2;
constexpr uint32_t kGridY = 2;
constexpr uint32_t kGridZ = 1;
constexpr uint32_t kBlockX = 1;
constexpr uint32_t kBlockY = 1;
constexpr uint32_t kBlockZ = 1;
constexpr uint32_t kTeamX = 2;
constexpr uint32_t kTeamY = 2;
constexpr uint32_t kBlockCount = kGridX * kGridY * kGridZ;
constexpr uint32_t kTileRows = 8;
constexpr uint32_t kGlobalStride = 512;
constexpr uint32_t kSlot0Offset = 0x80;
constexpr uint32_t kSlot0Size = 0x40;
constexpr uint32_t kSlot0Panels = 3;
constexpr uint32_t kSlot1Offset = 0x180;
constexpr uint32_t kSlot1Size = 0x60;
constexpr uint32_t kSlot1Panels = 5;
constexpr uint32_t kDxaStreamMode = 4;

vx_device_h device = nullptr;
vx_buffer_h dst_buffer = nullptr;
vx_buffer_h krnl_buffer = nullptr;
vx_buffer_h args_buffer = nullptr;
kernel_arg_t kernel_arg = {};

void cleanup() {
  if (device) {
    vx_mem_free(dst_buffer);
    vx_mem_free(krnl_buffer);
    vx_mem_free(args_buffer);
    vx_dev_close(device);
  }
}

uint32_t team_rank(uint32_t rank_x, uint32_t rank_y) {
  return rank_y * kTeamX + rank_x;
}

uint64_t slot0_global(uint32_t rank) {
  return 0x100000000ull + uint64_t(rank) * 0x1000ull;
}

uint64_t slot1_global(uint32_t rank) {
  return 0x200000000ull + uint64_t(rank) * 0x1000ull;
}

}

int main() {
  std::cout << "open device connection" << std::endl;
  RT_CHECK(vx_dev_open(&device));

  kernel_arg.grid_dim[0] = kGridX;
  kernel_arg.grid_dim[1] = kGridY;
  kernel_arg.grid_dim[2] = kGridZ;
  kernel_arg.block_dim[0] = kBlockX;
  kernel_arg.block_dim[1] = kBlockY;
  kernel_arg.block_dim[2] = kBlockZ;
  kernel_arg.team_dim[0] = kTeamX;
  kernel_arg.team_dim[1] = kTeamY;

  auto dst_buf_size = kBlockCount * sizeof(result_t);

  std::cout << "allocate device memory" << std::endl;
  RT_CHECK(vx_mem_alloc(device, dst_buf_size, VX_MEM_WRITE, &dst_buffer));
  RT_CHECK(vx_mem_address(dst_buffer, &kernel_arg.dst_addr));

  std::cout << "upload kernel binary" << std::endl;
  RT_CHECK(vx_upload_kernel_file(device, kernel_file, &krnl_buffer));

  std::cout << "upload kernel argument" << std::endl;
  RT_CHECK(vx_upload_bytes(device, &kernel_arg, sizeof(kernel_arg_t), &args_buffer));

  std::cout << "start device" << std::endl;
  RT_CHECK(vx_start(device, krnl_buffer, args_buffer));

  std::cout << "wait for completion" << std::endl;
  RT_CHECK(vx_ready_wait(device, VX_MAX_TIMEOUT));

  std::vector<result_t> results(kBlockCount);
  std::cout << "download results" << std::endl;
  RT_CHECK(vx_copy_from_dev(results.data(), dst_buffer, 0, dst_buf_size));

  int errors = 0;
  for (uint32_t block_y = 0; block_y < kGridY; ++block_y) {
    for (uint32_t block_x = 0; block_x < kGridX; ++block_x) {
      uint32_t index = block_x + block_y * kGridX;
      uint32_t rank = team_rank(block_x, block_y);
      auto& result = results.at(index);

      bool ok = true;
      ok &= result.team_id == 0;
      ok &= result.team_rank_x == block_x;
      ok &= result.team_rank_y == block_y;
      ok &= result.team_size_x == kTeamX;
      ok &= result.team_size_y == kTeamY;
      ok &= result.tile_rows == kTileRows;
      ok &= result.global_stride == kGlobalStride;
      ok &= result.slot0_offset == kSlot0Offset + rank * 4;
      ok &= result.slot0_size == kSlot0Size + rank * 8;
      ok &= result.slot0_mask == kSlot0Panels;
      ok &= result.slot0_mode == kDxaStreamMode;
      ok &= result.slot0_global == slot0_global(rank);
      ok &= result.slot1_offset == kSlot1Offset + rank * 4;
      ok &= result.slot1_size == kSlot1Size + rank * 8;
      ok &= result.slot1_mask == kSlot1Panels;
      ok &= result.slot1_mode == kDxaStreamMode;
      ok &= result.slot1_global == slot1_global(rank);
      ok &= result.status == 1;

      if (!ok) {
        std::cout << "*** error: block (" << block_x << ", " << block_y << ")"
                  << " team_id=" << result.team_id
                  << " rank=(" << result.team_rank_x << ", " << result.team_rank_y << ")"
                  << " size=(" << result.team_size_x << ", " << result.team_size_y << ")"
                  << " tile_rows=" << result.tile_rows
                  << " stride=" << result.global_stride
                  << " slot0={off=" << result.slot0_offset
                  << ", size=" << result.slot0_size
                  << ", mask=" << result.slot0_mask
                  << ", mode=" << result.slot0_mode
                  << ", global=0x" << std::hex << result.slot0_global << std::dec << "}"
                  << " slot1={off=" << result.slot1_offset
                  << ", size=" << result.slot1_size
                  << ", mask=" << result.slot1_mask
                  << ", mode=" << result.slot1_mode
                  << ", global=0x" << std::hex << result.slot1_global << std::dec << "}"
                  << " status=" << result.status
                  << std::endl;
        ++errors;
      }
    }
  }

  std::cout << "cleanup" << std::endl;
  cleanup();

  if (errors != 0) {
    std::cout << "Found " << errors << " errors!" << std::endl;
    std::cout << "FAILED!" << std::endl;
    return 1;
  }

  std::cout << "PASSED!" << std::endl;
  return 0;
}
