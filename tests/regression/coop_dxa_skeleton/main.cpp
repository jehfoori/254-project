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
constexpr uint32_t kDxaStreamMode = 4;
constexpr uint32_t kSlot0Panels = 2;
constexpr uint32_t kSlot1Panels = 2;

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
      auto& result = results.at(index);

      bool ok = true;
      ok &= result.team_rank_x == block_x;
      ok &= result.team_rank_y == block_y;
      ok &= result.slot0_mode == kDxaStreamMode;
      ok &= result.slot1_mode == kDxaStreamMode;
      ok &= result.slot0_panels == kSlot0Panels;
      ok &= result.slot1_panels == kSlot1Panels;
      ok &= result.status == 1;

      if (!ok) {
        std::cout << "*** error: block (" << block_x << ", " << block_y << ")"
                  << " rank=(" << result.team_rank_x << ", " << result.team_rank_y << ")"
                  << " slot0={mode=" << result.slot0_mode
                  << ", panels=" << result.slot0_panels << "}"
                  << " slot1={mode=" << result.slot1_mode
                  << ", panels=" << result.slot1_panels << "}"
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
