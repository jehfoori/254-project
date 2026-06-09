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

vx_device_h device = nullptr;
vx_buffer_h dst_buffer = nullptr;
vx_buffer_h krnl_buffer = nullptr;
vx_buffer_h args_buffer = nullptr;
kernel_arg_t kernel_arg = {};

uint32_t expected_accum(uint32_t a_offset, uint32_t b_offset,
                        uint32_t k_tiles, uint32_t n_tiles) {
  uint32_t acc = 0;
  uint32_t k_idx = 0;
  uint32_t n_idx = 0;
  uint32_t base = (a_offset >> 2) + (b_offset >> 2);
  uint32_t total = k_tiles * n_tiles;
  for (uint32_t step = 0; step < total; ++step) {
    acc += base + k_idx + n_idx;
    if (k_idx + 1 < k_tiles) {
      ++k_idx;
    } else {
      k_idx = 0;
      ++n_idx;
    }
  }
  return acc;
}

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
      uint32_t expected_rank = block_y * kTeamX + block_x;
      uint32_t expected_n_tiles = expected_rank + 1;
      uint32_t expected_a = 0x100 + 0x20 * expected_rank;
      uint32_t expected_b = 0x200 + 0x20 * expected_rank;
      uint32_t expected_c = 0x3000 + 0x40 * expected_rank;
      uint32_t expected_steps = 3 * expected_n_tiles;
      uint32_t expected_acc = expected_accum(expected_a, expected_b, 3, expected_n_tiles);
      uint32_t expected_signature = expected_acc ^ expected_steps ^ expected_n_tiles;

      if (result.team_id != 0
       || result.team_rank_x != block_x
       || result.team_rank_y != block_y
       || result.idle_status != 0
       || result.remaining_cycles != 0
       || result.k_tiles != 3
       || result.n_tiles != expected_n_tiles
       || result.a_offset != expected_a
       || result.b_offset != expected_b
       || result.c_addr_lo != expected_c
       || result.state != 4
       || result.compute_steps_done != expected_steps
       || result.compute_steps_total != expected_steps
       || result.writeback_count != 1
       || result.writeback_signature != expected_signature
       || result.accum != expected_acc
       || result.k_idx != 0
       || result.n_idx != expected_n_tiles - 1
       || result.write_addr != expected_c
       || result.write_data != expected_signature
       || result.ok != 1) {
        std::cout << "*** error: block (" << block_x << ", " << block_y << ")"
                  << " team_id=" << result.team_id
                  << " rank=(" << result.team_rank_x << ", " << result.team_rank_y << ")"
                  << " idle=" << result.idle_status
                  << " done_status=" << result.done_status
                  << " remaining=" << result.remaining_cycles
                  << " k=" << result.k_tiles
                  << " n=" << result.n_tiles
                  << " a=0x" << std::hex << result.a_offset
                  << " b=0x" << result.b_offset
                  << " c=0x" << result.c_addr_lo
                  << " state=" << std::dec << result.state
                  << " done_steps=" << result.compute_steps_done
                  << " total_steps=" << result.compute_steps_total
                  << " writebacks=" << result.writeback_count
                  << " accum=" << result.accum
                  << " k_idx=" << result.k_idx
                  << " n_idx=" << result.n_idx
                  << " waddr=0x" << std::hex << result.write_addr
                  << " wdata=0x" << result.write_data
                  << " sig=0x" << std::hex << result.writeback_signature
                  << " ok=" << std::dec << result.ok
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
