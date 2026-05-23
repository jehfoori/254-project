#include <chrono>
#include <cmath>
#include <iostream>
#include <string.h>
#include <unistd.h>
#include <vector>
#include <vortex.h>
#include "common.h"

#define FLOAT_ULP 6

#define RT_CHECK(_expr)                                         \
   do {                                                         \
     int _ret = _expr;                                          \
     if (0 == _ret)                                             \
       break;                                                   \
     printf("Error: '%s' returned %d!\n", #_expr, (int)_ret);   \
     cleanup();                                                 \
     exit(-1);                                                  \
   } while (false)

template <typename Type>
class Comparator {};

template <>
class Comparator<float> {
public:
  static const char* type_str() {
    return "float";
  }
  static float generate() {
    return static_cast<float>(rand()) / RAND_MAX;
  }
  static bool compare(float a, float b, int index, int errors) {
    union fi_t { float f; int32_t i; };
    fi_t fa, fb;
    fa.f = a;
    fb.f = b;
    auto d = std::abs(fa.i - fb.i);
    if (d > FLOAT_ULP) {
      if (errors < 100) {
        printf("*** error: [%d] expected=%f, actual=%f\n", index, b, a);
      }
      return false;
    }
    return true;
  }
};

static void matmul_cpu(TYPE* out, const TYPE* A, const TYPE* B, uint32_t width, uint32_t height) {
  for (uint32_t row = 0; row < height; ++row) {
    for (uint32_t col = 0; col < width; ++col) {
      TYPE sum(0);
      for (uint32_t e = 0; e < width; ++e) {
        sum += A[row * width + e] * B[e * width + col];
      }
      out[row * width + col] = sum;
    }
  }
}

const char* kernel_file = "kernel.vxbin";
uint32_t size = 16;
uint32_t tile_size = 4;
uint32_t mode = 3;

vx_device_h device = nullptr;
vx_buffer_h A_buffer = nullptr;
vx_buffer_h B_buffer = nullptr;
vx_buffer_h C_buffer = nullptr;
vx_buffer_h stats_buffer = nullptr;
vx_buffer_h krnl_buffer = nullptr;
vx_buffer_h args_buffer = nullptr;
kernel_arg_t kernel_arg = {};

static void show_usage() {
   std::cout << "Vortex Test." << std::endl;
   std::cout << "Usage: [-k kernel] [-n matrix_size] [-t tile_size] [-m 0|1|2|3] [-h help]" << std::endl;
}

static void parse_args(int argc, char **argv) {
  int c;
  while ((c = getopt(argc, argv, "n:t:m:k:h")) != -1) {
    switch (c) {
    case 'n':
      size = atoi(optarg);
      break;
    case 't':
      tile_size = atoi(optarg);
      break;
    case 'm':
      mode = atoi(optarg);
      break;
    case 'k':
      kernel_file = optarg;
      break;
    case 'h':
      show_usage();
      exit(0);
      break;
    default:
      show_usage();
      exit(-1);
    }
  }
}

void cleanup() {
  if (device) {
    vx_mem_free(A_buffer);
    vx_mem_free(B_buffer);
    vx_mem_free(C_buffer);
    vx_mem_free(stats_buffer);
    vx_mem_free(krnl_buffer);
    vx_mem_free(args_buffer);
    vx_dev_close(device);
  }
}

int main(int argc, char *argv[]) {
  parse_args(argc, argv);

  if ((size % tile_size) != 0 || (size % (2 * tile_size)) != 0) {
    std::cout << "Error: matrix size must be divisible by tile_size and 2*tile_size" << std::endl;
    return -1;
  }

  std::srand(50);

  std::cout << "open device connection" << std::endl;
  RT_CHECK(vx_dev_open(&device));

  uint64_t num_cores = 0;
  RT_CHECK(vx_dev_caps(device, VX_CAPS_NUM_CORES, &num_cores));
  if (num_cores < 4) {
    std::cout << "Error: cooperative SGEMM requires at least 4 cores" << std::endl;
    cleanup();
    return -1;
  }

  uint32_t size_sq = size * size;
  uint32_t buf_size = size_sq * sizeof(TYPE);
  uint32_t group_size = tile_size * tile_size;
  uint32_t local_mem = ((mode == 3) ? 4 : 2) * group_size * sizeof(TYPE);
  uint32_t grid_dim = size / tile_size;
  uint32_t block_count = grid_dim * grid_dim;
  uint32_t stats_buf_size = block_count * sizeof(block_stats_t);

  kernel_arg.grid_dim[0] = grid_dim;
  kernel_arg.grid_dim[1] = grid_dim;
  kernel_arg.block_dim[0] = tile_size;
  kernel_arg.block_dim[1] = tile_size;
  kernel_arg.team_dim[0] = 2;
  kernel_arg.team_dim[1] = 2;
  kernel_arg.size = size;
  kernel_arg.tile_size = tile_size;
  kernel_arg.mode = mode;

  std::cout << "data type: " << Comparator<TYPE>::type_str() << std::endl;
  std::cout << "matrix size: " << size << "x" << size << std::endl;
  std::cout << "tile size: " << tile_size << "x" << tile_size << std::endl;
  const char* mode_name = "split-local";
  if (mode == 1) {
    mode_name = "fused-local";
  } else if (mode == 2) {
    mode_name = "fused-global-sync";
  } else if (mode == 3) {
    mode_name = "fused-global-async";
  }
  std::cout << "copy mode: " << mode_name << std::endl;
  std::cout << "local memory: " << local_mem << " bytes" << std::endl;

  uint32_t max_localmem;
  RT_CHECK(vx_check_occupancy(device, group_size, &max_localmem));
  std::cout << "occupancy: max_localmem=" << max_localmem << " bytes" << std::endl;
  if (max_localmem < local_mem) {
    std::cout << "Error: Not enough local memory: needed=" << local_mem
              << ", available=" << max_localmem << std::endl;
    cleanup();
    return -1;
  }

  std::cout << "allocate device memory" << std::endl;
  RT_CHECK(vx_mem_alloc(device, buf_size, VX_MEM_READ, &A_buffer));
  RT_CHECK(vx_mem_address(A_buffer, &kernel_arg.A_addr));
  RT_CHECK(vx_mem_alloc(device, buf_size, VX_MEM_READ, &B_buffer));
  RT_CHECK(vx_mem_address(B_buffer, &kernel_arg.B_addr));
  RT_CHECK(vx_mem_alloc(device, buf_size, VX_MEM_WRITE, &C_buffer));
  RT_CHECK(vx_mem_address(C_buffer, &kernel_arg.C_addr));
  RT_CHECK(vx_mem_alloc(device, stats_buf_size, VX_MEM_WRITE, &stats_buffer));
  RT_CHECK(vx_mem_address(stats_buffer, &kernel_arg.stats_addr));

  std::vector<TYPE> h_A(size_sq);
  std::vector<TYPE> h_B(size_sq);
  std::vector<TYPE> h_C(size_sq);
  std::vector<block_stats_t> h_stats(block_count);
  for (uint32_t i = 0; i < size_sq; ++i) {
    h_A[i] = Comparator<TYPE>::generate();
    h_B[i] = Comparator<TYPE>::generate();
  }

  std::cout << "upload matrix A buffer" << std::endl;
  RT_CHECK(vx_copy_to_dev(A_buffer, h_A.data(), 0, buf_size));
  std::cout << "upload matrix B buffer" << std::endl;
  RT_CHECK(vx_copy_to_dev(B_buffer, h_B.data(), 0, buf_size));

  std::cout << "upload kernel binary" << std::endl;
  RT_CHECK(vx_upload_kernel_file(device, kernel_file, &krnl_buffer));
  std::cout << "upload kernel argument" << std::endl;
  RT_CHECK(vx_upload_bytes(device, &kernel_arg, sizeof(kernel_arg_t), &args_buffer));

  auto time_start = std::chrono::high_resolution_clock::now();

  std::cout << "start device" << std::endl;
  RT_CHECK(vx_start(device, krnl_buffer, args_buffer));
  std::cout << "wait for completion" << std::endl;
  RT_CHECK(vx_ready_wait(device, VX_MAX_TIMEOUT));

  auto time_end = std::chrono::high_resolution_clock::now();
  double elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(time_end - time_start).count();
  std::cout << "Elapsed time: " << elapsed << " ms" << std::endl;

  std::cout << "download destination buffer" << std::endl;
  RT_CHECK(vx_copy_from_dev(h_C.data(), C_buffer, 0, buf_size));
  std::cout << "download stats buffer" << std::endl;
  RT_CHECK(vx_copy_from_dev(h_stats.data(), stats_buffer, 0, stats_buf_size));

  std::cout << "verify result" << std::endl;
  int errors = 0;
  {
    std::vector<TYPE> h_ref(size_sq);
    matmul_cpu(h_ref.data(), h_A.data(), h_B.data(), size, size);
    for (uint32_t i = 0; i < h_ref.size(); ++i) {
      if (!Comparator<TYPE>::compare(h_C[i], h_ref[i], i, errors)) {
        ++errors;
      }
    }
  }

  uint32_t macro_grid = grid_dim / 2;
  uint32_t num_k_tiles = size / tile_size;
  uint32_t total_a_tile_loads = 0;
  uint32_t total_b_tile_loads = 0;
  for (uint32_t by = 0; by < grid_dim; ++by) {
    for (uint32_t bx = 0; bx < grid_dim; ++bx) {
      uint32_t idx = bx + by * grid_dim;
      auto& stats = h_stats.at(idx);
      uint32_t expected_team_id = (bx / 2) + (by / 2) * macro_grid;
      uint32_t expected_rank_x = bx % 2;
      uint32_t expected_rank_y = by % 2;
      uint32_t expected_a_tile_loads = (expected_rank_x == 0) ? num_k_tiles : 0;
      uint32_t expected_b_tile_loads = (expected_rank_y == 0) ? num_k_tiles : 0;
      uint32_t expected_team_barriers = 0;
      uint32_t expected_team_arrives = 0;
      uint32_t expected_team_waits = 0;
      if (mode == 0) {
        expected_team_barriers = num_k_tiles * 2;
      } else if (mode == 1 || mode == 2) {
        expected_team_barriers = num_k_tiles;
      } else {
        expected_team_barriers = 1;
        expected_team_arrives = (num_k_tiles > 0) ? (num_k_tiles - 1) : 0;
        expected_team_waits = expected_team_arrives;
      }
      if (stats.team_id != expected_team_id
       || stats.team_rank_x != expected_rank_x
       || stats.team_rank_y != expected_rank_y
       || stats.a_tile_loads != expected_a_tile_loads
       || stats.b_tile_loads != expected_b_tile_loads
       || stats.team_barriers != expected_team_barriers
       || stats.team_arrives != expected_team_arrives
       || stats.team_waits != expected_team_waits) {
        std::cout << "*** stats error: block (" << bx << ", " << by << ")"
                  << " team_id=" << stats.team_id
                  << " rank=(" << stats.team_rank_x << ", " << stats.team_rank_y << ")"
                  << " a_tiles=" << stats.a_tile_loads
                  << " b_tiles=" << stats.b_tile_loads
                  << " barriers=" << stats.team_barriers
                  << " arrives=" << stats.team_arrives
                  << " waits=" << stats.team_waits
                  << std::endl;
        ++errors;
      }
      total_a_tile_loads += stats.a_tile_loads;
      total_b_tile_loads += stats.b_tile_loads;
    }
  }

  uint32_t cooperative_tile_loads = total_a_tile_loads + total_b_tile_loads;
  uint32_t baseline_tile_loads = block_count * num_k_tiles * 2;
  std::cout << "cooperative tile loads: " << cooperative_tile_loads
            << " (A=" << total_a_tile_loads
            << ", B=" << total_b_tile_loads << ")" << std::endl;
  std::cout << "baseline tile loads: " << baseline_tile_loads << std::endl;

  if (cooperative_tile_loads * 2 != baseline_tile_loads) {
    std::cout << "*** load reduction check failed" << std::endl;
    ++errors;
  }

  std::cout << "cleanup" << std::endl;
  cleanup();

  if (errors != 0) {
    std::cout << "Found " << errors << " errors!" << std::endl;
    std::cout << "FAILED!" << std::endl;
    return errors;
  }

  std::cout << "PASSED!" << std::endl;
  return 0;
}
