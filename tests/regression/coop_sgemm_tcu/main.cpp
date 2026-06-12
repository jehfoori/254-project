#include "common.h"
#include <chrono>
#include <cmath>
#include <iostream>
#include <rvfloats.h>
#include <string.h>
#include <tensor_cfg.h>
#include <unistd.h>
#include <util.h>
#include <vector>
#include <vortex.h>

#define FLOAT_ULP 6
#define MAX_ERRORS 100

#define RT_CHECK(_expr)                                      \
  do {                                                       \
    int _ret = _expr;                                        \
    if (0 == _ret)                                           \
      break;                                                 \
    printf("Error: '%s' returned %d!\n", #_expr, (int)_ret); \
    cleanup();                                               \
    exit(-1);                                                \
  } while (false)

using namespace vortex;
namespace vt = tensor;
using cfg = vt::wmma_config_t<NUM_THREADS, vt::ITYPE, vt::OTYPE>;

using itype_t = typename vt::ITYPE::dtype;
using otype_t = typename vt::OTYPE::dtype;

static_assert(vt::ITYPE::bits == 16, "coop_sgemm_tcu currently assumes 16-bit tensor inputs");
static_assert(vt::OTYPE::bits == 32, "coop_sgemm_tcu currently assumes 32-bit tensor outputs");
static_assert(cfg::tileM == cfg::tileK, "coop_sgemm_tcu assumes tileM == tileK for panel layout");

static const char* kernel_file = "kernel.vxbin";
static uint32_t xm = 32;
static uint32_t xn = 32;
static uint32_t xk = 32;

static vx_device_h device = nullptr;
static vx_buffer_h A_buffer = nullptr;
static vx_buffer_h B_buffer = nullptr;
static vx_buffer_h C_buffer = nullptr;
static vx_buffer_h stats_buffer = nullptr;
static vx_buffer_h krnl_buffer = nullptr;
static vx_buffer_h args_buffer = nullptr;
static kernel_arg_t kernel_arg = {};

static uint16_t generate_fp16() {
  auto value = float(rand()) / RAND_MAX;
  return rv_ftoh_s(bit_cast<uint32_t>(value), 0, nullptr);
}

static float fp16_to_float(uint16_t value) {
  return bit_cast<float>(rv_htof_s(value, 0, nullptr));
}

static int32_t fp32_ulp_delta(float actual, float expected) {
  union fi_t {
    float f;
    int32_t i;
  };
  fi_t fa, fb;
  fa.f = actual;
  fb.f = expected;
  return std::abs(fa.i - fb.i);
}

static uint32_t fp32_bits(float value) {
  union fi_t {
    float f;
    uint32_t i;
  };
  fi_t v;
  v.f = value;
  return v.i;
}

static bool compare_fp32(float actual, float expected, uint32_t index, uint32_t errors) {
  auto d = fp32_ulp_delta(actual, expected);
  if (d > FLOAT_ULP) {
    if (errors < MAX_ERRORS) {
      std::cout << "*** error: [" << index << "] expected=" << expected
                << " (0x" << std::hex << fp32_bits(expected) << ")"
                << ", actual=" << std::dec << actual
                << " (0x" << std::hex << fp32_bits(actual) << ")"
                << std::dec << ", ulp=" << d << std::endl;
    }
    return false;
  }
  return true;
}

static void matmul_cpu(float* out, const uint16_t* A, const uint16_t* B, uint32_t M, uint32_t N, uint32_t K) {
  for (uint32_t row = 0; row < M; ++row) {
    for (uint32_t col = 0; col < N; ++col) {
      float sum = 0.0f;
      for (uint32_t e = 0; e < K; ++e) {
        sum += fp16_to_float(A[row * K + e]) * fp16_to_float(B[e * N + col]);
      }
      out[row * N + col] = sum;
    }
  }
}

static void show_usage() {
  std::cout << "Vortex cooperative tensor GEMM DXA panel test.\n";
  std::cout << "Usage: [-f kernel] [-m rows] [-n cols] [-k inner] [-h]\n";
}

static void parse_args(int argc, char** argv) {
  int c;
  while ((c = getopt(argc, argv, "m:n:k:f:h")) != -1) {
    switch (c) {
    case 'm':
      xm = atoi(optarg);
      break;
    case 'n':
      xn = atoi(optarg);
      break;
    case 'k':
      xk = atoi(optarg);
      break;
    case 'f':
      kernel_file = optarg;
      break;
    case 'h':
      show_usage();
      exit(0);
    default:
      show_usage();
      exit(-1);
    }
  }
}

static void cleanup() {
  if (device) {
    if (A_buffer) vx_mem_free(A_buffer);
    if (B_buffer) vx_mem_free(B_buffer);
    if (C_buffer) vx_mem_free(C_buffer);
    if (stats_buffer) vx_mem_free(stats_buffer);
    if (krnl_buffer) vx_mem_free(krnl_buffer);
    if (args_buffer) vx_mem_free(args_buffer);
    vx_dev_close(device);
  }
}

int main(int argc, char* argv[]) {
  parse_args(argc, argv);
  std::srand(50);

  std::cout << "open device connection" << std::endl;
  RT_CHECK(vx_dev_open(&device));

  uint64_t isa_flags = 0;
  RT_CHECK(vx_dev_caps(device, VX_CAPS_ISA_FLAGS, &isa_flags));
  if ((isa_flags & VX_ISA_EXT_TCU) == 0) {
    std::cout << "TCU extension not supported!" << std::endl;
    cleanup();
    return -1;
  }

  uint64_t num_cores = 0;
  RT_CHECK(vx_dev_caps(device, VX_CAPS_NUM_CORES, &num_cores));
  if (num_cores < 4) {
    std::cout << "Error: cooperative tensor GEMM requires at least 4 cores" << std::endl;
    cleanup();
    return -1;
  }

  uint64_t warp_threads = 0;
  RT_CHECK(vx_dev_caps(device, VX_CAPS_NUM_THREADS, &warp_threads));
  if (warp_threads != NUM_THREADS) {
    std::cout << "Error: device warp size (" << warp_threads
              << ") must match NUM_THREADS=" << NUM_THREADS << "!" << std::endl;
    cleanup();
    return -1;
  }

  uint32_t M = xm;
  uint32_t N = xn;
  uint32_t K = xk;
  if ((M % cfg::tileM) != 0 || (N % cfg::tileN) != 0 || (K % cfg::tileK) != 0) {
    std::cout << "Error: matrix dimensions must be multiples of tensor tile sizes" << std::endl;
    cleanup();
    return -1;
  }
  if ((N % (cfg::tileN * COOP_N_TILES)) != 0) {
    std::cout << "Error: N must be divisible by tileN * COOP_N_TILES" << std::endl;
    cleanup();
    return -1;
  }
  if ((M % (cfg::tileM * COOP_M_TILES)) != 0) {
    std::cout << "Error: M must be divisible by tileM * COOP_M_TILES" << std::endl;
    cleanup();
    return -1;
  }

  uint32_t grid_x = N / (cfg::tileN * COOP_N_TILES);
  uint32_t grid_y = M / (cfg::tileM * COOP_M_TILES);
  if ((grid_x % 2) != 0 || (grid_y % 2) != 0) {
    std::cout << "Error: grid dimensions must be divisible by 2 for 2x2 cooperative teams" << std::endl;
    cleanup();
    return -1;
  }

  size_t sizeA = M * K;
  size_t sizeB = K * N;
  size_t sizeC = M * N;
  uint32_t block_count = grid_x * grid_y;
#if COOP_PROFILE_STATS
  uint32_t stats_buf_size = block_count * sizeof(block_stats_t);
#endif
  uint32_t num_k_tiles = K / cfg::tileK;
  if ((num_k_tiles % PANEL_K_TILES) != 0) {
    std::cout << "Error: K tile count must be divisible by PANEL_K_TILES for the DXA panel baseline" << std::endl;
    cleanup();
    return -1;
  }
  constexpr uint32_t team_panel_window = 0x10000;
  uint32_t a_panel_bytes = PANEL_K_TILES * cfg::tileM * cfg::tileK * COOP_M_TILES * sizeof(itype_t);
  uint32_t b_panel_bytes = PANEL_K_TILES * cfg::tileK * cfg::tileN * COOP_N_TILES * sizeof(itype_t);
  uint32_t panel_slot_bytes = 2 * a_panel_bytes + 2 * b_panel_bytes;
  uint32_t active_panel_slots = (num_k_tiles > PANEL_K_TILES) ? 2 : 1;
  uint32_t required_panel_bytes = active_panel_slots * panel_slot_bytes;
  if (required_panel_bytes > team_panel_window) {
    std::cout << "Error: panel store window too small: needed=" << required_panel_bytes
              << ", available=" << team_panel_window << std::endl;
    cleanup();
    return -1;
  }

  kernel_arg.grid_dim[0] = grid_x;
  kernel_arg.grid_dim[1] = grid_y;
  kernel_arg.block_dim[0] = NUM_THREADS;
  kernel_arg.block_dim[1] = 1;
  kernel_arg.team_dim[0] = 2;
  kernel_arg.team_dim[1] = 2;
  kernel_arg.M = M;
  kernel_arg.N = N;
  kernel_arg.K = K;

  std::cout << "input data type: " << vt::ITYPE::name << " (id=" << vt::ITYPE::id << ")" << std::endl;
  std::cout << "output data type: " << vt::OTYPE::name << " (id=" << vt::OTYPE::id << ")" << std::endl;
  std::cout << "WMMA Tile Dimension: M=" << cfg::tileM << ", N=" << cfg::tileN
            << ", K=" << cfg::tileK << std::endl;
  std::cout << "matrix A: " << M << "x" << K << std::endl;
  std::cout << "matrix B: " << K << "x" << N << std::endl;
  std::cout << "matrix C: " << M << "x" << N << std::endl;
  std::cout << "panel K tiles: " << PANEL_K_TILES << std::endl;
  std::cout << "cooperative M tiles: " << COOP_M_TILES << std::endl;
  std::cout << "cooperative N tiles: " << COOP_N_TILES << std::endl;
  std::cout << "DXA pipeline: on" << std::endl;
  std::cout << "profile stats: " << (COOP_PROFILE_STATS ? "on" : "off") << std::endl;
  std::cout << "copy mode: dxa-panel" << std::endl;
  std::cout << "local memory: 0 bytes" << std::endl;

  uint32_t max_localmem = 0;
  RT_CHECK(vx_check_occupancy(device, NUM_THREADS, &max_localmem));
  std::cout << "occupancy: max_localmem=" << max_localmem << " bytes" << std::endl;

  std::cout << "allocate device memory" << std::endl;
  RT_CHECK(vx_mem_alloc(device, sizeA * sizeof(itype_t), VX_MEM_READ, &A_buffer));
  RT_CHECK(vx_mem_address(A_buffer, &kernel_arg.A_addr));
  RT_CHECK(vx_mem_alloc(device, sizeB * sizeof(itype_t), VX_MEM_READ, &B_buffer));
  RT_CHECK(vx_mem_address(B_buffer, &kernel_arg.B_addr));
  RT_CHECK(vx_mem_alloc(device, sizeC * sizeof(otype_t), VX_MEM_WRITE, &C_buffer));
  RT_CHECK(vx_mem_address(C_buffer, &kernel_arg.C_addr));
#if COOP_PROFILE_STATS
  RT_CHECK(vx_mem_alloc(device, stats_buf_size, VX_MEM_WRITE, &stats_buffer));
  RT_CHECK(vx_mem_address(stats_buffer, &kernel_arg.stats_addr));
#endif

  std::vector<itype_t> h_A(sizeA);
  std::vector<itype_t> h_B(sizeB);
  std::vector<otype_t> h_C(sizeC);
  std::vector<block_stats_t> h_stats(COOP_PROFILE_STATS ? block_count : 0);
  std::vector<float> h_ref(sizeC);

  for (uint32_t i = 0; i < sizeA; ++i) {
    h_A[i] = generate_fp16();
  }
  for (uint32_t i = 0; i < sizeB; ++i) {
    h_B[i] = generate_fp16();
  }

  std::cout << "upload matrix A buffer" << std::endl;
  RT_CHECK(vx_copy_to_dev(A_buffer, h_A.data(), 0, sizeA * sizeof(itype_t)));
  std::cout << "upload matrix B buffer" << std::endl;
  RT_CHECK(vx_copy_to_dev(B_buffer, h_B.data(), 0, sizeB * sizeof(itype_t)));

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
  RT_CHECK(vx_copy_from_dev(h_C.data(), C_buffer, 0, sizeC * sizeof(otype_t)));
#if COOP_PROFILE_STATS
  std::cout << "download stats buffer" << std::endl;
  RT_CHECK(vx_copy_from_dev(h_stats.data(), stats_buffer, 0, stats_buf_size));
#endif

  std::cout << "verify result" << std::endl;
  matmul_cpu(h_ref.data(), h_A.data(), h_B.data(), M, N, K);

  int errors = 0;
  int32_t max_ulp_delta = 0;
  uint32_t max_ulp_index = 0;
  uint32_t max_ulp_expected_bits = 0;
  uint32_t max_ulp_actual_bits = 0;
  for (uint32_t i = 0; i < sizeC; ++i) {
    auto ulp_delta = fp32_ulp_delta(h_C[i], h_ref[i]);
    if (ulp_delta > max_ulp_delta) {
      max_ulp_delta = ulp_delta;
      max_ulp_index = i;
      max_ulp_expected_bits = fp32_bits(h_ref[i]);
      max_ulp_actual_bits = fp32_bits(h_C[i]);
    }
    if (!compare_fp32(h_C[i], h_ref[i], i, errors)) {
      ++errors;
    }
  }
  std::cout << "max ULP delta: " << max_ulp_delta
            << " at index " << max_ulp_index
            << " expected_bits=0x" << std::hex << max_ulp_expected_bits
            << " actual_bits=0x" << max_ulp_actual_bits << std::dec << std::endl;

#if COOP_PROFILE_STATS
  uint32_t macro_grid_x = grid_x / 2;
  uint32_t num_k_panels = num_k_tiles / PANEL_K_TILES;
  uint32_t total_dxa_commands = 0;
  uint32_t total_dxa_a_panels = 0;
  uint32_t total_dxa_b_panels = 0;
  uint32_t total_team_barriers = 0;
  uint32_t total_dxa_stream_commands = 0;
  uint32_t total_dxa_slot_waits = 0;
  uint32_t total_dxa_global_bytes = 0;
  uint32_t total_baseline_global_bytes = 0;
  uint32_t total_avoided_global_bytes = 0;
  uint32_t total_panel_read_bytes = 0;
  uint32_t total_mma_steps = 0;
  uint32_t total_partial_panels = 0;
#if COOP_TIMING_STATS
  uint64_t total_timing_total_cycles = 0;
  uint64_t total_timing_dxa_wait_cycles = 0;
  uint64_t total_timing_panel_load_cycles = 0;
  uint64_t total_timing_mma_cycles = 0;
  uint64_t total_timing_store_cycles = 0;
#endif
  uint32_t has_partial_panel = ((num_k_tiles % PANEL_K_TILES) != 0) ? 1 : 0;

  for (uint32_t by = 0; by < grid_y; ++by) {
    for (uint32_t bx = 0; bx < grid_x; ++bx) {
      uint32_t idx = bx + by * grid_x;
      auto& stats = h_stats.at(idx);
      uint32_t expected_team_id = (bx / 2) + (by / 2) * macro_grid_x;
      uint32_t expected_rank_x = bx % 2;
      uint32_t expected_rank_y = by % 2;
      uint32_t expected_rank = expected_rank_y * 2 + expected_rank_x;
      uint32_t expected_commands = (expected_rank == 0) ? 1 : 0;
      uint32_t expected_stream_commands = expected_commands;
      uint32_t expected_slot_waits = num_k_panels;
      uint32_t expected_barriers = 0;
      uint32_t expected_a_panels = (expected_rank == 0) ? num_k_panels * 2 : 0;
      uint32_t expected_b_panels = (expected_rank == 0) ? num_k_panels * 2 : 0;
      uint32_t expected_baseline_bytes = num_k_tiles
                                       * (COOP_M_TILES * cfg::tileM * cfg::tileK
                                        + COOP_N_TILES * cfg::tileK * cfg::tileN)
                                       * sizeof(itype_t);
      uint32_t expected_panel_read_bytes = expected_baseline_bytes;
      uint32_t expected_mma_steps = num_k_tiles * COOP_M_TILES * COOP_N_TILES;
      uint32_t expected_partial_panels = has_partial_panel;
      if (stats.team_id != expected_team_id
       || stats.team_rank_x != expected_rank_x
       || stats.team_rank_y != expected_rank_y
       || stats.dxa_commands != expected_commands
       || stats.dxa_a_panels != expected_a_panels
       || stats.dxa_b_panels != expected_b_panels
       || stats.team_barriers != expected_barriers
       || stats.dxa_stream_commands != expected_stream_commands
       || stats.dxa_slot_waits != expected_slot_waits
       || stats.baseline_global_bytes != expected_baseline_bytes
       || stats.panel_read_bytes != expected_panel_read_bytes
       || stats.mma_steps != expected_mma_steps
       || stats.partial_panels != expected_partial_panels) {
        std::cout << "*** stats error: block (" << bx << ", " << by << ")"
                  << " team_id=" << stats.team_id
                  << " rank=(" << stats.team_rank_x << ", " << stats.team_rank_y << ")"
                  << " commands=" << stats.dxa_commands
                  << " a_panels=" << stats.dxa_a_panels
                  << " b_panels=" << stats.dxa_b_panels
                  << " barriers=" << stats.team_barriers
                  << " stream_commands=" << stats.dxa_stream_commands
                  << " slot_waits=" << stats.dxa_slot_waits
                  << " dxa_global_bytes=" << stats.dxa_global_bytes
                  << " baseline_global_bytes=" << stats.baseline_global_bytes
                  << " avoided_global_bytes=" << stats.avoided_global_bytes
                  << " panel_read_bytes=" << stats.panel_read_bytes
                  << " mma_steps=" << stats.mma_steps
                  << " partial_panels=" << stats.partial_panels
                  << std::endl;
        ++errors;
      }
      total_dxa_commands += stats.dxa_commands;
      total_dxa_a_panels += stats.dxa_a_panels;
      total_dxa_b_panels += stats.dxa_b_panels;
      total_team_barriers += stats.team_barriers;
      total_dxa_stream_commands += stats.dxa_stream_commands;
      total_dxa_slot_waits += stats.dxa_slot_waits;
      total_dxa_global_bytes += stats.dxa_global_bytes;
      total_baseline_global_bytes += stats.baseline_global_bytes;
      total_avoided_global_bytes += stats.avoided_global_bytes;
      total_panel_read_bytes += stats.panel_read_bytes;
      total_mma_steps += stats.mma_steps;
      total_partial_panels += stats.partial_panels;
#if COOP_TIMING_STATS
      total_timing_total_cycles += stats.timing_total_cycles;
      total_timing_dxa_wait_cycles += stats.timing_dxa_wait_cycles;
      total_timing_panel_load_cycles += stats.timing_panel_load_cycles;
      total_timing_mma_cycles += stats.timing_mma_cycles;
      total_timing_store_cycles += stats.timing_store_cycles;
#endif
    }
  }

  std::cout << "dxa commands: " << total_dxa_commands << std::endl;
  std::cout << "dxa unique A panels: " << total_dxa_a_panels << std::endl;
  std::cout << "dxa unique B panels: " << total_dxa_b_panels << std::endl;
  std::cout << "team barriers: " << total_team_barriers << std::endl;
  std::cout << "dxa stream commands: " << total_dxa_stream_commands << std::endl;
  std::cout << "dxa slot waits: " << total_dxa_slot_waits << std::endl;
  std::cout << "baseline local-staging global bytes: " << total_baseline_global_bytes << std::endl;
  std::cout << "dxa unique global bytes: " << total_dxa_global_bytes << std::endl;
  std::cout << "dxa avoided global bytes: " << total_avoided_global_bytes << std::endl;
  std::cout << "panel read bytes: " << total_panel_read_bytes << std::endl;
  std::cout << "mma steps: " << total_mma_steps << std::endl;
  std::cout << "partial panels: " << total_partial_panels << std::endl;
  if (total_dxa_global_bytes != 0) {
    std::cout << "global byte reduction: "
              << double(total_baseline_global_bytes) / double(total_dxa_global_bytes)
              << "x" << std::endl;
  }
  if (total_dxa_commands != 0) {
    std::cout << "mma steps per dxa command: "
              << double(total_mma_steps) / double(total_dxa_commands) << std::endl;
  }
#if COOP_TIMING_STATS
  std::cout << "timing total cycles: " << total_timing_total_cycles << std::endl;
  std::cout << "timing dxa wait cycles: " << total_timing_dxa_wait_cycles << std::endl;
  std::cout << "timing panel load cycles: " << total_timing_panel_load_cycles << std::endl;
  std::cout << "timing mma cycles: " << total_timing_mma_cycles << std::endl;
  std::cout << "timing store cycles: " << total_timing_store_cycles << std::endl;
#endif
#endif

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
