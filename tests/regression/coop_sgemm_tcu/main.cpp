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
static_assert(cfg::tileM == cfg::tileK, "coop_sgemm_tcu assumes tileM == tileK for shared multicast shape");

static const char* kernel_file = "kernel.vxbin";
static uint32_t xm = 32;
static uint32_t xn = 32;
static uint32_t xk = 32;
static uint32_t mode = 0;

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

static uint32_t env_u32(const char* name, uint32_t default_value) {
  auto value = getenv(name);
  if (value == nullptr || value[0] == '\0')
    return default_value;
  return strtoul(value, nullptr, 0);
}

static bool compare_fp32(float actual, float expected, uint32_t index, uint32_t errors) {
  union fi_t {
    float f;
    int32_t i;
  };
  fi_t fa, fb;
  fa.f = actual;
  fb.f = expected;
  auto d = std::abs(fa.i - fb.i);
  if (d > FLOAT_ULP) {
    if (errors < MAX_ERRORS) {
      std::cout << "*** error: [" << index << "] expected=" << expected
                << ", actual=" << actual << std::endl;
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
  std::cout << "Vortex cooperative tensor GEMM test.\n";
  std::cout << "Usage: [-f kernel] [-m rows] [-n cols] [-k inner] [-c 0|1] [-h]\n";
  std::cout << "  -c 0: shared-panel async\n";
  std::cout << "  -c 1: oracle shared-panel async\n";
}

static void parse_args(int argc, char** argv) {
  int c;
  while ((c = getopt(argc, argv, "m:n:k:c:f:h")) != -1) {
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
    case 'c':
      mode = atoi(optarg);
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
  if (mode > 1) {
    std::cout << "Error: copy mode must be 0 (shared-panel async) or 1 (oracle shared-panel async)" << std::endl;
    cleanup();
    return -1;
  }
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
  uint32_t local_mem = 0;
  uint32_t num_k_tiles = K / cfg::tileK;
  uint32_t num_k_panels = (num_k_tiles + PANEL_K_TILES - 1) / PANEL_K_TILES;
  constexpr uint32_t team_panel_window = 0x10000;
  uint32_t a_panel_bytes = PANEL_K_TILES * cfg::tileM * cfg::tileK * COOP_M_TILES * sizeof(itype_t);
  uint32_t b_panel_bytes = PANEL_K_TILES * cfg::tileK * cfg::tileN * COOP_N_TILES * sizeof(itype_t);
  uint32_t panel_slot_bytes = 2 * a_panel_bytes + 2 * b_panel_bytes;
  uint32_t active_panel_slots = (num_k_panels > 1) ? 2 : 1;
  uint32_t required_panel_bytes = active_panel_slots * panel_slot_bytes;
  if (required_panel_bytes > team_panel_window) {
    std::cout << "Error: panel store window too small: needed=" << required_panel_bytes
              << ", available=" << team_panel_window
              << " (PANEL_K_TILES=" << PANEL_K_TILES
              << ", COOP_N_TILES=" << COOP_N_TILES
              << ", active_slots=" << active_panel_slots << ")" << std::endl;
    cleanup();
    return -1;
  }

  kernel_arg.grid_dim[0] = grid_x;
  kernel_arg.grid_dim[1] = grid_y;
  kernel_arg.block_dim[0] = NUM_THREADS;
  kernel_arg.block_dim[1] = 1;
  kernel_arg.team_dim[0] = 2;
  kernel_arg.team_dim[1] = 2;
  kernel_arg.mode = mode;
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
  std::cout << "profile stats: " << (COOP_PROFILE_STATS ? "on" : "off") << std::endl;
  std::cout << "tensor feed oracle: " << (COOP_TENSOR_FEED_ORACLE ? "on" : "off") << std::endl;
  if (COOP_TENSOR_FEED_ORACLE) {
    std::cout << "tensor feed latency model: desc="
              << env_u32("VORTEX_TEAM_TENSOR_DESC_LATENCY", 0)
              << ", mma="
              << env_u32("VORTEX_TEAM_TENSOR_MMA_LATENCY", 0)
              << ", bytes/cycle="
              << env_u32("VORTEX_TEAM_TENSOR_BYTES_PER_CYCLE", 0)
              << std::endl;
  }
  std::cout << "copy mode: " << (mode == 0 ? "shared-panel-async" : "oracle-panel-async") << std::endl;
  std::cout << "local memory: " << local_mem << " bytes" << std::endl;

  uint32_t max_localmem = 0;
  RT_CHECK(vx_check_occupancy(device, NUM_THREADS, &max_localmem));
  std::cout << "occupancy: max_localmem=" << max_localmem << " bytes" << std::endl;
  if (max_localmem < local_mem) {
    std::cout << "Error: not enough local memory: needed=" << local_mem
              << ", available=" << max_localmem << std::endl;
    cleanup();
    return -1;
  }

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
  for (uint32_t i = 0; i < sizeC; ++i) {
    if (!compare_fp32(h_C[i], h_ref[i], i, errors)) {
      ++errors;
    }
  }

#if COOP_PROFILE_STATS
  uint32_t macro_grid_x = grid_x / 2;
  uint32_t total_a_tile_loads = 0;
  uint32_t total_b_tile_loads = 0;
  uint32_t total_team_barriers = 0;
  uint32_t total_team_arrives = 0;
  uint32_t total_team_waits = 0;
  uint32_t total_transfer_events = 0;
  uint32_t total_global_bytes = 0;
  uint32_t total_fanout_bytes = 0;
  uint32_t total_panel_write_bytes = 0;
  uint32_t total_panel_read_bytes = 0;
  uint32_t total_panel_transfers = 0;
  uint32_t total_mma_steps = 0;
  uint32_t total_partial_panels = 0;
  uint32_t total_slot_reuse_barriers = 0;
  uint32_t has_partial_panel = ((num_k_tiles % PANEL_K_TILES) != 0) ? 1 : 0;
  for (uint32_t by = 0; by < grid_y; ++by) {
    for (uint32_t bx = 0; bx < grid_x; ++bx) {
      uint32_t idx = bx + by * grid_x;
      auto& stats = h_stats.at(idx);
      uint32_t expected_team_id = (bx / 2) + (by / 2) * macro_grid_x;
      uint32_t expected_rank_x = bx % 2;
      uint32_t expected_rank_y = by % 2;
      uint32_t expected_a_loads = (expected_rank_x == 0) ? (num_k_tiles * COOP_M_TILES) : 0;
      uint32_t expected_b_loads = (expected_rank_y == 0) ? (num_k_tiles * COOP_N_TILES) : 0;
      uint32_t expected_transfer_events = 0;
      expected_transfer_events += (expected_rank_x == 0) ? num_k_panels : 0;
      expected_transfer_events += (expected_rank_y == 0) ? num_k_panels : 0;
      uint32_t expected_global_bytes = (expected_a_loads * cfg::tileM * cfg::tileK
                                      + expected_b_loads * cfg::tileK * cfg::tileN) * sizeof(itype_t);
      uint32_t expected_fanout_bytes = 0;
      uint32_t expected_panel_write_bytes = expected_global_bytes;
      uint32_t expected_panel_read_bytes = num_k_tiles
                                         * (COOP_M_TILES * cfg::tileM * cfg::tileK
                                          + COOP_N_TILES * cfg::tileK * cfg::tileN)
                                         * sizeof(itype_t);
      uint32_t expected_panel_transfers = expected_transfer_events;
      uint32_t expected_mma_steps = num_k_tiles * COOP_M_TILES * COOP_N_TILES;
      uint32_t expected_partial_panels = has_partial_panel;
      uint32_t sync_steps = num_k_panels;
      uint32_t expected_arrives = (sync_steps == 0) ? 0 : (sync_steps - 1);
      uint32_t reuse_barriers = (num_k_panels > 2) ? (num_k_panels - 2) : 0;
      uint32_t expected_barriers = (sync_steps == 0) ? 0 : 1 + reuse_barriers;
      uint32_t expected_waits = expected_arrives;
      if (stats.team_id != expected_team_id
       || stats.team_rank_x != expected_rank_x
       || stats.team_rank_y != expected_rank_y
       || stats.a_tile_loads != expected_a_loads
       || stats.b_tile_loads != expected_b_loads
       || stats.team_barriers != expected_barriers
       || stats.team_arrives != expected_arrives
       || stats.team_waits != expected_waits
       || stats.transfer_events != expected_transfer_events
       || stats.global_bytes != expected_global_bytes
       || stats.fanout_bytes != expected_fanout_bytes
       || stats.panel_write_bytes != expected_panel_write_bytes
       || stats.panel_read_bytes != expected_panel_read_bytes
       || stats.panel_transfers != expected_panel_transfers
       || stats.mma_steps != expected_mma_steps
       || stats.partial_panels != expected_partial_panels
       || stats.slot_reuse_barriers != reuse_barriers) {
        std::cout << "*** stats error: block (" << bx << ", " << by << ")"
                  << " team_id=" << stats.team_id
                  << " rank=(" << stats.team_rank_x << ", " << stats.team_rank_y << ")"
                  << " a_tiles=" << stats.a_tile_loads
                  << " b_tiles=" << stats.b_tile_loads
                  << " barriers=" << stats.team_barriers
                  << " arrives=" << stats.team_arrives
                  << " waits=" << stats.team_waits
                  << " transfers=" << stats.transfer_events
                  << " global_bytes=" << stats.global_bytes
                  << " fanout_bytes=" << stats.fanout_bytes
                  << " panel_write_bytes=" << stats.panel_write_bytes
                  << " panel_read_bytes=" << stats.panel_read_bytes
                  << " panel_transfers=" << stats.panel_transfers
                  << " mma_steps=" << stats.mma_steps
                  << " partial_panels=" << stats.partial_panels
                  << " slot_reuse_barriers=" << stats.slot_reuse_barriers
                  << std::endl;
        ++errors;
      }
      total_a_tile_loads += stats.a_tile_loads;
      total_b_tile_loads += stats.b_tile_loads;
      total_team_barriers += stats.team_barriers;
      total_team_arrives += stats.team_arrives;
      total_team_waits += stats.team_waits;
      total_transfer_events += stats.transfer_events;
      total_global_bytes += stats.global_bytes;
      total_fanout_bytes += stats.fanout_bytes;
      total_panel_write_bytes += stats.panel_write_bytes;
      total_panel_read_bytes += stats.panel_read_bytes;
      total_panel_transfers += stats.panel_transfers;
      total_mma_steps += stats.mma_steps;
      total_partial_panels += stats.partial_panels;
      total_slot_reuse_barriers += stats.slot_reuse_barriers;
    }
  }

  uint32_t cooperative_tile_loads = total_a_tile_loads + total_b_tile_loads;
  uint32_t baseline_tile_loads = block_count * num_k_tiles * COOP_M_TILES * COOP_N_TILES * 2;
  std::cout << "cooperative tile loads: " << cooperative_tile_loads
            << " (A=" << total_a_tile_loads
            << ", B=" << total_b_tile_loads << ")" << std::endl;
  std::cout << "baseline tile loads: " << baseline_tile_loads << std::endl;
  std::cout << "team barriers: " << total_team_barriers << std::endl;
  std::cout << "team arrives: " << total_team_arrives << std::endl;
  std::cout << "team waits: " << total_team_waits << std::endl;
  std::cout << "slot reuse barriers: " << total_slot_reuse_barriers << std::endl;
  std::cout << "transfer events: " << total_transfer_events << std::endl;
  std::cout << "global transfer bytes: " << total_global_bytes << std::endl;
  std::cout << "fanout write bytes: " << total_fanout_bytes << std::endl;
  std::cout << "panel write bytes: " << total_panel_write_bytes << std::endl;
  std::cout << "panel read bytes: " << total_panel_read_bytes << std::endl;
  std::cout << "panel transfers: " << total_panel_transfers << std::endl;
  std::cout << "mma steps: " << total_mma_steps << std::endl;
  std::cout << "partial panels: " << total_partial_panels << std::endl;
  if (total_transfer_events != 0) {
    std::cout << "mma steps per transfer event: "
              << double(total_mma_steps) / double(total_transfer_events) << std::endl;
  }
  if (total_team_barriers != 0) {
    std::cout << "mma steps per team barrier: "
              << double(total_mma_steps) / double(total_team_barriers) << std::endl;
  }
  uint32_t total_team_sync_events = total_team_barriers + total_team_arrives + total_team_waits;
  if (total_team_sync_events != 0) {
    std::cout << "mma steps per team sync event: "
              << double(total_mma_steps) / double(total_team_sync_events) << std::endl;
  }
  if (cooperative_tile_loads == 0) {
    std::cout << "*** load reduction check failed: no cooperative loads recorded" << std::endl;
    ++errors;
  } else {
    std::cout << "tile load reduction: "
              << double(baseline_tile_loads) / double(cooperative_tile_loads)
              << "x" << std::endl;
  }
#endif

#if COOP_TENSOR_FEED_ORACLE
  uint64_t tensor_commands = block_count * COOP_M_TILES;
  uint64_t tensor_mma_steps = tensor_commands * num_k_tiles * COOP_N_TILES;
  uint64_t tensor_input_bytes = tensor_commands
                              * (uint64_t(cfg::tileM) * K
                               + uint64_t(K) * cfg::tileN * COOP_N_TILES)
                              * sizeof(itype_t);
  uint64_t tensor_output_bytes = tensor_commands
                               * uint64_t(cfg::tileM) * cfg::tileN * COOP_N_TILES
                               * sizeof(otype_t);
  uint64_t avoided_matrix_loads = tensor_commands * num_k_tiles * (COOP_M_TILES + COOP_N_TILES);
  uint64_t oracle_cooperative_tile_loads = 0;
  for (uint32_t by = 0; by < grid_y; ++by) {
    for (uint32_t bx = 0; bx < grid_x; ++bx) {
      oracle_cooperative_tile_loads += (bx % 2 == 0) ? (num_k_tiles * COOP_M_TILES) : 0;
      oracle_cooperative_tile_loads += (by % 2 == 0) ? (num_k_tiles * COOP_N_TILES) : 0;
    }
  }
  uint64_t oracle_baseline_tile_loads = block_count * num_k_tiles * COOP_M_TILES * COOP_N_TILES * 2;
  std::cout << "tensor feed commands: " << tensor_commands << std::endl;
  std::cout << "tensor feed mma steps: " << tensor_mma_steps << std::endl;
  std::cout << "tensor feed input bytes: " << tensor_input_bytes << std::endl;
  std::cout << "tensor feed output bytes: " << tensor_output_bytes << std::endl;
  std::cout << "software matrix loads avoided: " << avoided_matrix_loads << std::endl;
  std::cout << "software mma steps offloaded: " << tensor_mma_steps << std::endl;
  std::cout << "cooperative tile loads: " << oracle_cooperative_tile_loads << std::endl;
  std::cout << "baseline tile loads: " << oracle_baseline_tile_loads << std::endl;
  if (oracle_cooperative_tile_loads != 0) {
    std::cout << "tile load reduction: "
              << double(oracle_baseline_tile_loads) / double(oracle_cooperative_tile_loads)
              << "x" << std::endl;
  }
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
