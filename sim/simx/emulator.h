// Copyright © 2019-2023
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef __WARP_H
#define __WARP_H

#include <vector>
#include <sstream>
#include <stack>
#include <mem.h>
#include "types.h"
#include "instr.h"
#ifdef EXT_TCU_ENABLE
#include "tcu/tensor_unit.h"
#endif
#ifdef EXT_V_ENABLE
#include "vpu/vec_unit.h"
#endif

class DebugModule;  // Forward declaration (global scope)

namespace vortex {

class Arch;
class DCRS;
class Core;
class Instr;
class instr_trace_t;

struct ipdom_entry_t {
  ThreadMask  orig_tmask;
  ThreadMask  else_tmask;
  Word        PC;
  bool        fallthrough;

  ipdom_entry_t(const ThreadMask &orig_tmask, const ThreadMask &else_tmask, Word PC)
    : orig_tmask (orig_tmask)
    , else_tmask (else_tmask)
    , PC         (PC)
    , fallthrough(false)
  {}
};

///////////////////////////////////////////////////////////////////////////////

struct warp_t {
  std::vector<std::vector<Word>>    ireg_file;
  std::vector<std::vector<uint64_t>>freg_file;
  std::deque<Instr::Ptr>            ibuffer;
  std::stack<ipdom_entry_t>         ipdom_stack;
  ThreadMask                        tmask;
  Word                              PC;
  Byte                              fcsr;
  uint32_t                          uuid;

  warp_t(uint32_t num_threads);

  void reset(uint64_t startup_addr);
};

///////////////////////////////////////////////////////////////////////////////

struct wspawn_t {
  bool      valid;
  uint32_t  num_warps;
  Word      nextPC;
};

struct cooperative_ctx_t {
  static constexpr uint32_t kMaxCopies = 2;

  uint32_t team_id;
  uint32_t team_rank_x;
  uint32_t team_rank_y;
  uint32_t team_size_x;
  uint32_t team_size_y;
  uint32_t tile_rows;
  uint32_t global_stride;
  bool team_panel_enabled;
  uint32_t copy_mode[kMaxCopies];
  uint64_t global_addr[kMaxCopies];
  uint32_t src_offset[kMaxCopies];
  uint32_t copy_size[kMaxCopies];
  uint32_t dst_mask[kMaxCopies];
  uint32_t copy_tile_rows[kMaxCopies];
  uint32_t copy_global_stride[kMaxCopies];
  uint32_t tensor_a_offset;
  uint32_t tensor_b_offset;
  uint64_t tensor_c_addr;
  uint32_t tensor_c_stride;
  uint32_t tensor_a_stride;
  uint32_t tensor_b_stride;
  uint32_t tensor_k_tiles;

  cooperative_ctx_t()
    : team_id(0xffffffff)
    , team_rank_x(0)
    , team_rank_y(0)
    , team_size_x(0)
    , team_size_y(0)
    , tile_rows(0)
    , global_stride(0)
    , team_panel_enabled(false)
    , tensor_a_offset(0)
    , tensor_b_offset(0)
    , tensor_c_addr(0)
    , tensor_c_stride(0)
    , tensor_a_stride(0)
    , tensor_b_stride(0)
    , tensor_k_tiles(0)
  {
    for (uint32_t i = 0; i < kMaxCopies; ++i) {
      copy_mode[i] = 0;
      global_addr[i] = 0;
      src_offset[i] = 0;
      copy_size[i] = 0;
      dst_mask[i] = 0;
      copy_tile_rows[i] = 0;
      copy_global_stride[i] = 0;
    }
  }
};

///////////////////////////////////////////////////////////////////////////////

class Emulator {
public:
  Emulator(const Arch &arch, const DCRS &dcrs, Core* core);

  ~Emulator();

  void reset();

  void attach_ram(RAM* ram);

#ifdef VM_ENABLE
  void set_satp(uint64_t satp) ;
#endif

  instr_trace_t* step();

  bool running() const;

  void suspend(uint32_t wid);

  void resume(uint32_t wid);

  bool barrier(uint32_t bar_id, uint32_t count, uint32_t wid);

  bool wspawn(uint32_t num_warps, Word nextPC);

  int get_exitcode() const;

  void dcache_read(void* data, uint64_t addr, uint32_t size);

  void dcache_write(const void* data, uint64_t addr, uint32_t size);

  // Get warp by index (for debug module access)
  warp_t& get_warp(uint32_t wid) {
    return warps_.at(wid);
  }

  // Debug module interface
  void set_debug_module(::DebugModule* dm);
  ::DebugModule* get_debug_module() const;

  const cooperative_ctx_t& cooperative_ctx() const {
    return cooperative_ctx_;
  }

  uint32_t take_csr_latency() {
    uint32_t value = csr_latency_;
    csr_latency_ = 0;
    return value;
  }

  void clear_cooperative_copy();

private:

  uint32_t fetch(uint32_t wid, uint64_t uuid);

  void decode(uint32_t code, uint32_t wid, uint64_t uuid);

  instr_trace_t* execute(const Instr &instr, uint32_t wid);

  void fetch_registers(std::vector<reg_data_t>& out, uint32_t wid, uint32_t src_index, const RegOpd& reg);

  void icache_read(void* data, uint64_t addr, uint32_t size);

  void dcache_amo_reserve(uint64_t addr);

  bool dcache_amo_check(uint64_t addr);

  void writeToStdOut(const void* data, uint64_t addr, uint32_t size);

  void cout_flush();

  Word get_csr(uint32_t addr, uint32_t wid, uint32_t tid);

  void set_csr(uint32_t addr, Word value, uint32_t wid, uint32_t tid);

  uint32_t get_fpu_rm(uint32_t funct3, uint32_t wid, uint32_t tid);

  void update_fcrs(uint32_t fflags, uint32_t wid, uint32_t tid);

  // temporarily added for riscv-vector tests
  // TODO: remove once ecall/ebreak are supported
  void trigger_ecall();
  void trigger_ebreak();

  const Arch& arch_;
  const DCRS& dcrs_;
  Core*       core_;
  ::DebugModule* debug_module_;

  std::vector<warp_t> warps_;
  WarpMask    active_warps_;
  WarpMask    stalled_warps_;
  WarpMask    cooperative_barrier_;
  std::vector<WarpMask> barriers_;
  std::unordered_map<int, std::stringstream> print_bufs_;
  MemoryUnit  mmu_;
  uint32_t    ipdom_size_;
  Word        csr_mscratch_;
  wspawn_t    wspawn_;
  cooperative_ctx_t cooperative_ctx_;
  uint32_t    csr_latency_;

  // PC of the last warp to become inactive, used by the debug module to
  // report the final PC when the program completes.
  Word        last_inactive_warp_pc_ = 0;
  bool        last_inactive_warp_pc_valid_ = false;

#ifdef EXT_TCU_ENABLE
  TensorUnit::Ptr tensor_unit_;
#endif

#ifdef EXT_V_ENABLE
  VecUnit::Ptr vec_unit_;
#endif

  PoolAllocator<Instr, 64> instr_pool_;
};

}

#endif
