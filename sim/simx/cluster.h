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

#pragma once

#include <simobject.h>
#include <unordered_map>
#include "dcrs.h"
#include "arch.h"
#include "cache_cluster.h"
#include "local_mem.h"
#include "core.h"
#include "socket.h"
#include "constants.h"

namespace vortex {

class ProcessorImpl;
class RAM;

class Cluster : public SimObject<Cluster> {
public:
  struct PerfStats {
    CacheSim::PerfStats l2cache;
  };

  std::vector<SimPort<MemReq>> mem_req_ports;
  std::vector<SimPort<MemRsp>> mem_rsp_ports;

  Cluster(const SimContext& ctx,
          uint32_t cluster_id,
          ProcessorImpl* processor,
          const Arch &arch,
          const DCRS &dcrs);

  ~Cluster();

  uint32_t id() const {
    return cluster_id_;
  }

  ProcessorImpl* processor() const {
    return processor_;
  }

  void reset();

  void tick();

  void attach_ram(RAM* ram);

  #ifdef VM_ENABLE
  void set_satp(uint64_t satp);
  #endif

  bool running() const;

  int get_exitcode() const;

  void barrier(uint32_t bar_id, uint32_t count, uint32_t core_id);

  void cooperative_barrier(uint32_t core_id);
  void cooperative_arrive(uint32_t core_id);
  void cooperative_wait(uint32_t core_id);
  void cooperative_dxa_start(uint32_t core_id);
  void cooperative_dxa_wait_slot(uint32_t core_id, uint32_t slot);
  bool is_team_panel_addr(uint64_t addr) const;
  void team_panel_read(uint32_t core_id, void* data, uint64_t addr, uint32_t size);

  PerfStats perf_stats() const;

  const std::vector<Socket::Ptr>& sockets() const {
    return sockets_;
  }

private:
  struct TeamState {
    struct CopyDesc {
      uint32_t copy_idx;
      uint32_t copy_mode;
      uint64_t global_addr;
      uint32_t src_offset;
      uint32_t copy_size;
      uint32_t dst_mask;
      uint32_t tile_rows;
      uint32_t global_stride;
      uint32_t src_core_id;
    };

    uint32_t team_size;
    uint32_t team_size_x;
    CoreMask arrived;
    CoreMask waiters;
    bool transfer_ready;
    uint32_t transfer_cycles;
    uint32_t transfer_epoch;
    uint32_t ready_epoch;
    std::vector<uint32_t> participants;
    std::vector<uint32_t> rank_to_core;
    std::vector<CopyDesc> pending_copies;
    std::vector<uint8_t> panel_store;
    std::unordered_map<uint32_t, uint32_t> wait_epoch;
    bool dxa_stream_active;
    bool dxa_stream_inflight;
    uint32_t dxa_stream_panel_count;
    uint32_t dxa_stream_next_issue;
    uint32_t dxa_stream_inflight_panel;
    uint32_t dxa_stream_inflight_slot;
    uint32_t dxa_stream_panel_slot_bytes;
    CopyDesc dxa_stream_base[2];
    std::vector<CopyDesc> dxa_stream_pending_copies;
    uint32_t dxa_slot_panel[2];
    bool dxa_slot_ready[2];
    CoreMask dxa_slot_waiters[2];
    std::unordered_map<uint32_t, uint32_t> dxa_next_wait_panel;

    TeamState()
      : team_size(0)
      , team_size_x(0)
      , transfer_ready(false)
      , transfer_cycles(0)
      , transfer_epoch(0)
      , ready_epoch(0)
      , dxa_stream_active(false)
      , dxa_stream_inflight(false)
      , dxa_stream_panel_count(0)
      , dxa_stream_next_issue(0)
      , dxa_stream_inflight_panel(0)
      , dxa_stream_inflight_slot(0)
      , dxa_stream_panel_slot_bytes(0)
    {
      for (uint32_t i = 0; i < 2; ++i) {
        dxa_stream_base[i] = {};
        dxa_slot_panel[i] = 0xffffffff;
        dxa_slot_ready[i] = false;
      }
    }
  };

  Core* get_core(uint32_t local_core_id) const;
  std::vector<uint8_t> fetch_global_tile(const cooperative_ctx_t& ctx, uint32_t copy_idx) const;
  std::vector<uint8_t> fetch_global_tile(const TeamState::CopyDesc& copy_desc) const;
  TeamState& get_team_state(uint32_t local_core_id, const cooperative_ctx_t& coop);
  void write_team_panel(TeamState& team, uint32_t panel_offset, const std::vector<uint8_t>& data);
  void execute_dxa_panel_copy(TeamState& team, const TeamState::CopyDesc& copy_desc);
  void complete_dxa_stream_transfer(TeamState& team);
  void execute_pending_copies(TeamState& team);
  void resume_waiters(TeamState& team);
  void resume_dxa_slot_waiters(TeamState& team);
  void try_schedule_dxa_stream(TeamState& team);
  bool can_overwrite_dxa_slot(const TeamState& team, uint32_t slot) const;
  uint32_t estimate_transfer_cycles(const TeamState& team, const std::vector<TeamState::CopyDesc>& copies) const;
  uint32_t estimate_transfer_cycles(const TeamState& team) const;

  uint32_t                    cluster_id_;
  ProcessorImpl*              processor_;
  RAM*                        ram_;
  std::vector<Socket::Ptr>    sockets_;
  std::vector<CoreMask>       barriers_;
  std::unordered_map<uint32_t, TeamState> cooperative_teams_;
  CacheSim::Ptr               l2cache_;
  uint32_t                    cores_per_socket_;
};

} // namespace vortex
