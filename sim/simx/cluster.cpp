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

#include "cluster.h"
#include <algorithm>
#include <cstring>

using namespace vortex;

static constexpr uint32_t kGlobalMulticastMode = 1;
static constexpr uint32_t kTeamPanelMode = 2;
static constexpr uint32_t kTeamPanelOracleMode = 3;
static constexpr uint32_t kTeamPanelOffset = 0x2000;
static constexpr uint32_t kTeamPanelWindow = 0x10000;

static bool is_panel_mode(uint32_t copy_mode) {
  return copy_mode == kTeamPanelMode || copy_mode == kTeamPanelOracleMode;
}

Cluster::Cluster(const SimContext& ctx,
                 uint32_t cluster_id,
                 ProcessorImpl* processor,
                 const Arch &arch,
                 const DCRS &dcrs)
  : SimObject(ctx, StrFormat("cluster%d", cluster_id))
  , mem_req_ports(L2_MEM_PORTS, this)
  , mem_rsp_ports(L2_MEM_PORTS, this)
  , cluster_id_(cluster_id)
  , processor_(processor)
  , ram_(nullptr)
  , sockets_(NUM_SOCKETS)
  , barriers_(arch.num_barriers(), 0)
  , cores_per_socket_(arch.socket_size())
{
  char sname[100];

  uint32_t sockets_per_cluster = sockets_.size();

  // create sockets

  for (uint32_t i = 0; i < sockets_per_cluster; ++i) {
    uint32_t socket_id = cluster_id * sockets_per_cluster + i;
    sockets_.at(i) = Socket::Create(socket_id, this, arch, dcrs);
  }

  // Create l2cache

  snprintf(sname, 100, "%s-l2cache", this->name().c_str());
  l2cache_ = CacheSim::Create(sname, CacheSim::Config{
    !L2_ENABLED,
    log2ceil(L2_CACHE_SIZE),// C
    log2ceil(MEM_BLOCK_SIZE),// L
    log2ceil(L1_LINE_SIZE), // W
    log2ceil(L2_NUM_WAYS),  // A
    log2ceil(L2_NUM_BANKS), // B
    XLEN,                   // address bits
    L2_NUM_REQS,            // request size
    L2_MEM_PORTS,           // memory ports
    L2_WRITEBACK,           // write-back
    false,                  // write response
    L2_MSHR_SIZE,           // mshr size
    2,                      // pipeline latency
  });

  // connect l2cache core interfaces
  for (uint32_t i = 0; i < sockets_per_cluster; ++i) {
    for (uint32_t j = 0; j < L1_MEM_PORTS; ++j) {
      sockets_.at(i)->mem_req_ports.at(j).bind(&l2cache_->CoreReqPorts.at(i * L1_MEM_PORTS + j));
      l2cache_->CoreRspPorts.at(i * L1_MEM_PORTS + j).bind(&sockets_.at(i)->mem_rsp_ports.at(j));
    }
  }

  // connect l2cache memory interfaces
  for (uint32_t i = 0; i < L2_MEM_PORTS; ++i) {
    l2cache_->MemReqPorts.at(i).bind(&this->mem_req_ports.at(i));
    this->mem_rsp_ports.at(i).bind(&l2cache_->MemRspPorts.at(i));
  }
}

Cluster::~Cluster() {
  //--
}

void Cluster::reset() {
  for (auto& barrier : barriers_) {
    barrier.reset();
  }
  cooperative_teams_.clear();
}

void Cluster::tick() {
  for (auto& entry : cooperative_teams_) {
    auto& team = entry.second;
    if (team.transfer_cycles == 0)
      continue;

    --team.transfer_cycles;
    if (team.transfer_cycles == 0) {
      execute_pending_copies(team);
      team.ready_epoch = team.transfer_epoch;
      team.transfer_ready = true;
      resume_waiters(team);
    }
  }
}

void Cluster::attach_ram(RAM* ram) {
  ram_ = ram;
  for (auto& socket : sockets_) {
    socket->attach_ram(ram);
  }
}

#ifdef VM_ENABLE
void Cluster::set_satp(uint64_t satp) {
  for (auto& socket : sockets_) {
    socket->set_satp(satp);
  }
}
#endif

bool Cluster::running() const {
  for (auto& socket : sockets_) {
    if (socket->running())
      return true;
  }
  return false;
}

int Cluster::get_exitcode() const {
  int exitcode = 0;
  for (auto& socket : sockets_) {
    exitcode |= socket->get_exitcode();
  }
  return exitcode;
}

void Cluster::barrier(uint32_t bar_id, uint32_t count, uint32_t core_id) {
  auto& barrier = barriers_.at(bar_id);

  auto sockets_per_cluster = sockets_.size();
  auto cores_per_socket = cores_per_socket_;

  uint32_t cores_per_cluster = sockets_per_cluster * cores_per_socket;
  uint32_t local_core_id = core_id % cores_per_cluster;
  barrier.set(local_core_id);

  DP(3, "*** Suspend core #" << core_id << " at barrier #" << bar_id);

  if (barrier.count() == (size_t)count) {
      // resume all suspended cores
      for (uint32_t s = 0; s < sockets_per_cluster; ++s) {
        for (uint32_t c = 0; c < cores_per_socket; ++c) {
          uint32_t i = s * cores_per_socket + c;
          if (barrier.test(i)) {
            DP(3, "*** Resume core #" << i << " at barrier #" << bar_id);
            sockets_.at(s)->resume(c);
          }
        }
      }
      barrier.reset();
    }
}

void Cluster::cooperative_barrier(uint32_t core_id) {
  auto sockets_per_cluster = sockets_.size();
  auto cores_per_socket = cores_per_socket_;
  uint32_t cores_per_cluster = sockets_per_cluster * cores_per_socket;
  uint32_t local_core_id = core_id % cores_per_cluster;

  auto* core = this->get_core(local_core_id);
  auto& coop = core->emulator().cooperative_ctx();
  auto& team = get_team_state(local_core_id, coop);

  team.arrived.set(local_core_id);

  DP(3, "*** Suspend core #" << core_id << " in cooperative team #" << coop.team_id);

  if (team.arrived.count() != team.team_size)
    return;

  execute_pending_copies(team);

  for (auto local_id : team.participants) {
    auto* team_core = this->get_core(local_id);
    team_core->emulator().clear_cooperative_copy();
    uint32_t socket_id = local_id / cores_per_socket;
    uint32_t socket_core_id = local_id % cores_per_socket;
    DP(3, "*** Resume core #" << local_id << " in cooperative team #" << coop.team_id);
    sockets_.at(socket_id)->resume(socket_core_id);
  }

  team.arrived.reset();
  team.waiters.reset();
  team.transfer_ready = false;
  team.transfer_cycles = 0;
  team.ready_epoch = team.transfer_epoch;
  team.pending_copies.clear();
}

void Cluster::cooperative_arrive(uint32_t core_id) {
  auto sockets_per_cluster = sockets_.size();
  auto cores_per_socket = cores_per_socket_;
  uint32_t cores_per_cluster = sockets_per_cluster * cores_per_socket;
  uint32_t local_core_id = core_id % cores_per_cluster;

  auto* core = this->get_core(local_core_id);
  auto& coop = core->emulator().cooperative_ctx();
  auto& team = get_team_state(local_core_id, coop);

  if (!team.arrived.any())
    assert(team.transfer_cycles == 0);

  team.arrived.set(local_core_id);
  DP(3, "*** Arrive core #" << core_id << " in cooperative team #" << coop.team_id);

  if (team.arrived.count() != team.team_size)
    return;

  assert(team.transfer_cycles == 0);
  team.transfer_ready = false;
  team.pending_copies.clear();
  ++team.transfer_epoch;

  for (auto src_core_id : team.participants) {
    auto* src_core = this->get_core(src_core_id);
    auto& src_ctx = src_core->emulator().cooperative_ctx();
    for (uint32_t copy_idx = 0; copy_idx < cooperative_ctx_t::kMaxCopies; ++copy_idx) {
      if (src_ctx.copy_size[copy_idx] == 0 || src_ctx.dst_mask[copy_idx] == 0)
        continue;

      team.pending_copies.push_back(TeamState::CopyDesc{
        src_ctx.copy_mode[copy_idx],
        src_ctx.global_addr[copy_idx],
        src_ctx.src_offset[copy_idx],
        src_ctx.copy_size[copy_idx],
        src_ctx.dst_mask[copy_idx],
        src_ctx.copy_tile_rows[copy_idx],
        src_ctx.copy_global_stride[copy_idx],
        src_core_id,
      });
    }
    src_core->emulator().clear_cooperative_copy();
  }

  for (auto local_id : team.participants) {
    team.wait_epoch[local_id] = team.transfer_epoch;
  }
  team.transfer_cycles = estimate_transfer_cycles(team);
  if (team.transfer_cycles == 0) {
    execute_pending_copies(team);
    team.ready_epoch = team.transfer_epoch;
    team.transfer_ready = true;
  }
  team.arrived.reset();
}

void Cluster::cooperative_wait(uint32_t core_id) {
  auto sockets_per_cluster = sockets_.size();
  auto cores_per_socket = cores_per_socket_;
  uint32_t cores_per_cluster = sockets_per_cluster * cores_per_socket;
  uint32_t local_core_id = core_id % cores_per_cluster;

  auto* core = this->get_core(local_core_id);
  auto& coop = core->emulator().cooperative_ctx();
  auto& team = get_team_state(local_core_id, coop);

  auto target_epoch = team.wait_epoch[local_core_id];
  if (target_epoch == 0 || team.ready_epoch >= target_epoch) {
    uint32_t socket_id = local_core_id / cores_per_socket;
    uint32_t socket_core_id = local_core_id % cores_per_socket;
    sockets_.at(socket_id)->resume(socket_core_id);
    return;
  }

  team.waiters.set(local_core_id);
  DP(3, "*** Wait core #" << core_id << " in cooperative team #" << coop.team_id);
}

bool Cluster::is_team_panel_addr(uint64_t addr) const {
  auto local_offset = addr - uint64_t(LMEM_BASE_ADDR);
  return addr >= uint64_t(LMEM_BASE_ADDR)
      && local_offset >= kTeamPanelOffset
      && local_offset < (kTeamPanelOffset + kTeamPanelWindow);
}

void Cluster::team_panel_read(uint32_t core_id, void* data, uint64_t addr, uint32_t size) {
  auto sockets_per_cluster = sockets_.size();
  auto cores_per_socket = cores_per_socket_;
  uint32_t cores_per_cluster = sockets_per_cluster * cores_per_socket;
  uint32_t local_core_id = core_id % cores_per_cluster;

  auto* core = this->get_core(local_core_id);
  auto& coop = core->emulator().cooperative_ctx();
  auto& team = get_team_state(local_core_id, coop);

  uint32_t panel_offset = addr - uint64_t(LMEM_BASE_ADDR) - kTeamPanelOffset;
  assert(panel_offset + size <= team.panel_store.size());
  std::memcpy(data, team.panel_store.data() + panel_offset, size);
}

Core* Cluster::get_core(uint32_t local_core_id) const {
  uint32_t socket_id = local_core_id / cores_per_socket_;
  uint32_t socket_core_id = local_core_id % cores_per_socket_;
  return sockets_.at(socket_id)->cores().at(socket_core_id).get();
}

Cluster::PerfStats Cluster::perf_stats() const {
  PerfStats perf_stats;
  perf_stats.l2cache = l2cache_->perf_stats();
  return perf_stats;
}

std::vector<uint8_t> Cluster::fetch_global_tile(const cooperative_ctx_t& ctx, uint32_t copy_idx) const {
  assert(ram_ != nullptr);
  assert(ctx.copy_tile_rows[copy_idx] != 0);
  assert(ctx.copy_global_stride[copy_idx] != 0);
  assert((ctx.copy_size[copy_idx] % ctx.copy_tile_rows[copy_idx]) == 0);

  uint32_t packed_bytes = ctx.copy_size[copy_idx];
  uint32_t row_bytes = packed_bytes / ctx.copy_tile_rows[copy_idx];
  uint64_t span_bytes = uint64_t(ctx.copy_tile_rows[copy_idx] - 1) * ctx.copy_global_stride[copy_idx] + row_bytes;

  std::vector<uint8_t> packed_buffer(packed_bytes);
  if (row_bytes == ctx.copy_global_stride[copy_idx]) {
    ram_->read(packed_buffer.data(), ctx.global_addr[copy_idx], packed_bytes);
    return packed_buffer;
  }

  std::vector<uint8_t> span_buffer(span_bytes);
  ram_->read(span_buffer.data(), ctx.global_addr[copy_idx], span_bytes);
  for (uint32_t row = 0; row < ctx.copy_tile_rows[copy_idx]; ++row) {
    std::memcpy(packed_buffer.data() + row * row_bytes,
                span_buffer.data() + row * ctx.copy_global_stride[copy_idx],
                row_bytes);
  }
  return packed_buffer;
}

std::vector<uint8_t> Cluster::fetch_global_tile(const TeamState::CopyDesc& copy_desc) const {
  assert(ram_ != nullptr);
  assert(copy_desc.tile_rows != 0);
  assert(copy_desc.global_stride != 0);
  assert((copy_desc.copy_size % copy_desc.tile_rows) == 0);

  uint32_t packed_bytes = copy_desc.copy_size;
  uint32_t row_bytes = packed_bytes / copy_desc.tile_rows;
  uint64_t span_bytes = uint64_t(copy_desc.tile_rows - 1) * copy_desc.global_stride + row_bytes;

  std::vector<uint8_t> packed_buffer(packed_bytes);
  if (row_bytes == copy_desc.global_stride) {
    ram_->read(packed_buffer.data(), copy_desc.global_addr, packed_bytes);
    return packed_buffer;
  }

  std::vector<uint8_t> span_buffer(span_bytes);
  ram_->read(span_buffer.data(), copy_desc.global_addr, span_bytes);
  for (uint32_t row = 0; row < copy_desc.tile_rows; ++row) {
    std::memcpy(packed_buffer.data() + row * row_bytes,
                span_buffer.data() + row * copy_desc.global_stride,
                row_bytes);
  }
  return packed_buffer;
}

Cluster::TeamState& Cluster::get_team_state(uint32_t local_core_id, const cooperative_ctx_t& coop) {
  uint32_t team_size = coop.team_size_x * coop.team_size_y;
  uint32_t team_rank = coop.team_rank_y * coop.team_size_x + coop.team_rank_x;

  assert(team_size > 0);
  assert(team_rank < team_size);

  auto& team = cooperative_teams_[coop.team_id];
  if (team.team_size == 0) {
    team.team_size = team_size;
    team.team_size_x = coop.team_size_x;
    team.rank_to_core.resize(team_size, 0xffffffff);
  }

  team.rank_to_core.at(team_rank) = local_core_id;
  if (std::find(team.participants.begin(), team.participants.end(), local_core_id) == team.participants.end()) {
    team.participants.push_back(local_core_id);
  }

  return team;
}

void Cluster::write_team_panel(TeamState& team, uint32_t panel_offset, const std::vector<uint8_t>& data) {
  auto end_offset = panel_offset + data.size();
  if (team.panel_store.size() < end_offset) {
    team.panel_store.resize(end_offset);
  }
  std::memcpy(team.panel_store.data() + panel_offset, data.data(), data.size());
}

void Cluster::execute_pending_copies(TeamState& team) {
  for (auto src_core_id : team.participants) {
    auto* src_core = this->get_core(src_core_id);
    auto& src_ctx = src_core->emulator().cooperative_ctx();
    for (uint32_t copy_idx = 0; copy_idx < cooperative_ctx_t::kMaxCopies; ++copy_idx) {
      if (src_ctx.copy_size[copy_idx] == 0 || src_ctx.dst_mask[copy_idx] == 0)
        continue;

      std::vector<uint8_t> multicast_buffer;
      if (src_ctx.copy_mode[copy_idx] == kGlobalMulticastMode) {
        multicast_buffer = fetch_global_tile(src_ctx, copy_idx);
      } else if (is_panel_mode(src_ctx.copy_mode[copy_idx])) {
        multicast_buffer = fetch_global_tile(src_ctx, copy_idx);
        write_team_panel(team, src_ctx.src_offset[copy_idx], multicast_buffer);
        continue;
      }

      for (uint32_t dst_rank = 0; dst_rank < team.team_size; ++dst_rank) {
        if ((src_ctx.dst_mask[copy_idx] & (1u << dst_rank)) == 0)
          continue;
        auto dst_core_id = team.rank_to_core.at(dst_rank);
        assert(dst_core_id != 0xffffffff);
        auto* dst_core = this->get_core(dst_core_id);
        if (src_ctx.copy_mode[copy_idx] == kGlobalMulticastMode) {
          dst_core->local_mem()->write(multicast_buffer.data(),
                                       src_ctx.src_offset[copy_idx],
                                       src_ctx.copy_size[copy_idx]);
        } else {
          dst_core->local_mem()->copy_from(*src_core->local_mem(),
                                           src_ctx.src_offset[copy_idx],
                                           src_ctx.src_offset[copy_idx],
                                           src_ctx.copy_size[copy_idx]);
        }
      }
    }
  }

  for (const auto& copy_desc : team.pending_copies) {
    std::vector<uint8_t> multicast_buffer;
    if (copy_desc.copy_mode == kGlobalMulticastMode) {
      multicast_buffer = fetch_global_tile(copy_desc);
    } else if (is_panel_mode(copy_desc.copy_mode)) {
      multicast_buffer = fetch_global_tile(copy_desc);
      write_team_panel(team, copy_desc.src_offset, multicast_buffer);
      continue;
    }

    for (uint32_t dst_rank = 0; dst_rank < team.team_size; ++dst_rank) {
      if ((copy_desc.dst_mask & (1u << dst_rank)) == 0)
        continue;
      auto dst_core_id = team.rank_to_core.at(dst_rank);
      assert(dst_core_id != 0xffffffff);
      auto* dst_core = this->get_core(dst_core_id);
      if (copy_desc.copy_mode == kGlobalMulticastMode) {
        dst_core->local_mem()->write(multicast_buffer.data(),
                                     copy_desc.src_offset,
                                     copy_desc.copy_size);
      } else {
        auto* src_core = this->get_core(copy_desc.src_core_id);
        dst_core->local_mem()->copy_from(*src_core->local_mem(),
                                         copy_desc.src_offset,
                                         copy_desc.src_offset,
                                         copy_desc.copy_size);
      }
    }
  }

  team.pending_copies.clear();
}

void Cluster::resume_waiters(TeamState& team) {
  auto cores_per_socket = cores_per_socket_;
  for (auto local_id : team.participants) {
    if (!team.waiters.test(local_id))
      continue;
    if (team.ready_epoch < team.wait_epoch[local_id])
      continue;
    uint32_t socket_id = local_id / cores_per_socket;
    uint32_t socket_core_id = local_id % cores_per_socket;
    sockets_.at(socket_id)->resume(socket_core_id);
    team.waiters.reset(local_id);
  }
}

uint32_t Cluster::estimate_transfer_cycles(const TeamState& team) const {
  uint32_t source_bytes = 0;
  uint32_t fanout_bytes = 0;
  for (const auto& copy_desc : team.pending_copies) {
    if (copy_desc.copy_mode == kTeamPanelOracleMode) {
      continue;
    }
    source_bytes += copy_desc.copy_size;
    if (copy_desc.copy_mode == kTeamPanelMode) {
      fanout_bytes += copy_desc.copy_size;
    } else {
      fanout_bytes += copy_desc.copy_size * std::max<uint32_t>(1, __builtin_popcount(copy_desc.dst_mask));
    }
  }
  uint32_t cycles = (source_bytes + 15) / 16 + (fanout_bytes + 31) / 32;
  return std::max<uint32_t>(1, cycles);
}
