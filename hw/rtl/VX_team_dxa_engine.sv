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

`include "VX_define.vh"

module VX_team_dxa_engine import VX_gpu_pkg::*; #(
    parameter NUM_REQS = 1,
    parameter `STRING INSTANCE_ID = ""
) (
    input wire clk,
    input wire reset,

    input team_csr_state_t [NUM_REQS-1:0] team_csr_state,

    VX_mem_bus_if.master dxa_lmem_bus_if [NUM_REQS],
    VX_mem_bus_if.master dxa_dcache_bus_if,

    VX_gbar_bus_if.slave  core_gbar_bus_if [NUM_REQS],
    VX_gbar_bus_if.master gbar_bus_if [NUM_REQS]
);
    `UNUSED_SPARAM (INSTANCE_ID)

    localparam [31:0] VX_TEAM_DXA_START_ID      = 32'hc0000003;
    localparam [31:0] VX_TEAM_DXA_WAIT_SLOT0_ID = 32'hc0000004;
    localparam [31:0] VX_TEAM_DXA_WAIT_SLOT1_ID = 32'hc0000005;
    localparam [31:0] VX_TEAM_COPY_MODE_DXA_STREAM = 32'd4;
    localparam DXA_PANEL_OFFSET = 32'h2000;
    localparam DXA_MAX_PANELS = 4;
    localparam DXA_LMEM_ADDR_WIDTH = `MEM_ADDR_WIDTH - $clog2(LSU_WORD_SIZE);
    localparam DXA_DCACHE_ADDR_WIDTH = `MEM_ADDR_WIDTH - $clog2(DCACHE_WORD_SIZE);

    localparam DXA_STATE_IDLE      = 3'd0;
    localparam DXA_STATE_WAIT_SLOT = 3'd1;
    localparam DXA_STATE_READ_REQ  = 3'd2;
    localparam DXA_STATE_READ_RSP  = 3'd3;
    localparam DXA_STATE_WRITE     = 3'd4;

    `STATIC_ASSERT(DCACHE_WORD_SIZE >= LSU_WORD_SIZE, ("DXA v1 requires D-cache words to cover local-memory beats"))
    `STATIC_ASSERT((DCACHE_WORD_SIZE % LSU_WORD_SIZE) == 0, ("DXA v1 requires integral local-memory beats per D-cache word"))

    function automatic logic is_dxa_barrier(input logic [31:0] raw_id);
        begin
            is_dxa_barrier = (raw_id == VX_TEAM_DXA_START_ID)
                          || (raw_id == VX_TEAM_DXA_WAIT_SLOT0_ID)
                          || (raw_id == VX_TEAM_DXA_WAIT_SLOT1_ID);
        end
    endfunction

    function automatic integer count_mask(input logic [`NUM_CORES-1:0] mask);
        integer i;
        begin
            count_mask = 0;
            for (i = 0; i < `NUM_CORES; ++i) begin
                count_mask = count_mask + integer'(mask[i]);
            end
        end
    endfunction

    reg [`NUM_CORES-1:0] start_mask;
    reg [`NUM_CORES-1:0] wait_mask [2];
    reg [NC_WIDTH:0] wait_count_goal [2];
    reg [NB_WIDTH-1:0] wait_rsp_id [2];
    reg [NUM_REQS-1:0] pending_dxa_rsp;
    reg [1:0] slot_valid;
    reg [31:0] slot_panel [2];
    reg [31:0] next_wait_panel [NUM_REQS];
    reg [NUM_REQS-1:0] dxa_active_mask;
    team_copy_desc_t [1:0] latched_leader_copy;
    reg [15:0] latched_rank_x [NUM_REQS];
    reg [15:0] latched_rank_y [NUM_REQS];
    reg [15:0] latched_team_size_x;
    reg [15:0] latched_team_size_y;
    reg [31:0] latched_panel_count;
    reg [31:0] panel_slot_bytes;

    reg dxa_rsp_valid;
    reg [NB_WIDTH-1:0] dxa_rsp_id;

    reg [2:0] dxa_state;
    reg current_copy;
    reg [31:0] current_panel;
    reg [31:0] current_group;
    reg [31:0] current_row;
    reg [31:0] current_byte;
    reg [`CLOG2(NUM_REQS)-1:0] write_core;
    reg [(LSU_WORD_SIZE * 8)-1:0] read_word;

    wire [NUM_REQS-1:0] core_req_valid;
    wire [NUM_REQS-1:0] core_req_is_dxa;
    wire [NUM_REQS-1:0][NB_WIDTH-1:0] core_req_id;
    wire [NUM_REQS-1:0][31:0] core_req_raw_id;
    wire [NUM_REQS-1:0][NC_WIDTH-1:0] core_req_size_m1;
    wire [NUM_REQS-1:0][NC_WIDTH-1:0] core_req_core_id;

    wire [NUM_REQS-1:0] normal_rsp_valid;
    for (genvar i = 0; i < NUM_REQS; ++i) begin : g_core_req_data
        assign core_req_valid[i] = core_gbar_bus_if[i].req_valid;
        assign core_req_id[i] = core_gbar_bus_if[i].req_data.id;
        assign core_req_raw_id[i] = core_gbar_bus_if[i].req_data.raw_id;
        assign core_req_size_m1[i] = core_gbar_bus_if[i].req_data.size_m1;
        assign core_req_core_id[i] = core_gbar_bus_if[i].req_data.core_id;
        assign core_req_is_dxa[i] = core_req_valid[i] && is_dxa_barrier(core_req_raw_id[i]);
        assign normal_rsp_valid[i] = gbar_bus_if[i].rsp_valid;
    end
    wire normal_rsp_any = (| normal_rsp_valid);
    wire dxa_rsp_fire = dxa_rsp_valid && !normal_rsp_any;

    reg dxa_req_valid;
    reg [`CLOG2(NUM_REQS)-1:0] dxa_req_index;
    reg [NB_WIDTH-1:0] dxa_req_id;
    reg [31:0] dxa_req_raw_id;
    reg [NC_WIDTH-1:0] dxa_req_size_m1;
    reg [NC_WIDTH-1:0] dxa_req_core_id;

    always @(*) begin
        dxa_req_valid = 0;
        dxa_req_index = '0;
        dxa_req_id = '0;
        dxa_req_raw_id = '0;
        dxa_req_size_m1 = '0;
        dxa_req_core_id = '0;
        for (integer i = 0; i < NUM_REQS; ++i) begin
            if (!dxa_req_valid && core_req_is_dxa[i]) begin
                dxa_req_valid = 1;
                dxa_req_index = `CLOG2(NUM_REQS)'(i);
                dxa_req_id = core_req_id[i];
                dxa_req_raw_id = core_req_raw_id[i];
                dxa_req_size_m1 = core_req_size_m1[i];
                dxa_req_core_id = core_req_core_id[i];
            end
        end
    end

    wire dxa_req_ready = !dxa_rsp_valid;
    wire dxa_req_fire = dxa_req_valid && dxa_req_ready;

    team_copy_desc_t [1:0] leader_copy;
    reg [15:0] leader_team_size_x;
    reg [15:0] leader_team_size_y;
    always @(*) begin
        leader_copy[0] = '0;
        leader_copy[1] = '0;
        leader_team_size_x = '0;
        leader_team_size_y = '0;
        for (integer i = 0; i < NUM_REQS; ++i) begin
            if (team_csr_state[i].team_rank_x == 0 && team_csr_state[i].team_rank_y == 0) begin
                leader_copy[0] = team_csr_state[i].copy[0];
                leader_copy[1] = team_csr_state[i].copy[1];
                leader_team_size_x = team_csr_state[i].team_size_x;
                leader_team_size_y = team_csr_state[i].team_size_y;
            end
        end
    end

    wire [`NUM_CORES-1:0] dxa_req_core_mask = (`NUM_CORES'(1) << dxa_req_core_id);
    wire [`NUM_CORES-1:0] wait0_arrivals = wait_mask[0] | dxa_req_core_mask;
    wire [`NUM_CORES-1:0] wait1_arrivals = wait_mask[1] | dxa_req_core_mask;
    wire [31:0] leader_panel_count = (leader_copy[0].dst_mask != 0) ? leader_copy[0].dst_mask : leader_copy[1].dst_mask;

    function automatic logic slot_can_overwrite(input logic slot);
        begin
            slot_can_overwrite = 1;
            if (slot_valid[slot]) begin
                for (integer i = 0; i < NUM_REQS; ++i) begin
                    if (dxa_active_mask[i] && (next_wait_panel[i] < (slot_panel[slot] + 32'd2))) begin
                        slot_can_overwrite = 0;
                    end
                end
            end
        end
    endfunction

    function automatic logic wait_arrivals_ready(input logic slot, input logic [`NUM_CORES-1:0] arrivals);
        begin
            wait_arrivals_ready = slot_valid[slot];
            for (integer i = 0; i < NUM_REQS; ++i) begin
                if (arrivals[i] && (next_wait_panel[i] != slot_panel[slot])) begin
                    wait_arrivals_ready = 0;
                end
            end
        end
    endfunction

    wire current_panel_slot = current_panel[0];
    wire current_desc_nonzero = latched_leader_copy[current_copy].copy_size != 0;
    wire current_desc_stream = latched_leader_copy[current_copy].copy_mode == VX_TEAM_COPY_MODE_DXA_STREAM;
    wire [31:0] current_tile_rows = latched_leader_copy[current_copy].tile_rows;
    wire [31:0] current_global_stride = latched_leader_copy[current_copy].global_stride;
    wire [31:0] current_copy_size = latched_leader_copy[current_copy].copy_size;
    wire [31:0] current_row_bytes = (current_tile_rows != 0) ? (current_copy_size / current_tile_rows) : 32'd0;
    wire [31:0] current_group_count = current_copy ? 32'(latched_team_size_x) : 32'(latched_team_size_y);
    wire current_desc_supported = !current_desc_nonzero
                               || (current_desc_stream
                                && (latched_panel_count >= 1)
                                && (latched_panel_count <= DXA_MAX_PANELS)
                                && (current_tile_rows != 0)
                                && (current_global_stride != 0)
                                && ((current_copy_size % current_tile_rows) == 0)
                                && ((current_row_bytes % LSU_WORD_SIZE) == 0)
                                && (current_group_count != 0)
                                && (current_group < current_group_count)
                                && (latched_leader_copy[current_copy].src_offset[$clog2(LSU_WORD_SIZE)-1:0] == '0));

    wire [63:0] current_group_global_offset =
        current_copy ? (64'(current_panel) * 64'(current_tile_rows) * 64'(current_global_stride)
                      + 64'(current_group) * 64'(current_row_bytes))
                     : (64'(current_panel) * 64'(current_row_bytes)
                      + 64'(current_group) * 64'(current_tile_rows) * 64'(current_global_stride));
    wire [63:0] current_global_byte_addr = 64'(latched_leader_copy[current_copy].global_addr)
                                         + current_group_global_offset
                                         + 64'(current_row) * 64'(current_global_stride)
                                         + 64'(current_byte);
    wire [63:0] current_lmem_byte_offset = 64'(DXA_PANEL_OFFSET)
                                         + 64'(latched_leader_copy[current_copy].src_offset)
                                         + (current_panel_slot ? 64'(panel_slot_bytes) : 64'd0)
                                         + 64'(current_group) * 64'(current_copy_size)
                                         + 64'(current_row) * 64'(current_row_bytes)
                                         + 64'(current_byte);

    wire current_core_active = dxa_active_mask[write_core];
    wire current_core_match = current_core_active
                           && (current_copy ? (32'(latched_rank_x[write_core]) == current_group)
                                            : (32'(latched_rank_y[write_core]) == current_group));
    wire [DXA_LMEM_ADDR_WIDTH-1:0] current_lmem_word_addr =
        DXA_LMEM_ADDR_WIDTH'(current_lmem_byte_offset >> $clog2(LSU_WORD_SIZE));
    wire [DXA_DCACHE_ADDR_WIDTH-1:0] current_dcache_word_addr =
        DXA_DCACHE_ADDR_WIDTH'(current_global_byte_addr >> $clog2(DCACHE_WORD_SIZE));
    wire [$clog2(DCACHE_WORD_SIZE)-1:0] current_dcache_byte_offset =
        current_global_byte_addr[$clog2(DCACHE_WORD_SIZE)-1:0];
    wire [$clog2(DCACHE_WORD_SIZE * 8)-1:0] current_dcache_bit_offset =
        ($clog2(DCACHE_WORD_SIZE * 8))'(current_dcache_byte_offset) << 3;
    wire current_global_lsu_aligned = current_global_byte_addr[$clog2(LSU_WORD_SIZE)-1:0] == '0;
    wire [NUM_REQS-1:0] dxa_lmem_req_ready;
    wire current_write_ready = dxa_lmem_req_ready[write_core];
    wire current_write_fire = (dxa_state == DXA_STATE_WRITE) && current_core_match && current_write_ready;
    wire current_write_skip = (dxa_state == DXA_STATE_WRITE) && !current_core_match;
    wire current_write_step = current_write_fire || current_write_skip;
    wire dxa_dcache_req_fire = dxa_dcache_bus_if.req_valid && dxa_dcache_bus_if.req_ready;
    wire dxa_dcache_rsp_fire = dxa_dcache_bus_if.rsp_valid && dxa_dcache_bus_if.rsp_ready;

    `RUNTIME_ASSERT(~((dxa_state == DXA_STATE_READ_REQ) && current_desc_nonzero && !current_desc_supported),
        ("%t: *** %s unsupported DXA descriptor: copy=%0d size=%0d rows=%0d stride=%0d panels=%0d",
         $time, INSTANCE_ID, current_copy, current_copy_size, current_tile_rows, current_global_stride, latched_panel_count))
    `RUNTIME_ASSERT(~((dxa_state == DXA_STATE_READ_REQ) && current_desc_nonzero && !current_global_lsu_aligned),
        ("%t: *** %s unaligned DXA global address: 0x%0h", $time, INSTANCE_ID, current_global_byte_addr))

    for (genvar i = 0; i < NUM_REQS; ++i) begin : g_gbar_filter
        wire is_selected_dxa_req = dxa_req_valid && (`CLOG2(NUM_REQS)'(i) == dxa_req_index);

        assign gbar_bus_if[i].req_valid = core_gbar_bus_if[i].req_valid && !core_req_is_dxa[i];
        assign gbar_bus_if[i].req_data = core_gbar_bus_if[i].req_data;
        assign core_gbar_bus_if[i].req_ready = core_req_is_dxa[i] ? (is_selected_dxa_req && dxa_req_ready)
                                                                  : gbar_bus_if[i].req_ready;

        assign core_gbar_bus_if[i].rsp_valid = (dxa_rsp_fire && pending_dxa_rsp[i]) ? 1'b1
                                             : ((pending_dxa_rsp[i] || core_req_is_dxa[i]) ? 1'b0
                                                                                           : gbar_bus_if[i].rsp_valid);
        assign core_gbar_bus_if[i].rsp_data.id = dxa_rsp_fire ? dxa_rsp_id
                                           : gbar_bus_if[i].rsp_data.id;
    end

    assign dxa_dcache_bus_if.req_valid = (dxa_state == DXA_STATE_READ_REQ) && current_desc_nonzero && current_desc_supported;
    assign dxa_dcache_bus_if.req_data.rw = 1'b0;
    assign dxa_dcache_bus_if.req_data.addr = current_dcache_word_addr;
    assign dxa_dcache_bus_if.req_data.data = '0;
    assign dxa_dcache_bus_if.req_data.byteen = '1;
    assign dxa_dcache_bus_if.req_data.flags = '0;
    assign dxa_dcache_bus_if.req_data.tag = '0;
    assign dxa_dcache_bus_if.rsp_ready = (dxa_state == DXA_STATE_READ_RSP);

    for (genvar i = 0; i < NUM_REQS; ++i) begin : g_dxa_lmem_bus_if
        assign dxa_lmem_bus_if[i].req_valid = (dxa_state == DXA_STATE_WRITE) && current_core_match && (`CLOG2(NUM_REQS)'(i) == write_core);
        assign dxa_lmem_bus_if[i].req_data.rw = 1'b1;
        assign dxa_lmem_bus_if[i].req_data.addr = current_lmem_word_addr;
        assign dxa_lmem_bus_if[i].req_data.data = read_word;
        assign dxa_lmem_bus_if[i].req_data.byteen = '1;
        assign dxa_lmem_bus_if[i].req_data.flags = '0;
        assign dxa_lmem_bus_if[i].req_data.tag = '0;
        assign dxa_lmem_bus_if[i].rsp_ready = 1'b1;
        assign dxa_lmem_req_ready[i] = dxa_lmem_bus_if[i].req_ready;
        `UNUSED_VAR (dxa_lmem_bus_if[i].rsp_valid)
        `UNUSED_VAR (dxa_lmem_bus_if[i].rsp_data)
    end

    always @(posedge clk) begin
        if (reset) begin
            start_mask <= '0;
            wait_mask[0] <= '0;
            wait_mask[1] <= '0;
            wait_count_goal[0] <= '0;
            wait_count_goal[1] <= '0;
            wait_rsp_id[0] <= '0;
            wait_rsp_id[1] <= '0;
            pending_dxa_rsp <= '0;
            slot_valid <= '0;
            slot_panel[0] <= '0;
            slot_panel[1] <= '0;
            dxa_active_mask <= '0;
            dxa_rsp_valid <= 0;
            dxa_rsp_id <= '0;
            dxa_state <= DXA_STATE_IDLE;
            current_copy <= 0;
            current_panel <= '0;
            current_group <= '0;
            current_row <= '0;
            current_byte <= '0;
            write_core <= '0;
            read_word <= '0;
            latched_leader_copy[0] <= '0;
            latched_leader_copy[1] <= '0;
            latched_team_size_x <= '0;
            latched_team_size_y <= '0;
            latched_panel_count <= '0;
            panel_slot_bytes <= '0;
            for (integer i = 0; i < NUM_REQS; ++i) begin
                next_wait_panel[i] <= '0;
                latched_rank_x[i] <= '0;
                latched_rank_y[i] <= '0;
            end
        end else begin
            if (dxa_rsp_fire) begin
                dxa_rsp_valid <= 0;
                pending_dxa_rsp <= '0;
            end

            if (dxa_state == DXA_STATE_WAIT_SLOT) begin
                if ((current_panel < latched_panel_count) && slot_can_overwrite(current_panel_slot)) begin
                    current_copy <= 0;
                    current_group <= '0;
                    current_row <= '0;
                    current_byte <= '0;
                    write_core <= '0;
                    dxa_state <= DXA_STATE_READ_REQ;
                end else if (current_panel >= latched_panel_count) begin
                    dxa_state <= DXA_STATE_IDLE;
                end
            end else if (dxa_state == DXA_STATE_READ_REQ) begin
                if (!current_desc_nonzero) begin
                    if (!current_copy) begin
                        current_copy <= 1;
                    end else begin
                        slot_valid[current_panel_slot] <= 1;
                        slot_panel[current_panel_slot] <= current_panel;
                        if ((current_panel + 32'd1) < latched_panel_count) begin
                            current_panel <= current_panel + 32'd1;
                            dxa_state <= DXA_STATE_WAIT_SLOT;
                        end else begin
                            current_panel <= latched_panel_count;
                            dxa_state <= DXA_STATE_IDLE;
                        end
                    end
                end else if (dxa_dcache_req_fire) begin
                    dxa_state <= DXA_STATE_READ_RSP;
                end
            end else if (dxa_state == DXA_STATE_READ_RSP) begin
                if (dxa_dcache_rsp_fire) begin
                    read_word <= dxa_dcache_bus_if.rsp_data.data[current_dcache_bit_offset +: (LSU_WORD_SIZE * 8)];
                    dxa_state <= DXA_STATE_WRITE;
                    write_core <= '0;
                end
            end else if (current_write_step) begin
                if (write_core == `CLOG2(NUM_REQS)'(NUM_REQS-1)) begin
                    if ((current_byte + LSU_WORD_SIZE) < current_row_bytes) begin
                        current_byte <= current_byte + LSU_WORD_SIZE;
                        write_core <= '0;
                        dxa_state <= DXA_STATE_READ_REQ;
                    end else if ((current_row + 32'd1) < current_tile_rows) begin
                        current_byte <= '0;
                        current_row <= current_row + 32'd1;
                        write_core <= '0;
                        dxa_state <= DXA_STATE_READ_REQ;
                    end else if ((current_group + 32'd1) < current_group_count) begin
                        current_byte <= '0;
                        current_row <= '0;
                        current_group <= current_group + 32'd1;
                        write_core <= '0;
                        dxa_state <= DXA_STATE_READ_REQ;
                    end else if (!current_copy) begin
                        current_byte <= '0;
                        current_row <= '0;
                        current_group <= '0;
                        current_copy <= 1;
                        write_core <= '0;
                        dxa_state <= DXA_STATE_READ_REQ;
                    end else begin
                        slot_valid[current_panel_slot] <= 1;
                        slot_panel[current_panel_slot] <= current_panel;
                        current_byte <= '0;
                        current_row <= '0;
                        current_group <= '0;
                        current_copy <= 0;
                        write_core <= '0;
                        if ((current_panel + 32'd1) < latched_panel_count) begin
                            current_panel <= current_panel + 32'd1;
                            dxa_state <= DXA_STATE_WAIT_SLOT;
                        end else begin
                            current_panel <= latched_panel_count;
                            dxa_state <= DXA_STATE_IDLE;
                        end
                    end
                end else begin
                    write_core <= write_core + `CLOG2(NUM_REQS)'(1);
                end
            end

            if (!dxa_rsp_valid && (| wait_mask[0]) && wait_arrivals_ready(1'b0, wait_mask[0])
             && (count_mask(wait_mask[0]) == integer'(wait_count_goal[0]))) begin
                for (integer i = 0; i < NUM_REQS; ++i) begin
                    if (wait_mask[0][i]) begin
                        next_wait_panel[i] <= next_wait_panel[i] + 32'd1;
                    end
                end
                wait_mask[0] <= '0;
                dxa_rsp_valid <= 1;
                dxa_rsp_id <= wait_rsp_id[0];
            end else if (!dxa_rsp_valid && (| wait_mask[1]) && wait_arrivals_ready(1'b1, wait_mask[1])
             && (count_mask(wait_mask[1]) == integer'(wait_count_goal[1]))) begin
                for (integer i = 0; i < NUM_REQS; ++i) begin
                    if (wait_mask[1][i]) begin
                        next_wait_panel[i] <= next_wait_panel[i] + 32'd1;
                    end
                end
                wait_mask[1] <= '0;
                dxa_rsp_valid <= 1;
                dxa_rsp_id <= wait_rsp_id[1];
            end

            if (dxa_req_fire) begin
                pending_dxa_rsp[dxa_req_index] <= 1;
                if (dxa_req_raw_id == VX_TEAM_DXA_START_ID) begin
                    if (count_mask(start_mask) == integer'(dxa_req_size_m1)) begin
                        start_mask <= '0;
                        wait_mask[0] <= '0;
                        wait_mask[1] <= '0;
                        wait_count_goal[0] <= '0;
                        wait_count_goal[1] <= '0;
                        latched_leader_copy[0] <= leader_copy[0];
                        latched_leader_copy[1] <= leader_copy[1];
                        latched_team_size_x <= leader_team_size_x;
                        latched_team_size_y <= leader_team_size_y;
                        latched_panel_count <= leader_panel_count;
                        panel_slot_bytes <= 32'(leader_team_size_y) * leader_copy[0].copy_size
                                          + 32'(leader_team_size_x) * leader_copy[1].copy_size;
                        slot_valid <= 2'b00;
                        slot_panel[0] <= '0;
                        slot_panel[1] <= '0;
                        dxa_active_mask <= start_mask | (NUM_REQS'(1) << dxa_req_core_id);
                        dxa_state <= DXA_STATE_WAIT_SLOT;
                        current_panel <= '0;
                        current_copy <= 0;
                        current_group <= '0;
                        current_row <= '0;
                        current_byte <= '0;
                        write_core <= '0;
                        read_word <= '0;
                        for (integer i = 0; i < NUM_REQS; ++i) begin
                            next_wait_panel[i] <= '0;
                            latched_rank_x[i] <= team_csr_state[i].team_rank_x;
                            latched_rank_y[i] <= team_csr_state[i].team_rank_y;
                        end
                        dxa_rsp_valid <= 1;
                        dxa_rsp_id <= dxa_req_id;
                    end else begin
                        start_mask[dxa_req_core_id] <= 1;
                    end
                end else if (dxa_req_raw_id == VX_TEAM_DXA_WAIT_SLOT0_ID) begin
                    if (wait_arrivals_ready(1'b0, wait0_arrivals)
                     && (count_mask(wait0_arrivals) == integer'(dxa_req_size_m1) + 1)) begin
                        for (integer i = 0; i < NUM_REQS; ++i) begin
                            if (wait0_arrivals[i]) begin
                                next_wait_panel[i] <= next_wait_panel[i] + 32'd1;
                            end
                        end
                        wait_mask[0] <= '0;
                        dxa_rsp_valid <= 1;
                        dxa_rsp_id <= dxa_req_id;
                    end else begin
                        wait_mask[0] <= wait0_arrivals;
                        wait_count_goal[0] <= NC_WIDTH'(dxa_req_size_m1) + NC_WIDTH'(1);
                        wait_rsp_id[0] <= dxa_req_id;
                    end
                end else begin
                    if (wait_arrivals_ready(1'b1, wait1_arrivals)
                     && (count_mask(wait1_arrivals) == integer'(dxa_req_size_m1) + 1)) begin
                        for (integer i = 0; i < NUM_REQS; ++i) begin
                            if (wait1_arrivals[i]) begin
                                next_wait_panel[i] <= next_wait_panel[i] + 32'd1;
                            end
                        end
                        wait_mask[1] <= '0;
                        dxa_rsp_valid <= 1;
                        dxa_rsp_id <= dxa_req_id;
                    end else begin
                        wait_mask[1] <= wait1_arrivals;
                        wait_count_goal[1] <= NC_WIDTH'(dxa_req_size_m1) + NC_WIDTH'(1);
                        wait_rsp_id[1] <= dxa_req_id;
                    end
                end
            end
        end
    end

endmodule
