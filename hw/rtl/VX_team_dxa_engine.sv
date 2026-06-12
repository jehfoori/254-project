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

    VX_gbar_bus_if.slave  core_gbar_bus_if [NUM_REQS],
    VX_gbar_bus_if.master gbar_bus_if [NUM_REQS]
);
    `UNUSED_SPARAM (INSTANCE_ID)

    localparam [31:0] VX_TEAM_DXA_START_ID      = 32'hc0000003;
    localparam [31:0] VX_TEAM_DXA_WAIT_SLOT0_ID = 32'hc0000004;
    localparam [31:0] VX_TEAM_DXA_WAIT_SLOT1_ID = 32'hc0000005;
    localparam [31:0] VX_TEAM_COPY_MODE_DXA_STREAM = 32'd4;
    localparam DXA_STAMP_BYTES = 4;
    localparam DXA_LMEM_ADDR_WIDTH = `MEM_ADDR_WIDTH - $clog2(LSU_WORD_SIZE);

    `STATIC_ASSERT(LSU_WORD_SIZE >= DXA_STAMP_BYTES, ("DXA synthetic stamp requires at least 32-bit local-memory words"))

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
    reg [1:0] slot_done;
    reg [NUM_REQS-1:0] dxa_active_mask;
    team_csr_state_t [NUM_REQS-1:0] latched_team_csr_state;

    reg dxa_rsp_valid;
    reg [NB_WIDTH-1:0] dxa_rsp_id;

    reg write_active;
    reg write_slot;
    reg [`CLOG2(NUM_REQS)-1:0] write_core;

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

    wire current_core_active = dxa_active_mask[write_core];
    wire current_desc_valid = current_core_active
                           && (latched_team_csr_state[write_core].copy[write_slot].copy_mode == VX_TEAM_COPY_MODE_DXA_STREAM)
                           && (latched_team_csr_state[write_core].copy[write_slot].copy_size != 0);
    wire [DXA_LMEM_ADDR_WIDTH-1:0] current_lmem_word_addr =
        DXA_LMEM_ADDR_WIDTH'(64'(latched_team_csr_state[write_core].copy[write_slot].src_offset) >> $clog2(LSU_WORD_SIZE));
    wire [31:0] current_stamp = 32'hd0a00000
                              | ({31'b0, write_slot} << 16)
                              | 32'(write_core);
    wire [NUM_REQS-1:0] dxa_lmem_req_ready;
    wire current_write_ready = dxa_lmem_req_ready[write_core];
    wire current_write_fire = write_active && current_desc_valid && current_write_ready;
    wire current_write_skip = write_active && !current_desc_valid;
    wire current_write_step = current_write_fire || current_write_skip;

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

    for (genvar i = 0; i < NUM_REQS; ++i) begin : g_dxa_lmem_bus_if
        assign dxa_lmem_bus_if[i].req_valid = write_active && current_desc_valid && (`CLOG2(NUM_REQS)'(i) == write_core);
        assign dxa_lmem_bus_if[i].req_data.rw = 1'b1;
        assign dxa_lmem_bus_if[i].req_data.addr = current_lmem_word_addr;
        assign dxa_lmem_bus_if[i].req_data.data = (LSU_WORD_SIZE * 8)'(current_stamp);
        assign dxa_lmem_bus_if[i].req_data.byteen = LSU_WORD_SIZE'({{(LSU_WORD_SIZE-DXA_STAMP_BYTES){1'b0}}, {DXA_STAMP_BYTES{1'b1}}});
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
            slot_done <= '0;
            dxa_active_mask <= '0;
            dxa_rsp_valid <= 0;
            dxa_rsp_id <= '0;
            write_active <= 0;
            write_slot <= 0;
            write_core <= '0;
            for (integer i = 0; i < NUM_REQS; ++i) begin
                latched_team_csr_state[i] <= '0;
            end
        end else begin
            if (dxa_rsp_fire) begin
                dxa_rsp_valid <= 0;
                pending_dxa_rsp <= '0;
            end

            if (current_write_step) begin
                if (write_core == `CLOG2(NUM_REQS)'(NUM_REQS-1)) begin
                    if (write_slot == 0) begin
                        slot_done[0] <= 1;
                        write_slot <= 1;
                        write_core <= '0;
                    end else begin
                        slot_done[1] <= 1;
                        write_active <= 0;
                        write_slot <= 0;
                        write_core <= '0;
                    end
                end else begin
                    write_core <= write_core + `CLOG2(NUM_REQS)'(1);
                end
            end

            if (!dxa_rsp_valid && (| wait_mask[0]) && slot_done[0] && (count_mask(wait_mask[0]) == integer'(wait_count_goal[0]))) begin
                wait_mask[0] <= '0;
                dxa_rsp_valid <= 1;
                dxa_rsp_id <= wait_rsp_id[0];
            end else if (!dxa_rsp_valid && (| wait_mask[1]) && slot_done[1] && (count_mask(wait_mask[1]) == integer'(wait_count_goal[1]))) begin
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
                        for (integer i = 0; i < NUM_REQS; ++i) begin
                            latched_team_csr_state[i] <= team_csr_state[i];
                        end
                        slot_done <= 2'b00;
                        dxa_active_mask <= start_mask | (NUM_REQS'(1) << dxa_req_core_id);
                        write_active <= 1;
                        write_slot <= 0;
                        write_core <= '0;
                        dxa_rsp_valid <= 1;
                        dxa_rsp_id <= dxa_req_id;
                    end else begin
                        start_mask[dxa_req_core_id] <= 1;
                    end
                end else if (dxa_req_raw_id == VX_TEAM_DXA_WAIT_SLOT0_ID) begin
                    if (slot_done[0] && (count_mask(wait_mask[0]) == integer'(dxa_req_size_m1))) begin
                        wait_mask[0] <= '0;
                        dxa_rsp_valid <= 1;
                        dxa_rsp_id <= dxa_req_id;
                    end else begin
                        wait_mask[0][dxa_req_core_id] <= 1;
                        wait_count_goal[0] <= NC_WIDTH'(dxa_req_size_m1) + NC_WIDTH'(1);
                        wait_rsp_id[0] <= dxa_req_id;
                    end
                end else begin
                    if (slot_done[1] && (count_mask(wait_mask[1]) == integer'(dxa_req_size_m1))) begin
                        wait_mask[1] <= '0;
                        dxa_rsp_valid <= 1;
                        dxa_rsp_id <= dxa_req_id;
                    end else begin
                        wait_mask[1][dxa_req_core_id] <= 1;
                        wait_count_goal[1] <= NC_WIDTH'(dxa_req_size_m1) + NC_WIDTH'(1);
                        wait_rsp_id[1] <= dxa_req_id;
                    end
                end
            end
        end
    end

    `UNUSED_VAR (latched_team_csr_state)

endmodule
