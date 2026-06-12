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

    VX_gbar_bus_if.slave  core_gbar_bus_if [NUM_REQS],
    VX_gbar_bus_if.master gbar_bus_if [NUM_REQS]
);
    `UNUSED_SPARAM (INSTANCE_ID)

    localparam [31:0] VX_TEAM_DXA_START_ID      = 32'hc0000003;
    localparam [31:0] VX_TEAM_DXA_WAIT_SLOT0_ID = 32'hc0000004;
    localparam [31:0] VX_TEAM_DXA_WAIT_SLOT1_ID = 32'hc0000005;

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
    reg [NUM_REQS-1:0] pending_dxa_rsp;
    reg [1:0] slot_done;
    team_csr_state_t [NUM_REQS-1:0] latched_team_csr_state;

    reg dxa_rsp_valid;
    reg [NB_WIDTH-1:0] dxa_rsp_id;

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

    always @(posedge clk) begin
        if (reset) begin
            start_mask <= '0;
            wait_mask[0] <= '0;
            wait_mask[1] <= '0;
            pending_dxa_rsp <= '0;
            slot_done <= '0;
            dxa_rsp_valid <= 0;
            dxa_rsp_id <= '0;
            for (integer i = 0; i < NUM_REQS; ++i) begin
                latched_team_csr_state[i] <= '0;
            end
        end else begin
            if (dxa_rsp_fire) begin
                dxa_rsp_valid <= 0;
                pending_dxa_rsp <= '0;
            end

            if (dxa_req_fire) begin
                pending_dxa_rsp[dxa_req_index] <= 1;
                if (dxa_req_raw_id == VX_TEAM_DXA_START_ID) begin
                    if (count_mask(start_mask) == integer'(dxa_req_size_m1)) begin
                        start_mask <= '0;
                        wait_mask[0] <= '0;
                        wait_mask[1] <= '0;
                        for (integer i = 0; i < NUM_REQS; ++i) begin
                            latched_team_csr_state[i] <= team_csr_state[i];
                        end
                        slot_done <= 2'b11;
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
                    end
                end else begin
                    if (slot_done[1] && (count_mask(wait_mask[1]) == integer'(dxa_req_size_m1))) begin
                        wait_mask[1] <= '0;
                        dxa_rsp_valid <= 1;
                        dxa_rsp_id <= dxa_req_id;
                    end else begin
                        wait_mask[1][dxa_req_core_id] <= 1;
                    end
                end
            end
        end
    end

    `UNUSED_VAR (latched_team_csr_state)

endmodule
