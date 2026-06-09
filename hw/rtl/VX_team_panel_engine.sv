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

module VX_team_panel_engine #(
    parameter `STRING INSTANCE_ID = ""
) (
    input wire              clk,
    input wire              reset,

    input wire              run_valid,
    input wire [`XLEN-1:0]  a_offset,
    input wire [`XLEN-1:0]  b_offset,
    input wire [`XLEN-1:0]  c_addr,
    input wire [`XLEN-1:0]  c_stride,
    input wire [`XLEN-1:0]  a_stride,
    input wire [`XLEN-1:0]  b_stride,
    input wire [`XLEN-1:0]  k_tiles,
    input wire [`XLEN-1:0]  n_tiles,
    input wire [`XLEN-1:0]  a_seed,
    input wire [`XLEN-1:0]  b_seed,
    input wire [`XLEN-1:0]  a_word0,
    input wire [`XLEN-1:0]  a_word1,
    input wire [`XLEN-1:0]  b_word0,
    input wire [`XLEN-1:0]  b_word1,
    input wire [`XLEN-1:0]  status_sel,

    VX_mem_bus_if.master    mem_bus_if,

    output wire [`XLEN-1:0] status,
    output wire             busy,
    output wire             done
);
    `UNUSED_SPARAM (INSTANCE_ID)
    `UNUSED_VAR (a_word1)
    `UNUSED_VAR (b_word1)
    reg [`XLEN-1:0] a_offset_r;
    reg [`XLEN-1:0] b_offset_r;
    reg [`XLEN-1:0] c_addr_r;
    reg [`XLEN-1:0] c_stride_r;
    reg [`XLEN-1:0] a_stride_r;
    reg [`XLEN-1:0] b_stride_r;
    reg [`XLEN-1:0] a_seed_r;
    reg [`XLEN-1:0] b_seed_r;
    reg [`XLEN-1:0] a_panel_base_addr_r;
    reg [`XLEN-1:0] b_panel_base_addr_r;
    reg [`XLEN-1:0] k_tiles_r;
    reg [`XLEN-1:0] n_tiles_r;
    reg [`XLEN-1:0] compute_steps_total_r;
    reg [`XLEN-1:0] compute_steps_done_r;
    reg [`XLEN-1:0] writeback_count_r;
    reg [`XLEN-1:0] writeback_signature_r;
    reg [`XLEN-1:0] accum_r;
    reg [`XLEN-1:0] k_idx_r;
    reg [`XLEN-1:0] n_idx_r;
    reg [`XLEN-1:0] write_addr_r;
    reg [`XLEN-1:0] write_data_r;
    reg [`XLEN-1:0] remaining_cycles_r;
    reg [3:0]       state_r;
    reg [`XLEN-1:0] fetch_index_r;
    reg             fetch_phase_b_r;
    reg             busy_r;
    reg             done_r;

    localparam integer TILE_M = 8;
    localparam integer TILE_N = 4;
    localparam integer TILE_K = 8;
    localparam integer PANEL_WORD_ELEMS = `XLEN / 16;
    localparam integer TILE_N_WORDS = TILE_N / PANEL_WORD_ELEMS;
    localparam integer MAX_A_PANEL_WORDS = 256;
    localparam integer MAX_B_PANEL_WORDS = 512;
    localparam integer MEM_BUS_DATA_SIZE = VX_gpu_pkg::DCACHE_WORD_SIZE;
    localparam integer A_PANEL_IDX_W = `CLOG2(MAX_A_PANEL_WORDS);
    localparam integer B_PANEL_IDX_W = `CLOG2(MAX_B_PANEL_WORDS);
    reg [`XLEN-1:0] a_panel_words_r [0:MAX_A_PANEL_WORDS-1];
    reg [`XLEN-1:0] b_panel_words_r [0:MAX_B_PANEL_WORDS-1];

    wire [`XLEN-1:0] run_k_tiles = (k_tiles != 0) ? k_tiles : `XLEN'(1);
    wire [`XLEN-1:0] run_n_tiles = (n_tiles != 0) ? n_tiles : `XLEN'(1);
    wire [`XLEN-1:0] run_cycles  = run_k_tiles * run_n_tiles;
    wire [`XLEN-1:0] a_words_per_row = a_stride_r >> `XLEN'(1);
    wire [`XLEN-1:0] b_words_per_row = b_stride_r >> `XLEN'(1);
    wire [`XLEN-1:0] a_fetch_words = (`XLEN'(TILE_M) * a_stride_r) >> `XLEN'(1);
    wire [`XLEN-1:0] b_fetch_words = (a_stride_r * b_stride_r) >> `XLEN'(1);
    wire [`XLEN-1:0] a_fetch_words_limited = (a_fetch_words < `XLEN'(MAX_A_PANEL_WORDS)) ? a_fetch_words : `XLEN'(MAX_A_PANEL_WORDS);
    wire [`XLEN-1:0] b_fetch_words_limited = (b_fetch_words < `XLEN'(MAX_B_PANEL_WORDS)) ? b_fetch_words : `XLEN'(MAX_B_PANEL_WORDS);

    wire [`XLEN-1:0] k_block_word = k_idx_r << `XLEN'(2);
    wire [`XLEN-1:0] n_block_word = n_idx_r * `XLEN'(TILE_N_WORDS);
    wire [`XLEN-1:0] a_idx0 = k_block_word;
    wire [`XLEN-1:0] a_idx1 = ((`XLEN'(TILE_M - 1) * a_words_per_row) + k_block_word + `XLEN'(3));
    wire [`XLEN-1:0] b_idx0 = ((k_idx_r * `XLEN'(TILE_K)) * b_words_per_row) + n_block_word;
    wire [`XLEN-1:0] b_idx1 = (((k_idx_r * `XLEN'(TILE_K) + `XLEN'(TILE_K - 1)) * b_words_per_row) + n_block_word + `XLEN'(1));

    function automatic [`XLEN-1:0] panel_read_a(input [`XLEN-1:0] index);
    begin
        if (index < `XLEN'(MAX_A_PANEL_WORDS)) begin
            panel_read_a = a_panel_words_r[index[A_PANEL_IDX_W-1:0]];
        end else begin
            panel_read_a = '0;
        end
    end
    endfunction

    function automatic [`XLEN-1:0] panel_read_b(input [`XLEN-1:0] index);
    begin
        if (index < `XLEN'(MAX_B_PANEL_WORDS)) begin
            panel_read_b = b_panel_words_r[index[B_PANEL_IDX_W-1:0]];
        end else begin
            panel_read_b = '0;
        end
    end
    endfunction

    wire [`XLEN-1:0] panel_a_word0 = panel_read_a(a_idx0);
    wire [`XLEN-1:0] panel_a_word1 = panel_read_a(a_idx1);
    wire [`XLEN-1:0] panel_b_word0 = panel_read_b(b_idx0);
    wire [`XLEN-1:0] panel_b_word1 = panel_read_b(b_idx1);

    wire [15:0] a0_lo = panel_a_word0[15:0];
    wire [15:0] a0_hi = panel_a_word0[31:16];
    wire [15:0] a1_lo = panel_a_word1[15:0];
    wire [15:0] a1_hi = panel_a_word1[31:16];
    wire [15:0] b0_lo = panel_b_word0[15:0];
    wire [15:0] b0_hi = panel_b_word0[31:16];
    wire [15:0] b1_lo = panel_b_word1[15:0];
    wire [15:0] b1_hi = panel_b_word1[31:16];

    wire [`XLEN-1:0] synth_mac = (`XLEN'(a0_lo) * `XLEN'(b0_lo))
                               + (`XLEN'(a0_hi) * `XLEN'(b0_hi))
                               + (`XLEN'(a1_lo) * `XLEN'(b1_lo))
                               + (`XLEN'(a1_hi) * `XLEN'(b1_hi))
                               + (k_tiles_r << `XLEN'(3))
                               + (n_tiles_r << `XLEN'(1))
                               + (k_idx_r << `XLEN'(4))
                               + (n_idx_r << `XLEN'(6));

    localparam integer DCACHE_WORD_SHIFT = `CLOG2(MEM_BUS_DATA_SIZE);

    wire [`XLEN-1:0] fetch_word_elem_index = fetch_index_r * `XLEN'(PANEL_WORD_ELEMS);
    wire [`XLEN-1:0] fetch_b_row = (b_stride_r != 0) ? (fetch_word_elem_index / b_stride_r) : '0;
    wire [`XLEN-1:0] fetch_b_col = (b_stride_r != 0) ? (fetch_word_elem_index % b_stride_r) : '0;
    wire [`XLEN-1:0] fetch_addr_byte = fetch_phase_b_r
                                     ? (b_panel_base_addr_r
                                      + (((fetch_b_row * c_stride_r) + fetch_b_col) << `XLEN'(1)))
                                     : (a_panel_base_addr_r
                                      + (fetch_word_elem_index << `XLEN'(1)));
    wire [DCACHE_WORD_SHIFT-1:0] fetch_addr_offset = fetch_addr_byte[DCACHE_WORD_SHIFT-1:0];
    wire [MEM_BUS_DATA_SIZE*8-1:0] fetch_rsp_shifted = mem_bus_if.rsp_data.data >> (`XLEN'(fetch_addr_offset) * `XLEN'(8));
    wire [`XLEN-1:0] fetch_rsp_word = fetch_rsp_shifted[`XLEN-1:0];
    wire [`MEM_ADDR_WIDTH-DCACHE_WORD_SHIFT-1:0] fetch_word_addr = fetch_addr_byte[`MEM_ADDR_WIDTH-1:DCACHE_WORD_SHIFT];
    `UNUSED_VAR (fetch_rsp_shifted[MEM_BUS_DATA_SIZE*8-1:`XLEN])

    assign mem_bus_if.req_valid = (state_r == STATE_FETCH_REQ);
    assign mem_bus_if.req_data.rw = 1'b0;
    assign mem_bus_if.req_data.addr = fetch_word_addr;
    assign mem_bus_if.req_data.data = '0;
    assign mem_bus_if.req_data.byteen = {MEM_BUS_DATA_SIZE{1'b1}};
    assign mem_bus_if.req_data.flags = '0;
    assign mem_bus_if.req_data.tag = '0;
    assign mem_bus_if.rsp_ready = 1'b1;

    localparam [3:0] STATE_IDLE      = 4'd0;
    localparam [3:0] STATE_FETCH_REQ = 4'd1;
    localparam [3:0] STATE_FETCH_RSP = 4'd2;
    localparam [3:0] STATE_LOAD_DESC = 4'd3;
    localparam [3:0] STATE_COMPUTE   = 4'd4;
    localparam [3:0] STATE_WRITEBACK = 4'd5;
    localparam [3:0] STATE_DONE      = 4'd6;

    always @(posedge clk) begin
        if (reset) begin
            a_offset_r <= '0;
            b_offset_r <= '0;
            c_addr_r <= '0;
            c_stride_r <= '0;
            a_stride_r <= '0;
            b_stride_r <= '0;
            a_seed_r <= '0;
            b_seed_r <= '0;
            a_panel_base_addr_r <= '0;
            b_panel_base_addr_r <= '0;
            k_tiles_r <= '0;
            n_tiles_r <= '0;
            compute_steps_total_r <= '0;
            compute_steps_done_r <= '0;
            writeback_count_r <= '0;
            writeback_signature_r <= '0;
            accum_r <= '0;
            k_idx_r <= '0;
            n_idx_r <= '0;
            write_addr_r <= '0;
            write_data_r <= '0;
            remaining_cycles_r <= '0;
            state_r <= STATE_IDLE;
            fetch_index_r <= '0;
            fetch_phase_b_r <= 1'b0;
            busy_r <= 1'b0;
            done_r <= 1'b0;
            for (integer i = 0; i < MAX_A_PANEL_WORDS; ++i) begin
                a_panel_words_r[i] <= '0;
            end
            for (integer i = 0; i < MAX_B_PANEL_WORDS; ++i) begin
                b_panel_words_r[i] <= '0;
            end
        end else begin
            if (run_valid) begin
                a_offset_r <= a_offset;
                b_offset_r <= b_offset;
                c_addr_r <= c_addr;
                c_stride_r <= c_stride;
                a_stride_r <= a_stride;
                b_stride_r <= b_stride;
                a_seed_r <= a_seed;
                b_seed_r <= b_seed;
                a_panel_base_addr_r <= a_word0;
                b_panel_base_addr_r <= b_word0;
                k_tiles_r <= run_k_tiles;
                n_tiles_r <= run_n_tiles;
                compute_steps_total_r <= run_cycles;
                compute_steps_done_r <= '0;
                writeback_count_r <= '0;
                writeback_signature_r <= '0;
                accum_r <= '0;
                k_idx_r <= '0;
                n_idx_r <= '0;
                write_addr_r <= c_addr;
                write_data_r <= '0;
                remaining_cycles_r <= run_cycles;
                state_r <= STATE_FETCH_REQ;
                fetch_index_r <= '0;
                fetch_phase_b_r <= 1'b0;
                busy_r <= 1'b1;
                done_r <= 1'b0;
                for (integer i = 0; i < MAX_A_PANEL_WORDS; ++i) begin
                    a_panel_words_r[i] <= '0;
                end
                for (integer i = 0; i < MAX_B_PANEL_WORDS; ++i) begin
                    b_panel_words_r[i] <= '0;
                end
            end else begin
                case (state_r)
                    STATE_IDLE: begin
                        busy_r <= 1'b0;
                    end
                    STATE_FETCH_REQ: begin
                        if (mem_bus_if.req_ready) begin
                            state_r <= STATE_FETCH_RSP;
                        end
                    end
                    STATE_FETCH_RSP: begin
                        if (mem_bus_if.rsp_valid) begin
                            if (fetch_phase_b_r) begin
                                if (fetch_index_r < `XLEN'(MAX_B_PANEL_WORDS)) begin
                                    b_panel_words_r[fetch_index_r] <= fetch_rsp_word;
                                end
                                if (fetch_index_r + `XLEN'(1) >= b_fetch_words_limited) begin
                                    state_r <= STATE_LOAD_DESC;
                                end else begin
                                    fetch_index_r <= fetch_index_r + `XLEN'(1);
                                    state_r <= STATE_FETCH_REQ;
                                end
                            end else begin
                                if (fetch_index_r < `XLEN'(MAX_A_PANEL_WORDS)) begin
                                    a_panel_words_r[fetch_index_r] <= fetch_rsp_word;
                                end
                                if (fetch_index_r + `XLEN'(1) >= a_fetch_words_limited) begin
                                    fetch_phase_b_r <= 1'b1;
                                    fetch_index_r <= '0;
                                    state_r <= STATE_FETCH_REQ;
                                end else begin
                                    fetch_index_r <= fetch_index_r + `XLEN'(1);
                                    state_r <= STATE_FETCH_REQ;
                                end
                            end
                        end
                    end
                    STATE_LOAD_DESC: begin
                        state_r <= STATE_COMPUTE;
                    end
                    STATE_COMPUTE: begin
                        accum_r <= accum_r + synth_mac;
                        if (compute_steps_done_r + `XLEN'(1) < compute_steps_total_r) begin
                            compute_steps_done_r <= compute_steps_done_r + `XLEN'(1);
                            if (remaining_cycles_r != 0) begin
                                remaining_cycles_r <= remaining_cycles_r - `XLEN'(1);
                            end
                            if (k_idx_r + `XLEN'(1) < k_tiles_r) begin
                                k_idx_r <= k_idx_r + `XLEN'(1);
                            end else begin
                                k_idx_r <= '0;
                                n_idx_r <= n_idx_r + `XLEN'(1);
                            end
                        end else begin
                            compute_steps_done_r <= compute_steps_total_r;
                            remaining_cycles_r <= '0;
                            k_idx_r <= '0;
                            state_r <= STATE_WRITEBACK;
                        end
                    end
                    STATE_WRITEBACK: begin
                        writeback_count_r <= writeback_count_r + `XLEN'(1);
                        write_addr_r <= c_addr_r + (n_tiles_r << `XLEN'(2));
                        write_data_r <= accum_r
                                      ^ compute_steps_total_r
                                      ^ (n_tiles_r << `XLEN'(8))
                                      ^ (k_tiles_r << `XLEN'(12))
                                      ^ c_stride_r;
                        writeback_signature_r <= accum_r
                                               ^ compute_steps_total_r
                                               ^ (n_tiles_r << `XLEN'(8))
                                               ^ (k_tiles_r << `XLEN'(12))
                                               ^ c_stride_r;
                        state_r <= STATE_DONE;
                    end
                    STATE_DONE: begin
                        busy_r <= 1'b0;
                        done_r <= 1'b1;
                    end
                    default: begin
                        state_r <= STATE_IDLE;
                        busy_r <= 1'b0;
                        done_r <= 1'b0;
                        c_stride_r <= '0;
                        a_stride_r <= '0;
                        b_stride_r <= '0;
                        a_seed_r <= '0;
                        b_seed_r <= '0;
                        a_panel_base_addr_r <= '0;
                        b_panel_base_addr_r <= '0;
                        remaining_cycles_r <= '0;
                        compute_steps_done_r <= '0;
                        writeback_count_r <= '0;
                        writeback_signature_r <= '0;
                        accum_r <= '0;
                        k_idx_r <= '0;
                        n_idx_r <= '0;
                        write_addr_r <= '0;
                        write_data_r <= '0;
                        fetch_index_r <= '0;
                        fetch_phase_b_r <= 1'b0;
                    end
                endcase
            end
        end
    end

    reg [`XLEN-1:0] status_r;
    always @(*) begin
        case (status_sel)
            `XLEN'(0): status_r = {{(`XLEN-2){1'b0}}, busy_r, done_r};
            `XLEN'(1): status_r = remaining_cycles_r;
            `XLEN'(2): status_r = k_tiles_r;
            `XLEN'(3): status_r = n_tiles_r;
            `XLEN'(4): status_r = a_offset_r;
            `XLEN'(5): status_r = b_offset_r;
            `XLEN'(6): status_r = c_addr_r;
            `XLEN'(7): status_r = `XLEN'(state_r);
            `XLEN'(8): status_r = compute_steps_done_r;
            `XLEN'(9): status_r = compute_steps_total_r;
            `XLEN'(10): status_r = writeback_count_r;
            `XLEN'(11): status_r = writeback_signature_r;
            `XLEN'(12): status_r = accum_r;
            `XLEN'(13): status_r = k_idx_r;
            `XLEN'(14): status_r = n_idx_r;
            `XLEN'(15): status_r = write_addr_r;
            `XLEN'(16): status_r = write_data_r;
            `XLEN'(17): status_r = a_stride_r;
            `XLEN'(18): status_r = b_stride_r;
            `XLEN'(19): status_r = c_stride_r;
            `XLEN'(20): status_r = a_seed_r;
            `XLEN'(21): status_r = b_seed_r;
            `XLEN'(22): status_r = panel_a_word0;
            `XLEN'(23): status_r = panel_a_word1;
            `XLEN'(24): status_r = panel_b_word0;
            `XLEN'(25): status_r = panel_b_word1;
            `XLEN'(26): status_r = a_panel_base_addr_r;
            `XLEN'(27): status_r = b_panel_base_addr_r;
            `XLEN'(28): status_r = a_fetch_words_limited;
            `XLEN'(29): status_r = b_fetch_words_limited;
            default:    status_r = '0;
        endcase
    end

    assign status = status_r;
    assign busy = busy_r;
    assign done = done_r;

endmodule
