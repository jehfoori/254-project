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

`ifdef EXT_F_ENABLE
`include "VX_fpu_define.vh"
`endif

`ifdef XLEN_64
    `define CSR_READ_64(addr, dst, src) \
        addr : dst = `XLEN'(src)
`else
    `define CSR_READ_64(addr, dst, src) \
        addr : dst = src[31:0]; \
        addr+12'h80 : dst = 32'(src[$bits(src)-1:32])
`endif

module VX_csr_data
import VX_gpu_pkg::*;
`ifdef EXT_F_ENABLE
import VX_fpu_pkg::*;
`endif
#(
    parameter `STRING INSTANCE_ID = "",
    parameter CORE_ID = 0
) (
    input wire                          clk,
    input wire                          reset,

    input base_dcrs_t                   base_dcrs,

`ifdef PERF_ENABLE
    input sysmem_perf_t                 sysmem_perf,
    input pipeline_perf_t               pipeline_perf,
`endif

    VX_commit_csr_if.slave              commit_csr_if,

`ifdef EXT_F_ENABLE
    VX_fpu_csr_if.slave                 fpu_csr_if [`NUM_FPU_BLOCKS],
`endif

    VX_mem_bus_if.master                tensor_mem_bus_if,

    input wire [PERF_CTR_BITS-1:0]      cycles,
    input wire [`NUM_WARPS-1:0]         active_warps,
    input wire [`NUM_WARPS-1:0][`NUM_THREADS-1:0] thread_masks,

    input wire                          read_enable,
    input wire [UUID_WIDTH-1:0]         read_uuid,
    input wire [NW_WIDTH-1:0]           read_wid,
    input wire [`VX_CSR_ADDR_BITS-1:0]  read_addr,
    output wire [`XLEN-1:0]             read_data_ro,
    output wire [`XLEN-1:0]             read_data_rw,

    input wire                          write_enable,
    input wire [UUID_WIDTH-1:0]         write_uuid,
    input wire [NW_WIDTH-1:0]           write_wid,
    input wire [`VX_CSR_ADDR_BITS-1:0]  write_addr,
    input wire [`XLEN-1:0]              write_data
);

    `UNUSED_VAR (write_wid)

    // CSRs Write /////////////////////////////////////////////////////////////

    reg [`XLEN-1:0] mscratch;
    reg [`XLEN-1:0] team_id;
    reg [`XLEN-1:0] team_rank;
    reg [`XLEN-1:0] team_size;
    reg [`XLEN-1:0] team_tile_rows;
    reg [`XLEN-1:0] team_global_stride;
    reg [1:0][`XLEN-1:0] team_src_offset;
    reg [1:0][`XLEN-1:0] team_copy_size;
    reg [1:0][`XLEN-1:0] team_dst_mask;
    reg [1:0][`XLEN-1:0] team_copy_mode;
    reg [1:0][`XLEN-1:0] team_global_addr;
    reg [`XLEN-1:0] tensor_a_offset;
    reg [`XLEN-1:0] tensor_b_offset;
    reg [`XLEN-1:0] tensor_c_addr;
    reg [`XLEN-1:0] tensor_c_stride;
    reg [`XLEN-1:0] tensor_a_stride;
    reg [`XLEN-1:0] tensor_b_stride;
    reg [`XLEN-1:0] tensor_k_tiles;
    reg [`XLEN-1:0] tensor_run_n_tiles;
    reg [`XLEN-1:0] tensor_status_sel;
    reg [`XLEN-1:0] tensor_a_seed;
    reg [`XLEN-1:0] tensor_b_seed;
    reg [`XLEN-1:0] tensor_a_word0;
    reg [`XLEN-1:0] tensor_a_word1;
    reg [`XLEN-1:0] tensor_b_word0;
    reg [`XLEN-1:0] tensor_b_word1;
    wire [`XLEN-1:0] tensor_status;
    wire              tensor_busy;
    wire              tensor_done;
    wire              tensor_run_fire = write_enable && (write_addr == `VX_CSR_TEAM_TENSOR_RUN);

    `UNUSED_VAR (tensor_busy)
    `UNUSED_VAR (tensor_done)

    VX_team_panel_engine #(
        .INSTANCE_ID (INSTANCE_ID)
    ) team_panel_engine (
        .clk        (clk),
        .reset      (reset),
        .run_valid  (tensor_run_fire),
        .a_offset   (tensor_a_offset),
        .b_offset   (tensor_b_offset),
        .c_addr     (tensor_c_addr),
        .c_stride   (tensor_c_stride),
        .a_stride   (tensor_a_stride),
        .b_stride   (tensor_b_stride),
        .k_tiles    (tensor_k_tiles),
        .n_tiles    (write_data),
        .a_seed     (tensor_a_seed),
        .b_seed     (tensor_b_seed),
        .a_word0    (tensor_a_word0),
        .a_word1    (tensor_a_word1),
        .b_word0    (tensor_b_word0),
        .b_word1    (tensor_b_word1),
        .status_sel (tensor_status_sel),
        .mem_bus_if (tensor_mem_bus_if),
        .status     (tensor_status),
        .busy       (tensor_busy),
        .done       (tensor_done)
    );

`ifdef EXT_F_ENABLE
    reg [`NUM_WARPS-1:0][INST_FRM_BITS+`FP_FLAGS_BITS-1:0] fcsr, fcsr_n;
    wire [`NUM_FPU_BLOCKS-1:0]              fpu_write_enable;
    wire [`NUM_FPU_BLOCKS-1:0][NW_WIDTH-1:0] fpu_write_wid;
    fflags_t [`NUM_FPU_BLOCKS-1:0]          fpu_write_fflags;

    for (genvar i = 0; i < `NUM_FPU_BLOCKS; ++i) begin : g_fpu_write
        assign fpu_write_enable[i] = fpu_csr_if[i].write_enable;
        assign fpu_write_wid[i]    = fpu_csr_if[i].write_wid;
        assign fpu_write_fflags[i] = fpu_csr_if[i].write_fflags;
    end

    always @(*) begin
        fcsr_n = fcsr;
        for (integer i = 0; i < `NUM_FPU_BLOCKS; ++i) begin
            if (fpu_write_enable[i]) begin
                fcsr_n[fpu_write_wid[i]][`FP_FLAGS_BITS-1:0] = fcsr[fpu_write_wid[i]][`FP_FLAGS_BITS-1:0]
                                                             | fpu_write_fflags[i];
            end
        end
        if (write_enable) begin
            case (write_addr)
                `VX_CSR_FFLAGS: fcsr_n[write_wid][`FP_FLAGS_BITS-1:0] = write_data[`FP_FLAGS_BITS-1:0];
                `VX_CSR_FRM:    fcsr_n[write_wid][INST_FRM_BITS+`FP_FLAGS_BITS-1:`FP_FLAGS_BITS] = write_data[INST_FRM_BITS-1:0];
                `VX_CSR_FCSR:   fcsr_n[write_wid] = write_data[`FP_FLAGS_BITS+INST_FRM_BITS-1:0];
            default:;
            endcase
        end
    end

    for (genvar i = 0; i < `NUM_FPU_BLOCKS; ++i) begin : g_fpu_csr_read_frm
        assign fpu_csr_if[i].read_frm = fcsr[fpu_csr_if[i].read_wid][INST_FRM_BITS+`FP_FLAGS_BITS-1:`FP_FLAGS_BITS];
    end

    always @(posedge clk) begin
        if (reset) begin
            fcsr <= '0;
        end else begin
            fcsr <= fcsr_n;
        end
    end
`endif

    always @(posedge clk) begin
        if (reset) begin
            mscratch <= base_dcrs.startup_arg;
            team_id <= '0;
            team_rank <= '0;
            team_size <= '0;
            team_tile_rows <= '0;
            team_global_stride <= '0;
            team_src_offset <= '0;
            team_copy_size <= '0;
            team_dst_mask <= '0;
            team_copy_mode <= '0;
            team_global_addr <= '0;
            tensor_a_offset <= '0;
            tensor_b_offset <= '0;
            tensor_c_addr <= '0;
            tensor_c_stride <= '0;
            tensor_a_stride <= '0;
            tensor_b_stride <= '0;
            tensor_k_tiles <= '0;
            tensor_run_n_tiles <= '0;
            tensor_status_sel <= '0;
            tensor_a_seed <= '0;
            tensor_b_seed <= '0;
            tensor_a_word0 <= '0;
            tensor_a_word1 <= '0;
            tensor_b_word0 <= '0;
            tensor_b_word1 <= '0;
        end
        if (write_enable) begin
            case (write_addr)
            `ifdef EXT_F_ENABLE
                `VX_CSR_FFLAGS,
                `VX_CSR_FRM,
                `VX_CSR_FCSR,
            `endif
                `VX_CSR_SATP,
                `VX_CSR_MSTATUS,
                `VX_CSR_MNSTATUS,
                `VX_CSR_MEDELEG,
                `VX_CSR_MIDELEG,
                `VX_CSR_MIE,
                `VX_CSR_MTVEC,
                `VX_CSR_MEPC,
                `VX_CSR_PMPCFG0,
                `VX_CSR_PMPADDR0: begin
                    // do nothing!
                end
                `VX_CSR_MSCRATCH: begin
                    mscratch <= write_data;
                end
                `VX_CSR_TEAM_ID: begin
                    team_id <= write_data;
                end
                `VX_CSR_TEAM_RANK: begin
                    team_rank <= write_data;
                end
                `VX_CSR_TEAM_SIZE: begin
                    team_size <= write_data;
                end
                `VX_CSR_TEAM_TILE_ROWS: begin
                    team_tile_rows <= write_data;
                end
                `VX_CSR_TEAM_GLOBAL_STRIDE: begin
                    team_global_stride <= write_data;
                end
                `VX_CSR_TEAM_SRC_OFFSET: begin
                    team_src_offset[0] <= write_data;
                end
                `VX_CSR_TEAM_COPY_SIZE: begin
                    team_copy_size[0] <= write_data;
                end
                `VX_CSR_TEAM_DST_MASK: begin
                    team_dst_mask[0] <= write_data;
                end
                `VX_CSR_TEAM_COPY_MODE: begin
                    team_copy_mode[0] <= write_data;
                end
                `VX_CSR_TEAM_GLOBAL_ADDR: begin
                    team_global_addr[0] <= write_data;
                end
                `VX_CSR_TEAM_SRC_OFFSET_1: begin
                    team_src_offset[1] <= write_data;
                end
                `VX_CSR_TEAM_COPY_SIZE_1: begin
                    team_copy_size[1] <= write_data;
                end
                `VX_CSR_TEAM_DST_MASK_1: begin
                    team_dst_mask[1] <= write_data;
                end
                `VX_CSR_TEAM_COPY_MODE_1: begin
                    team_copy_mode[1] <= write_data;
                end
                `VX_CSR_TEAM_GLOBAL_ADDR_1: begin
                    team_global_addr[1] <= write_data;
                end
                `VX_CSR_TEAM_TENSOR_A_OFFSET: begin
                    tensor_a_offset <= write_data;
                end
                `VX_CSR_TEAM_TENSOR_B_OFFSET: begin
                    tensor_b_offset <= write_data;
                end
                `VX_CSR_TEAM_TENSOR_C_ADDR: begin
                    tensor_c_addr <= write_data;
                end
                `VX_CSR_TEAM_TENSOR_C_STRIDE: begin
                    tensor_c_stride <= write_data;
                end
                `VX_CSR_TEAM_TENSOR_A_STRIDE: begin
                    tensor_a_stride <= write_data;
                end
                `VX_CSR_TEAM_TENSOR_B_STRIDE: begin
                    tensor_b_stride <= write_data;
                end
                `VX_CSR_TEAM_TENSOR_K_TILES: begin
                    tensor_k_tiles <= write_data;
                end
                `VX_CSR_TEAM_TENSOR_A_SEED: begin
                    tensor_a_seed <= write_data;
                end
                `VX_CSR_TEAM_TENSOR_B_SEED: begin
                    tensor_b_seed <= write_data;
                end
                `VX_CSR_TEAM_TENSOR_A_WORD0: begin
                    tensor_a_word0 <= write_data;
                end
                `VX_CSR_TEAM_TENSOR_A_WORD1: begin
                    tensor_a_word1 <= write_data;
                end
                `VX_CSR_TEAM_TENSOR_B_WORD0: begin
                    tensor_b_word0 <= write_data;
                end
                `VX_CSR_TEAM_TENSOR_B_WORD1: begin
                    tensor_b_word1 <= write_data;
                end
                `VX_CSR_TEAM_TENSOR_STATUS_SEL: begin
                    tensor_status_sel <= write_data;
                end
                `VX_CSR_TEAM_TENSOR_RUN: begin
                    tensor_run_n_tiles <= write_data;
                end
                default: begin
                    `ASSERT(0, ("%t: *** %s invalid CSR write address: %0h (#%0d)", $time, INSTANCE_ID, write_addr, write_uuid));
                end
            endcase
        end
    end

    // CSRs read //////////////////////////////////////////////////////////////

    reg [`XLEN-1:0] read_data_ro_w;
    reg [`XLEN-1:0] read_data_rw_w;
    reg read_addr_valid_w;

    always @(*) begin
        read_data_ro_w    = '0;
        read_data_rw_w    = '0;
        read_addr_valid_w = 1;
        case (read_addr)
            `VX_CSR_MVENDORID  : read_data_ro_w = `XLEN'(`VENDOR_ID);
            `VX_CSR_MARCHID    : read_data_ro_w = `XLEN'(`ARCHITECTURE_ID);
            `VX_CSR_MIMPID     : read_data_ro_w = `XLEN'(`IMPLEMENTATION_ID);
            `VX_CSR_MISA       : read_data_ro_w = `XLEN'({2'(`CLOG2(`XLEN/16)), 30'(`MISA_STD)});
        `ifdef EXT_F_ENABLE
            `VX_CSR_FFLAGS     : read_data_rw_w = `XLEN'(fcsr[read_wid][`FP_FLAGS_BITS-1:0]);
            `VX_CSR_FRM        : read_data_rw_w = `XLEN'(fcsr[read_wid][INST_FRM_BITS+`FP_FLAGS_BITS-1:`FP_FLAGS_BITS]);
            `VX_CSR_FCSR       : read_data_rw_w = `XLEN'(fcsr[read_wid]);
        `endif
            `VX_CSR_MSCRATCH   : read_data_rw_w = mscratch;

            `VX_CSR_WARP_ID    : read_data_ro_w = `XLEN'(read_wid);
            `VX_CSR_CORE_ID    : read_data_ro_w = `XLEN'(CORE_ID);
            `VX_CSR_ACTIVE_THREADS: read_data_ro_w = `XLEN'(thread_masks[read_wid]);
            `VX_CSR_ACTIVE_WARPS: read_data_ro_w = `XLEN'(active_warps);
            `VX_CSR_NUM_THREADS: read_data_ro_w = `XLEN'(`NUM_THREADS);
            `VX_CSR_NUM_WARPS  : read_data_ro_w = `XLEN'(`NUM_WARPS);
            `VX_CSR_NUM_CORES  : read_data_ro_w = `XLEN'(`NUM_CORES * `NUM_CLUSTERS);
            `VX_CSR_LOCAL_MEM_BASE: read_data_ro_w = `XLEN'(`LMEM_BASE_ADDR);
            `VX_CSR_TEAM_ID: read_data_rw_w = team_id;
            `VX_CSR_TEAM_RANK: read_data_rw_w = team_rank;
            `VX_CSR_TEAM_SIZE: read_data_rw_w = team_size;
            `VX_CSR_TEAM_TILE_ROWS: read_data_rw_w = team_tile_rows;
            `VX_CSR_TEAM_GLOBAL_STRIDE: read_data_rw_w = team_global_stride;
            `VX_CSR_TEAM_SRC_OFFSET: read_data_rw_w = team_src_offset[0];
            `VX_CSR_TEAM_COPY_SIZE: read_data_rw_w = team_copy_size[0];
            `VX_CSR_TEAM_DST_MASK: read_data_rw_w = team_dst_mask[0];
            `VX_CSR_TEAM_COPY_MODE: read_data_rw_w = team_copy_mode[0];
            `VX_CSR_TEAM_GLOBAL_ADDR: read_data_rw_w = team_global_addr[0];
            `VX_CSR_TEAM_SRC_OFFSET_1: read_data_rw_w = team_src_offset[1];
            `VX_CSR_TEAM_COPY_SIZE_1: read_data_rw_w = team_copy_size[1];
            `VX_CSR_TEAM_DST_MASK_1: read_data_rw_w = team_dst_mask[1];
            `VX_CSR_TEAM_COPY_MODE_1: read_data_rw_w = team_copy_mode[1];
            `VX_CSR_TEAM_GLOBAL_ADDR_1: read_data_rw_w = team_global_addr[1];
            `VX_CSR_TEAM_TENSOR_A_OFFSET: read_data_rw_w = tensor_a_offset;
            `VX_CSR_TEAM_TENSOR_B_OFFSET: read_data_rw_w = tensor_b_offset;
            `VX_CSR_TEAM_TENSOR_C_ADDR: read_data_rw_w = tensor_c_addr;
            `VX_CSR_TEAM_TENSOR_C_STRIDE: read_data_rw_w = tensor_c_stride;
            `VX_CSR_TEAM_TENSOR_A_STRIDE: read_data_rw_w = tensor_a_stride;
            `VX_CSR_TEAM_TENSOR_B_STRIDE: read_data_rw_w = tensor_b_stride;
            `VX_CSR_TEAM_TENSOR_K_TILES: read_data_rw_w = tensor_k_tiles;
            `VX_CSR_TEAM_TENSOR_A_SEED: read_data_rw_w = tensor_a_seed;
            `VX_CSR_TEAM_TENSOR_B_SEED: read_data_rw_w = tensor_b_seed;
            `VX_CSR_TEAM_TENSOR_A_WORD0: read_data_rw_w = tensor_a_word0;
            `VX_CSR_TEAM_TENSOR_A_WORD1: read_data_rw_w = tensor_a_word1;
            `VX_CSR_TEAM_TENSOR_B_WORD0: read_data_rw_w = tensor_b_word0;
            `VX_CSR_TEAM_TENSOR_B_WORD1: read_data_rw_w = tensor_b_word1;
            `VX_CSR_TEAM_TENSOR_STATUS_SEL: read_data_rw_w = tensor_status_sel;
            `VX_CSR_TEAM_TENSOR_STATUS: read_data_rw_w = tensor_status;
            `VX_CSR_TEAM_TENSOR_RUN: read_data_rw_w = tensor_run_n_tiles;

            `CSR_READ_64(`VX_CSR_MCYCLE, read_data_ro_w, cycles);

            `VX_CSR_MPM_RESERVED : read_data_ro_w = 'x;
            `VX_CSR_MPM_RESERVED_H : read_data_ro_w = 'x;

            `CSR_READ_64(`VX_CSR_MINSTRET, read_data_ro_w, commit_csr_if.instret);

            `VX_CSR_SATP,
            `VX_CSR_MSTATUS,
            `VX_CSR_MNSTATUS,
            `VX_CSR_MEDELEG,
            `VX_CSR_MIDELEG,
            `VX_CSR_MIE,
            `VX_CSR_MTVEC,
            `VX_CSR_MEPC,
            `VX_CSR_PMPCFG0,
            `VX_CSR_PMPADDR0 : read_data_ro_w = `XLEN'(0);

            default: begin
                read_addr_valid_w = 0;
                if ((read_addr >= `VX_CSR_MPM_USER   && read_addr < (`VX_CSR_MPM_USER + 32))
                 || (read_addr >= `VX_CSR_MPM_USER_H && read_addr < (`VX_CSR_MPM_USER_H + 32))) begin
                    read_addr_valid_w = 1;
                `ifdef PERF_ENABLE
                    case (base_dcrs.mpm_class)
                    `VX_DCR_MPM_CLASS_CORE: begin
                        case (read_addr)
                        // PERF: pipeline
                        `CSR_READ_64(`VX_CSR_MPM_SCHED_ID, read_data_ro_w, pipeline_perf.sched.idles);
                        `CSR_READ_64(`VX_CSR_MPM_SCHED_ST, read_data_ro_w, pipeline_perf.sched.stalls);
                        `CSR_READ_64(`VX_CSR_MPM_IBUF_ST, read_data_ro_w, pipeline_perf.issue.ibf_stalls);
                        `CSR_READ_64(`VX_CSR_MPM_SCRB_ST, read_data_ro_w, pipeline_perf.issue.scb_stalls);
                        `CSR_READ_64(`VX_CSR_MPM_OPDS_ST, read_data_ro_w, pipeline_perf.issue.opd_stalls);
                        `CSR_READ_64(`VX_CSR_MPM_SCRB_ALU, read_data_ro_w, pipeline_perf.issue.units_uses[EX_ALU]);
                        `CSR_READ_64(`VX_CSR_MPM_SCRB_LSU, read_data_ro_w, pipeline_perf.issue.units_uses[EX_LSU]);
                        `CSR_READ_64(`VX_CSR_MPM_SCRB_SFU, read_data_ro_w, pipeline_perf.issue.units_uses[EX_SFU]);
                    `ifdef EXT_F_ENABLE
                        `CSR_READ_64(`VX_CSR_MPM_SCRB_FPU, read_data_ro_w, pipeline_perf.issue.units_uses[EX_FPU]);
                    `endif
                    `ifdef EXT_TCU_ENABLE
                        `CSR_READ_64(`VX_CSR_MPM_SCRB_TCU, read_data_ro_w, pipeline_perf.issue.units_uses[EX_TCU]);
                    `endif
                        `CSR_READ_64(`VX_CSR_MPM_SCRB_CSRS, read_data_ro_w, pipeline_perf.issue.sfu_uses[SFU_CSRS]);
                        `CSR_READ_64(`VX_CSR_MPM_SCRB_WCTL, read_data_ro_w, pipeline_perf.issue.sfu_uses[SFU_WCTL]);
                        // PERF: memory
                        `CSR_READ_64(`VX_CSR_MPM_IFETCHES, read_data_ro_w, pipeline_perf.ifetches);
                        `CSR_READ_64(`VX_CSR_MPM_LOADS, read_data_ro_w, pipeline_perf.loads);
                        `CSR_READ_64(`VX_CSR_MPM_STORES, read_data_ro_w, pipeline_perf.stores);
                        `CSR_READ_64(`VX_CSR_MPM_IFETCH_LT, read_data_ro_w, pipeline_perf.ifetch_latency);
                        `CSR_READ_64(`VX_CSR_MPM_LOAD_LT, read_data_ro_w, pipeline_perf.load_latency);
                        default:;
                        endcase
                    end
                    `VX_DCR_MPM_CLASS_MEM: begin
                        case (read_addr)
                        // PERF: icache
                        `CSR_READ_64(`VX_CSR_MPM_ICACHE_READS, read_data_ro_w, sysmem_perf.icache.reads);
                        `CSR_READ_64(`VX_CSR_MPM_ICACHE_MISS_R, read_data_ro_w, sysmem_perf.icache.read_misses);
                        `CSR_READ_64(`VX_CSR_MPM_ICACHE_MSHR_ST, read_data_ro_w, sysmem_perf.icache.mshr_stalls);
                        // PERF: dcache
                        `CSR_READ_64(`VX_CSR_MPM_DCACHE_READS, read_data_ro_w, sysmem_perf.dcache.reads);
                        `CSR_READ_64(`VX_CSR_MPM_DCACHE_WRITES, read_data_ro_w, sysmem_perf.dcache.writes);
                        `CSR_READ_64(`VX_CSR_MPM_DCACHE_MISS_R, read_data_ro_w, sysmem_perf.dcache.read_misses);
                        `CSR_READ_64(`VX_CSR_MPM_DCACHE_MISS_W, read_data_ro_w, sysmem_perf.dcache.write_misses);
                        `CSR_READ_64(`VX_CSR_MPM_DCACHE_BANK_ST, read_data_ro_w, sysmem_perf.dcache.bank_stalls);
                        `CSR_READ_64(`VX_CSR_MPM_DCACHE_MSHR_ST, read_data_ro_w, sysmem_perf.dcache.mshr_stalls);
                        // PERF: lmem
                        `CSR_READ_64(`VX_CSR_MPM_LMEM_READS, read_data_ro_w, sysmem_perf.lmem.reads);
                        `CSR_READ_64(`VX_CSR_MPM_LMEM_WRITES, read_data_ro_w, sysmem_perf.lmem.writes);
                        `CSR_READ_64(`VX_CSR_MPM_LMEM_BANK_ST, read_data_ro_w, sysmem_perf.lmem.bank_stalls);
                        // PERF: l2cache
                        `CSR_READ_64(`VX_CSR_MPM_L2CACHE_READS, read_data_ro_w, sysmem_perf.l2cache.reads);
                        `CSR_READ_64(`VX_CSR_MPM_L2CACHE_WRITES, read_data_ro_w, sysmem_perf.l2cache.writes);
                        `CSR_READ_64(`VX_CSR_MPM_L2CACHE_MISS_R, read_data_ro_w, sysmem_perf.l2cache.read_misses);
                        `CSR_READ_64(`VX_CSR_MPM_L2CACHE_MISS_W, read_data_ro_w, sysmem_perf.l2cache.write_misses);
                        `CSR_READ_64(`VX_CSR_MPM_L2CACHE_BANK_ST, read_data_ro_w, sysmem_perf.l2cache.bank_stalls);
                        `CSR_READ_64(`VX_CSR_MPM_L2CACHE_MSHR_ST, read_data_ro_w, sysmem_perf.l2cache.mshr_stalls);
                        // PERF: l3cache
                        `CSR_READ_64(`VX_CSR_MPM_L3CACHE_READS, read_data_ro_w, sysmem_perf.l3cache.reads);
                        `CSR_READ_64(`VX_CSR_MPM_L3CACHE_WRITES, read_data_ro_w, sysmem_perf.l3cache.writes);
                        `CSR_READ_64(`VX_CSR_MPM_L3CACHE_MISS_R, read_data_ro_w, sysmem_perf.l3cache.read_misses);
                        `CSR_READ_64(`VX_CSR_MPM_L3CACHE_MISS_W, read_data_ro_w, sysmem_perf.l3cache.write_misses);
                        `CSR_READ_64(`VX_CSR_MPM_L3CACHE_BANK_ST, read_data_ro_w, sysmem_perf.l3cache.bank_stalls);
                        `CSR_READ_64(`VX_CSR_MPM_L3CACHE_MSHR_ST, read_data_ro_w, sysmem_perf.l3cache.mshr_stalls);
                        // PERF: memory
                        `CSR_READ_64(`VX_CSR_MPM_MEM_READS, read_data_ro_w, sysmem_perf.mem.reads);
                        `CSR_READ_64(`VX_CSR_MPM_MEM_WRITES, read_data_ro_w, sysmem_perf.mem.writes);
                        `CSR_READ_64(`VX_CSR_MPM_MEM_LT, read_data_ro_w, sysmem_perf.mem.latency);
                        // PERF: coalescer
                        `CSR_READ_64(`VX_CSR_MPM_COALESCER_MISS, read_data_ro_w, sysmem_perf.coalescer.misses);
                        default:;
                        endcase
                    end
                    default:;
                    endcase
                `endif
                end
            end
        endcase
    end

    assign read_data_ro = read_data_ro_w;
    assign read_data_rw = read_data_rw_w;

    `UNUSED_VAR (base_dcrs)

    `RUNTIME_ASSERT(~read_enable || read_addr_valid_w, ("%t: *** invalid CSR read address: 0x%0h (#%0d)", $time, read_addr, read_uuid))

`ifdef PERF_ENABLE
    `UNUSED_VAR (sysmem_perf.icache);
    `UNUSED_VAR (sysmem_perf.lmem);
`endif

endmodule
