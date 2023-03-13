# AMX Benchmarks

This repository is intended to coalesce data about real-world performance of the Apple AMX coprocessor. The end goal is to run quantum chemistry (DFT) simulations as fast as possible on Apple silicon. Therefore, it only focuses on data types and performance parameters likely useful toward the end goal. These include FP64, FP32, and BF16.

## Core Computing Unit: The AMX Block

As with [metal-benchmarks](https://github.com/philipturner/metal-benchmarks), I will establish the AMX "block" as the core unit of processing power. This is analogous to an GPU core. On die shots of Apple chips, there are many different types of AMX coprocessors. The all have a common characteristic: numerous visible blocks. The amount of processing power can be calculated exactly from (a) the CPU clock speed and (b) the number of AMX blocks.

```
Formula:
FP64 GFLOPS = 2 * 4 * 4 * GHz = 32 * GHz
FP32 GFLOPS = 2 * 8 * 8 * GHz = 128 * GHz
BF16 GFLOPS = 2 * 16 * 16 * GHz / 2 = 256 * GHz
```

Notice that 4 * FP64 = 32 bytes, half the AMX register size. Divide 64B / 32B and square it. An AMX block executes with such a throughput, that one 64B x 64B instruction will be completed every (64B / 32B) x (64B / 32B) = 4 clock cycles. Keep in mind that this is for a single block, not the entire AMX coprocessor. That would be like singling out one of the M1 Max's 32 GPU cores, and claiming that single core was actually the entire GPU. Next, move onto general-purpose CPU cores. Notice that 512 bits equals 64 bytes, the register size of AMX. This is not a coincidence. The CPU core that passes two 64-byte operands (1024 bits) to the AMX coprocessor each cycle, which doubled to 2048 bits with the A15/M2 generation. By comparison, the P-CPU's ALUs consume 1024 bits/cycle (FADD, 2 operands) or 1536 bits/cycle (FFMA, 3 operands).

```
Formula:
BITWIDTH = (256, 512) bits
FP64 GFLOPS = 2 * BITWIDTH / 64 * GHz = (8, 16) * GHz
FP32 GFLOPS = 2 * BITWIDTH / 32 * GHz = (16, 32) * GHz
BF16 GFLOPS = 2 * BITWIDTH / 16 * GHz = (32, 64) * GHz
```

Here are my two setups. Notice that within each P-block\*, the generation's CPU-AMX bandwidth allows a single\*\* core to access all the available AMX blocks.  The A14/M1 generation has 4 AMX blocks/P-block. The A15/M2 generation has 8 AMX block/P-clock. The number of CPUs varies within each generation, either 2-4 per P-block. The M1 Max cannot access all 8 AMX blocks from one CPU core, but its P-CPU is divided into two P-blocks. The AMX has only a single P-block, but can harness more AMX blocks per P-block. 

> \*Clarifying terminology: a P-block is a visible segment of the silicon die, comprising 2-4 P-CPU cores and its own coherent L2 cache. The term "AMX block", while sounding similar, has absolutely no relation. The P-block has a better analogy with the M1 Max half-GPU. It is a fundamental chip building block composed of multiple cores. It is not one of those cores itself.
>
> \*\*Accelerate probably doesn't want to have >1 CPU cores/P-block active while using the AMX. That would slightly throttle the block's clock speed without boosting theoretical processing power. This aligns with my GEMM benchmarks on M1 Max, which never utilize more than 200% CPU in the activity monitor. The chip has 2 P-blocks, which translates to 2 threads at full utilization.

| AMX Processor | Clock Speed | AMX Blocks | FP64 GFLOPS | FP32 GFLOPS | BF16 GFLOPS |
| --------- | ----------- | ---------- | ----------- | ----------- | ----- |
| M1 Max P-CPU | 3.228 GHz | 8 | 826 | 3305 | n/a |
| A15 P-CPU | 3.204 GHz | 8 | 820 | 3280 | 6561 |
| M1 Max E-CPU | 2.064 GHz | 2 | 132 | 528 | n/a |
| A15 E-CPU | 2.016 GHz | 2 | 129 | 516 | 1032 |

| NEON Processor | Clock Speed | CPU Cores | Execution Width | FP64 GFLOPS | FP32 GFLOPS | BF16 GFLOPS |
| --------- | ----------- | ---------- | ----------- | ------- | ----------- | ------ |
| M1 Max P-CPU | 3.228 GHz | 8 | 512 bits |
| A15 P-CPU | 3.204 GHz | 2 | 512 bits |
| M1 Max E-CPU | 2.064 GHz | 2 | 256 bits |
| A15 E-CPU | 2.016 GHz | 4 | 256 bits |
