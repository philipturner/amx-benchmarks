# AMX Benchmarks

This repository is intended to coalesce data about real-world performance of the Apple AMX coprocessor. The end goal is to run quantum chemistry (DFT) simulations as fast as possible on Apple silicon.

## Core Computing Unit: The AMX Block

As with [metal-benchmarks](https://github.com/philipturner/metal-benchmarks), I will establish the AMX "block" as the core unit of processing power. This is analogous to an GPU core. On die shots of Apple chips, there are many different types of AMX coprocessors. The all have a common characteristic: numerous visible blocks. The amount of processing power can be calculated exactly (a) the CPU clock speed and (b) the number of AMX blocks.

```
Formula:
FP64 GFLOPS = 2 * 4 * 4 * GHz
FP32 GFLOPS = 2 * 8 * 8 * GHz
BF16 GFLOPS = 2 * 16 * 16 * GHz / 2
```

Notice that 4 * FP64 = 32 bytes, half the AMX register size. Divide 64B / 32B and square it. An AMX block executes with such a throughput, that one 64B x 64B instruction will be completed every (64B / 32B) x (64B / 32B) = 4 clock cycles. Keep in mind that this is for a single block, not the entire AMX coprocessor. That would be like singling out one of the M1 Max's 32 GPU cores, and claiming that single core was actually the entire GPU. Next, move onto general-purpose CPU cores. Notice that 512 bits equals 64 bytes, the register size of AMX. This is not a coincidence. The CPU core that passes two 64-byte operands (1024 bits) to the AMX each cycle, which doubled to 2048 bits with the A15/M2 generation. By comparison, the P-CPU's ALUs consume 1024 bits/cycle (FADD, 2 operands) or 1536 bits/cycle (FFMA, 3 operands).

```
Formula:
BITWIDTH = (256, 512) bits
FP64 GFLOPS = 2 * BITWIDTH / 64 * GHz
FP32 GFLOPS = 2 * BITWIDTH / 32 * GHz
BF16 GFLOPS = 2 * BITWIDTH / 16 * GHz
```

Here are my two setups:

| AMX Processor | Clock Speed | AMX Blocks | FP64 GFLOPS | FP32 GFLOPS | BF16 GFLOPS |
| --------- | ----------- | ---------- | ----------- | ----------- | ----- |
| M1 Max P-CPU | 3228 MHz
| A15 P-CPU | 3204 MHz |
| M1 Max E-CPU | 2064 MHz |
| A15 E-CPU | 2016 MHz |

| NEON Processor | Clock Speed | CPU Cores | Execution Width | FP64 GFLOPS | FP32 GFLOPS | BF16 GFLOPS |
| --------- | ----------- | ---------- | ----------- | ------- | ----------- | ------ |
| M1 Max P-CPU | 8 | 3228 MHz
| A15 P-CPU | 2 | 3204 MHz |
| M1 Max E-CPU |
| A15 E-CPU |
