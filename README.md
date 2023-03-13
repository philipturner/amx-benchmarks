# AMX Benchmarks

This repository is intended to coalesce data about real-world performance of the Apple AMX coprocessor. The end goal is to run quantum chemistry (DFT) simulations as fast as possible on Apple silicon. Therefore, the repository focuses on data types and performance parameters potentially useful toward the end goal. Relevant data types include FP64, FP32, and BF16.

Table of Contents
- [Core Computing Unit](#core-computing-unit-the-amx-block)
- [Linear Algebra Benchmark](#linear-algebra-benchmark-gflopsk)
- [Related Work](#related-work)

## Core Computing Unit: The AMX Block

As with [metal-benchmarks](https://github.com/philipturner/metal-benchmarks), I will establish the AMX "block" as the fundamental unit of processing power. This is analogous to the CPU "core" and the GPU\* "core". On die shots of Apple chips, there are many different types of AMX coprocessors. The all have a common characteristic: numerous visible blocks. The amount of processing power can be calculated exactly from (a) the CPU clock speed and (b) the number of AMX blocks.

> \*Fun fact: the Apple GPU has its own hidden CPU just to dispatch commands to all the GPU cores. This is probably inconspicuous - if someone locates it on the die shot, please notify me! I hypothesize it would be duplicated for the second half of the M1 Max GPU. It runs a modified version of the Darwin kernel and I have no idea what its clock speed is. AMD RDNA 3 has a "frontend" clock speed at 2.5 GHz, famously decoupled from the GPU cores' 2.3 GHz for power efficiency. The M1 Max half-GPU's dedicated CPU core would probably run at ~1 GHz.

```
Formula:
FP64 GFLOPS = 2 * 4 * 4 * GHz = 32 * GHz
FP32 GFLOPS = 2 * 8 * 8 * GHz = 128 * GHz
BF16 GFLOPS = 2 * 16 * 16 * GHz / 2 = 256 * GHz
```

Notice that 4 * FP64 = 32 bytes, half the AMX register size. Divide 64B / 32B and square it. An AMX block executes with such a throughput, that one 64B x 64B instruction will be completed every (64B / 32B) x (64B / 32B) = 4 clock cycles. Keep in mind that this is for a single block, not the entire AMX coprocessor. Confusing the two terms equates to singling out one of the M1 Max's 32 GPU cores, and claiming that single core is actually the entire GPU.

Next, move on to general-purpose CPU cores. 512 bits equals 64 bytes, the register size of AMX. The CPU core passes two 64-byte operands (1024 bits) to the AMX coprocessor each cycle, or 2048 bits with the A15/M2 generation. On A14/M1, the bandwidth can sustain one AMX instruction/cycle - 4 quarter-instructions/cycle. This directly corresponds to 4 AMX blocks that each execute 0.25 64B x 64B instructions/cycle. The CPU-AMX bandwidth doubles with A15/M2, enabling 8 AMX blocks per P-block\*. The result: the CPU core now needs to dispatch 2 full AMX instructions/cycle, or 8 quarter-instructions. 

> \*Clarifying terminology: a P-block is a visible segment of the silicon die, comprising 2-4 P-CPU cores and its own coherent L2 cache. The term "AMX block", while sounding similar, has absolutely no relation. The P-block has a better analogy with the M1 Max half-GPU. It is a fundamental chip building block composed of multiple cores. It is not one of those cores itself.

```
Formula:
BITWIDTH = (256, 512) bits
FP64 GFLOPS = 2 * BITWIDTH / 64 * GHz = (8, 16) * GHz
FP32 GFLOPS = 2 * BITWIDTH / 32 * GHz = (16, 32) * GHz
BF16 GFLOPS = 2 * BITWIDTH / 16 * GHz = (32, 64) * GHz
```

On both M1 and M2, the AMX bandwidth is also close to the register-ALU bandwidth. The P-CPU's ALUs consume 1024 bits/cycle (FADD, 2 operands) or 1536 bits/cycle (FFMA, 3 operands). On A14/M1, the E-cores have half the register bandwidth (256 bits/vector operand instead of 512) and probably half the AMX bandwidth too. The result: 2 AMX blocks/E-block instead of 4. Apple did not change this paradigm with A15/M2. The P-cores get double bandwidth, the E-cores do not.

```
GPU core:
ALU_WIDTH = 64 (A14), 128 (M1, A15)
FP32 GFLOPS = 2 * ALU_WIDTH * GHz = (128, 256) * GHz
FP16 GFLOPS = 2 * 128 * GHz = 256 * GHz

ANE core:
FP16 GFLOPS = 2 * 16 * 16 * GHz = 512 * GHz
```

| Core Type | FP64 GFLOPS Formula | FP32 GFLOPS Formula | FP16 GFLOPS Formula |
| --------- | ------------------- | ---------- | ------------------- |
| P-CPU core | 2 * 512 / 64 * GHz | 2 * 512 / 32 * GHz | 2 * 512 / 16 * GHz |
| M1 GPU core | - | 2 * 128 * GHz | - |
| A12 ANE core | - | - | 2 * 16 * 16 * GHz |
| AMX block | 2 * 4 * 4 * GHz | 2 * 8 * 8 * GHz | 2 * 16 * 16 * GHz / 2 |

| Core Type | FP64 GFLOPS Formula | FP32 GFLOPS Formula | FP16 GFLOPS Formula |
| --------- | ------------------- | ---------- | ------------------- |
| P-CPU core | 16 * GHz | 32 * GHz | 64 * GHz |
| M1 GPU core | - | 256 * GHz | - |
| A12 ANE core | - | - | 512 * GHz |
| AMX block | 32 * GHz | 128 * GHz | 256 * GHz |

Here are my two setups. Notice that within each P-block, the generation's CPU-AMX bandwidth allows a single\* core to access all the available AMX blocks. The A14/M1 generation has 4 AMX blocks/P-block. The A15/M2 generation has 8 AMX blocks/P-clock. The number of CPU cores varies within the generation, at either 2-4 per P-block. The M1 Max cannot access all 8 AMX blocks from one CPU core, but its P-CPU is divided into two P-blocks. In contrast, A15 has only a single P-block, but can harness more AMX blocks per P-block. 

> \*Accelerate probably doesn't want to have >1 CPU cores/P-block active while using the AMX. That would slightly throttle the block's clock speed without boosting theoretical processing power. This aligns with my GEMM benchmarks on M1 Max, which never utilize more than 200% CPU in the activity monitor. The chip has 2 P-blocks, which translates to 2 threads at full utilization.

| AMX Processor | Clock Speed | AMX Blocks | FP64 GFLOPS | FP32 GFLOPS | BF16 GFLOPS |
| --------- | ----------- | ---------- | ----------- | ----------- | ----- |
| M1 Max P-CPU | 3.228 GHz | 8 | 826 | 3305 | - |
| A15 P-CPU | 3.204 GHz | 8 | 820 | 3280 | 6561 |
| M1 Max E-CPU | 2.064 GHz | 2 | 132 | 528 | - |
| A15 E-CPU | 2.016 GHz | 2 | 129 | 516 | 1032 |

| NEON Processor | Clock Speed | CPU Cores | Execution Width | FP64 GFLOPS | FP32 GFLOPS | BF16 GFLOPS |
| --------- | ----------- | ---------- | ----------- | ------- | ----------- | ------ |
| M1 Max P-CPU | 3.228 GHz | 8 | 512 bits | 413 | 826 | - |
| A15 P-CPU | 3.204 GHz | 2 | 512 bits | 102 | 205 | 410 |
| M1 Max E-CPU | 2.064 GHz | 2 | 256 bits | 33 | 66 | - |
| A15 E-CPU | 2.016 GHz | 4 | 256 bits | 64 | 129 | 258 |

| Accessible from Single-Core | RAM GB/s | System-Level Cache | L2D/P-Block | AMX FP32 GFLOPS |
| - | - | - | - | - |
| M1 Max | 100 GB/s | 24-48 MB MB | 12 MB | 1652 |
| A15 | 34 GB/s | 32 MB | 12 MB | 3280 |

It looks like the A15 on my iPhone 13 Pro Max is more suited for running quantum chemistry simulations than my M1 Max. With just a single CPU core, it can access the same AMX compute power as my M1 Max. This would simplify linear algebra kernels by removing the need for communication between multiple CPU threads. The A15 has less memory bandwidth, but that shouldn't be a problem if I keep everything in SLC. The A15's SLC can fit one 1448x1448 double-complex matrix, or the valence electrons of 362 carbon atoms. That amount of atoms is vastly more than what's feasible on modern desktops. So there's a very high chance I can keep the entire DFT simulation in SLC or even P-CPU L2D.

Furthermore, BF16 would double the arithmetic intensity of AMX calculations, only requiring 1024 bits/cycle of CPU-AMX bandwidth. The greater dynamic range of BF16 makes it more likely than FP16 to be useful for general-purpose processing. BF16 would also double the single-core NEON general-purpose FFMA processing power to 205 GFLOPS. It may halve the <s>memory</s> SLC bandwidth and accelerate O(n^2) GEMV operations that don't fit on the AMX. But even without utilizing BF16, the A15 still seems like the ideal chip for mixed FP32-FP64 quantum chemistry.

## Linear Algebra Benchmark: GFLOPS/k

TODO: Explain O(kn^3), uncertainty in computational complexity, visualize GFLOPS/k like the slope of a line, GFLOPS/0.25k for complex-valued operations

```
Real:    GFLOPS = GFLOPS/k * k_real
Complex: GFLOPS = GFLOPS/0.25k * 0.25k_complex

k_complex = 4k_real
k_real = 0.25k_complex
```

TODO: Compare Apple's new BLAS library to the old BLAS library:
- sgemm, dgemm, zgemm
- ssyevd, dsyevd, zheevd, faster \_2stage approaches added to the new LAPACK library
- xcholesky, xpotrf, xtrsm
- appleblas_xgeadd added to the new LAPACK library

## Related Work

| | ISA Documentation | Performance Documentation |
| - | - | - |
| Apple AMX | [corsix/amx](https://github.com/corsix/amx) | [philipturner/amx-benchmarks](https://github.com/philipturner/amx-benchmarks) |
| Apple GPU | [dougallj/applegpu](https://github.com/dougallj/applegpu) | [philipturner/metal-benchmarks](https://github.com/philipturner/metal-benchmarks) |
