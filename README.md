# AMX Benchmarks

This document coalesces data about real-world performance of the Apple AMX coprocessor. The motivating use case is electronic structure calculations (DFT simulations), which use complex-valued matrix multiplications and eigendecompositions. Interleaved complex numbers incur additional overhead compared to split complex numbers, but BLAS only accepts the interleaved format. This format underutilizes both NEON and AMX units.

Table of Contents
- [Motivation](#motivation)
- [Linear Algebra Benchmark](#linear-algebra-benchmark-gflopsk)
- [Related Work](#related-work)

## Motivation

This research should let me run 10x more large-scale quantum simulations, or $\sqrt[3]{10}$ times larger simulations at the same throughput. The intent is to do something like high-throughput materials screening. This means I should not focus too much on optimizing the completion time of a single supermassive simulation. Rather, optimize latency when several medium-sized systems are solved simultaneously.

According a [recent research paper (2023)](https://pubs.acs.org/doi/10.1021/acs.jctc.2c00983), the SCF iterations can be partitioned into stages with different precisions. The first iterations use single precision, while the last iterations use double precision. The researchers used a consumer RTX A4000 with a 1:32 ratio of FP64:FP32 compute power. They achieved 6-8x speedup over GPU FP64 and negligible accuracy loss. A host CPU was used for ZHEEV, but that operation only consumed 3-10% of the total time.

For Apple silicon, the best adaptation of this algorithm would use CPU and GPU simultaneously. The AMX would not perform the majority of operations, but its presence would still be important. Below is a tentative illustration of the scheme\**:

- 65% of iterations: GPU FP32 (CHEMM) + GPU FP32 (CHEEV)
- 15% of iterations: GPU FP32 (CHEMM) + GPU double-single (ZHEEV)
- 15% of iterations: AMX FP32 (CHEMM) + NEON FP64 (ZHEEV)
- 5% of iterations: AMX FP64 (ZHEMM) + NEON FP64 (ZHEEV)

> \*De-interleaves the complex multiplications (CHEMM, ZHEMM) into four separate multiplications of their real and complex parts (SGEMM, DGEMM). This improves ALU utilization with the AMX and `simdgroup_matrix`.

Using 75% of the performance cores' NEON, all of the AMX's FP64 GEMM compute, and all of the GPU's eFP64, the M1 Max could reach 1658 GFLOPS FP64. This is 4.3x faster than 100% of the performance cores' NEON alone and 2.8x faster than the GPU's eFP64 alone. However, using all of that simultaneously may cause thermal throttling, decreasing performance by up to 1.5x.

In another scheme, the AMX would perform most of the computations. Matrix sizes used for GEMM exceed the matrix sizes used for ZHEEV. ZHEEV is kn^3, where n is the number of valence electrons. Meanwhile, GEMM is kLn^2, where L is the number of grid cells. There are significantly more grid cells than valence electrons, by multiple orders of magnitude.

- 65% of iterations: AMX FP32 (CHEMM) + NEON FP32 (CHEEV)
- 30% of iterations: AMX FP32 (CHEMM) + NEON FP64 (ZHEEV)
- 5% of iterations: AMX FP64 (ZHEMM) + NEON FP64 (ZHEEV)

New scheme:

|                                  | AMX Vector | NEON Vector | GPU Matrix | GPU Vector | AMX Matrix |
| -------------------------------- | ---------- | ----------- | ---------- | ---------- | ---------- |
| Max Clock @ Full Utilization     | 3.228 GHz  | 3.132 GHz   | 1.296 GHz  | 1.296 GHz  | 3.228 GHz  |
| Max Observed Power               | ~12 W ???  | 43.9 W      | 42 W       | 51.1 W     | ~12 W ???  |
| Max Observed GFLOPS F32          | TBD        | 655         | 8435       | 10400      | 2746       |
| Max Observed GFLOPS F64          | TBD        | 352         | 0          | 0          | 700        |
| Max Theoretical GFLOPS FFMA32    | 413        | 801         | 8493       | 10617      | 3305       |
| Max Theoretical GFLOPS FDIV32    | 0          | 200         | 0          | 1769       | 0          |
| Max Theoretical GFLOPS FSQRT32   | 0          | 200         | 0          | 1327       | 0          |
| Max Theoretical GFLOPS (e)FFMA64 | 206        | 400         | TBD        | 589        | 826        |
| Max Theoretical GFLOPS (e)FDIV64 | 0          | 100         | 0          | 183        | 0          |
| Max Theoretical GFLOPS (e)FSQRT64 | 0         | 100         | 0          | 189        | 0          |

Perform integral calculations on the GPU, because eFP64 transcendentals are faster than NEON. The matrix multiplications take on the order of 100,000 microseconds for 100-atom systems, enough time to load-balance across AMX matrix and GPU matrix. Eigendecomposition should happen on NEON if possible (instead of the default AMX vector), and Cholesky on either NEON vector or AMX matrix. The GPU cannot offload work from the non-GEMM operations. However, Cholesky requires less operations than GEMM of the same dimension.

## Linear Algebra Benchmark: GFLOPS/k

GFLOPS is not a plural noun. GFLOPS is a rate: (G)Billion (FL)Floating Point (OP)Operations per (S)Second. The term GFLOPS/second is often used to remove ambiguity, except that translates to GFLOP/second/second. Data throughput is a measure of speed - speed requires units of velocity, not acceleration. Therefore, this repository uses the original term GFLOPS.

GFLOPs is a plural noun. Occasionally, I use GFLOPs to specify the number of floating-point operations required for a linear algebra operation. The capitalization of `s` will distinguish the metric from GFLOPS. There are not many other concise, consistent ways to describe both of these terms.

TODO: Explain O(kn^3), uncertainty in computational complexity, universal measure of time-to-solution (agnostic of precision or algorithm), why I used GFLOPS/0.25k for complex-valued operations to normalize for ALU utilization

```
GFLOPS/k = (matrix dimension)^3 / (time to solution)
Imagine a processor has 1000 GFLOPS, uses 10 watts.
OpenBLAS GEMM: real GLOPS/k = 800, but complex GFLOPS/k = 190
Real has 80% ALU / 8.0 watts.
Complex has 76% ALU / 7.6 watts, not 19% ALU / 1.9 watts.
Both operations have ~80% ALU and ~8 watts.

However, GFLOPS/0.25k = 4 * (GFLOPS/k) ~ 760
80% ALU is much closer to 76%, and shows that complex is 4% slower,
but not because it requires more computations. Also, you would think 
it's 4% **faster**, because it has **more** arithmetic intensity.
GFLOPS/0.25k is a fairer, more insightful comparison.

k_complex = 4k_real
k_real = 0.25k_complex

Real:    GFLOPS = GFLOPS/k * k_real
Complex: GFLOPS = GFLOPS/0.25k * 0.25k_complex
```

TODO: Compare Apple's new BLAS library to the old BLAS library:
- sgemm, dgemm, zgemm
- ssymm, dsymm, zhemm
- ssyev, dsyev, zheev, newer \_2stage approaches added to the newer Accelerate
- xcholesky, xpotrf, xtrsm
- appleblas_xgeadd added to the new LAPACK library

<!-- Testing 10 different configurations - increments of 128 between 256 and 1408, reporting fastest speed / optimal matrix size. Speed reported in GFLOPS/k for real, GFLOPS/0.25k for complex. Eigendecompositions will use \_2stage with the new BLAS, unless the divide-and-conquer algorithm shows a performance delta. OpenBLAS is accessed through NumPy. That may put OpenBLAS at a slight disadvantage; Accelerate is accessed through lower-overhead Swift bindings. -->

<!--
TODO: GPT-3.5 generated the code below. Use GPT-4 to generate the profiling tests. Use the same documentation practices as pioneered in [philipturner/applegpuinfo](https://github.com/philipturner/applegpuinfo).

<details>
<summary>Generated code</summary>

```swift
// Command 1: Generate C code that calls into the BLAS library to perform the DGEMM operation.
// Command 2: Translate what you just wrote to Swift.
// ---

import Accelerate

let m = 3, n = 4, k = 2  // Dimensions of A, B, and C

// Define matrices A, B, and C as arrays
var A = [Double](repeating: 0.0, count: m * k)
var B = [Double](repeating: 0.0, count: k * n)
var C = [Double](repeating: 0.0, count: m * n)

// Fill matrices A and B with some data
for i in 0..<m*k {
    A[i] = Double(i)
}
for i in 0..<k*n {
    B[i] = Double(i)
}

// Call BLAS to perform DGEMM
let lda = k, ldb = n, ldc = n
let alpha = 1.0, beta = 0.0
cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, Int32(m), Int32(n), Int32(k), alpha, A, lda, B, ldb, beta, &C, ldc)

// Print the result
for i in 0..<m {
    for j in 0..<n {
        print("\(C[i * n + j]) ", terminator: "")
    }
    print("")
}
```

</details>
-->

| Operation | k<sub>real</sub> | M1 Max, OpenBLAS | M1 Max, New Accelerate | A15, New Accelerate | Max GFLOPS | AMX ALU % | NEON ALU % |
| --------- | ---------------- | ---------------- | ---------------- | ------------- | ---------- | ---------- | -------- |
| SGEMM | 2 | 327.7 / 1408 | 1327.4 / 896 | | 2654.8 | 85.4% | 341.7% |
| DGEMM | 2 | 176.2 / 1152 | 337.9 / 608 | | 675.8 | 87.0% | 174.0% |
| ZGEMM | 2 | 148.4 / 1216 | 223.6 / 1408 | | 447.2 | 57.6% | 115.1% |
| SSYEV (e-vecs) | n/a |
| DSYEV (e-vecs) | n/a |
| ZHEEV (e-vecs) | n/a |
| SGESV | - |
| DGESV | - |
| ZGESV | - |
| SPOTRF | - |
| DPOTRF | - |
| ZPOTRF | - |
| STRSM | - |
| DTRSM | - |
| ZTRSM | - |

_GFLOPS/k for each operation used in quantum chemistry. This metric compares each operation's execution speed regardless of the algorithm used to perform it, or the formula used to estimate GFLOPS. Complex-valued operations use GFLOPS/0.25k to directly compare ALU utilization to real-valued operations. For every operation listed so far, complex-valued versions are slower because they must de-interleave the numbers before processing them._

## Related Work

| | ISA Documentation | Performance Documentation | OSS GEMM Libraries |
| - | - | - | - |
| Apple AMX | [corsix/amx](https://github.com/corsix/amx) | [philipturner/amx-benchmarks](https://github.com/philipturner/amx-benchmarks) | [xrq-phys/blis_apple](https://github.com/xrq-phys/blis_apple) |
| Apple GPU | [dougallj/applegpu](https://github.com/dougallj/applegpu) | [philipturner/metal-benchmarks](https://github.com/philipturner/metal-benchmarks) | [geohot/tinygrad](https://github.com/geohot/tinygrad) |
