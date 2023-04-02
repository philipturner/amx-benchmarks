# AMX Benchmarks

This repository is intended to coalesce data about real-world performance of the Apple AMX coprocessor. The end goal is to run quantum chemistry (DFT) simulations as fast as possible on Apple silicon. Therefore, the repository focuses on data types and performance parameters potentially useful toward the end goal. Relevant data types include FP64, FP32, and BF16.

Table of Contents
- [Motivation](#motivation)
- [Linear Algebra Benchmark](#linear-algebra-benchmark-gflopsk)
- [Related Work](#related-work)

## Motivation

Hopefully this lets me run 10x more large-scale simulations, or more complex simulations at the same throughput. The intent is to generate enough data to train a neural network, or enable real-time reinforcement learning. This means I should not focus too much on optimizing the completion time of a single supermassive simulation. Rather, optimize latency when ~32 medium-sized simulations run on multiple CPU cores simultaneously.

If a neural network can be trained, it might traverse a solution space more efficiently\* than brute force. This would boost performance much more than 10x. According a [recent research paper (2023)](https://pubs.acs.org/doi/10.1021/acs.jctc.2c00983), the SCF iterations can be partitioned into stages with different precisions. The first iterations use single precision, while the last iterations use double precision. The researchers used a consumer RTX A4000 with 1:32 FP64:FP32 ratio and with negligible accuracy loss. For Apple silicon, the best adaptation of this algorithm would use CPU and GPU simultaneously. The AMX would not perform the majority of operations, but its presence it still important. Below is a tentative illustration of the scheme:

- ~75% of iterations: GPU FP32 (GEMM) + GPU FP32 (ZHEEV)
- ~10% of iterations: GPU FP32 (GEMM) + GPU double-single (ZHEEV)
- ~10% of iterations: AMX FP32 (GEMM) + NEON FP64 (ZHEEV)
- ~5% of iterations: AMX FP64 (GEMM) + NEON FP64 (ZHEEV)

> \*Or generalize its knowledge to nanosystems so large, they take a day to validate through DFT.

## Linear Algebra Benchmark: GFLOPS/k

GFLOPS is not a plural noun. GFLOPS is a rate: (G)Billion (FL)Floating Point (OP)Operations per (S)Second. People sometimes say GFLOPS/second to clarify. That translates to GFLOP/second/second. Data throughput is a measure of speed. Speed requires units of velocity, not acceleration. GFLOPs is a plural noun. Occasionally, I use GFLOPs to specify the number of floating-point operations required for a linear algebra operation.  Pay close attention to the capitalization of `s`. Source code in this repository should use `numGFLOP(S|s)` to clarify the difference, while respecting the convention of camel case.

TODO: Explain O(kn^3), uncertainty in computational complexity, universal measure of time-to-solution (agnostic of precision or algorithm), visualize GFLOPS/k like the slope of a line, GFLOPS/0.25k for complex-valued operations to normalize for ALU utilization

```
Real:    GFLOPS = GFLOPS/k * k_real
Complex: GFLOPS = GFLOPS/0.25k * 0.25k_complex

k_complex = 4k_real
k_real = 0.25k_complex
```

TODO: Compare Apple's new BLAS library to the old BLAS library:
- sgemm, dgemm, zgemm
- ssymm, dsymm, zhemm
- ssyevd, dsyevd, zheevd, faster \_2stage approaches added to the new LAPACK library
- xcholesky, xpotrf, xtrsm
- appleblas_xgeadd added to the new LAPACK library

Testing 10 different configurations - increments of 128 between 256 and 1408, reporting fastest speed / optimal matrix size. Speed reported in GFLOPS/k for real, GFLOPS/0.25k for complex. Eigendecompositions will use \_2stage with the new BLAS, unless the divide-and-conquer algorithm shows a performance delta. OpenBLAS is accessed through NumPy. That may put OpenBLAS at a slight disadvantage; Accelerate is accessed through lower-overhead Swift bindings.

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

| Operation | k<sub>real</sub> | M1 Max, OpenBLAS | M1 Max, Old BLAS | M1 Max, New BLAS | A15, Old BLAS | A15, New BLAS |
| --------- | ---------------- | ---------------- | ---------------- | ---------------- | ------------- | ------------- |
| SGEMM | 2 |
| DGEMM | 2 |
| ZGEMM | 2 |
| SSYMM | 2 |
| DSYMM | 2 |
| ZHEMM | 2 |
| SSYEVD (e-vals) | n/a |
| DSYEVD (e-vals) | n/a |
| ZHEEVD (e-vals) | n/a |
| SSYEVD (e-vecs) | n/a |
| DSYEVD (e-vecs) | n/a |
| ZHEEVD (e-vecs) | n/a |

## Related Work

| | ISA Documentation | Performance Documentation | OSS GEMM Libraries |
| - | - | - | - |
| Apple AMX | [corsix/amx](https://github.com/corsix/amx) | [philipturner/amx-benchmarks](https://github.com/philipturner/amx-benchmarks) | [xrq-phys/blis_apple](https://github.com/xrq-phys/blis_apple) |
| Apple GPU | [dougallj/applegpu](https://github.com/dougallj/applegpu) | [philipturner/metal-benchmarks](https://github.com/philipturner/metal-benchmarks) | [geohot/tinygrad](https://github.com/geohot/tinygrad) |
