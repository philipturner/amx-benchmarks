# AMX Benchmarks

This repository is intended to coalesce data about real-world performance of the Apple AMX coprocessor. The end goal is to run quantum chemistry (DFT) simulations as fast as possible on Apple silicon. Therefore, the repository focuses on data types and performance parameters potentially useful toward the end goal. Relevant data types include FP64, FP32, and BF16.

Table of Contents
- [Motivation](#motivation)
- [Linear Algebra Benchmark](#linear-algebra-benchmark-gflopsk)
- [Related Work](#related-work)

## Motivation

Hopefully this lets me run 10x more large-scale simulations, or more complex simulations at the same throughput. The intent is to generate enough data to train a neural network, or enable real-time reinforcement learning. This means I should not focus too much on optimizing the completion time of a single supermassive simulation. Rather, optimize latency when ~32 medium-sized simulations run on multiple CPU cores simultaneously.

If a neural network can be trained, it might traverse a solution space more efficiently\* than brute force. This would boost performance much more than 10x. It also leads to another similar idea. The GPU can be harnessed for mixed-precision simulations (~1% FP64), with higher throughput but not catching ill-conditioned systems. After finding a few candidate nanostructures with promising results, validate them on the AMX with ~10% FP64. A final human validation can occur in 100% FP64.

> \*Or generalize its knowledge to nanosystems so large, they take a day to validate through DFT.

A good illustration might be this hierarchy:
- 3D neural network: 50% FP16, 50% FP32, extremely high screening throughput - O(n)
- GPU DFT: 99-99.9% FP32, 0.1-1% FP64, high screening throughput - O(n^3)
- AMX DFT: 80-90% FP32, 10-20% FP64, reduced screening throughput - O(n^3)
- AMX DFT: 100% FP64, optimized for single-simulation latency - O(n^3)

## Linear Algebra Benchmark: GFLOPS/k

GFLOPS is not a plural noun. GFLOPS is a rate: (G)Billion (FL)Floating Point (OP)Operations per (S)Second. People sometimes say GFLOPS/second to clarify. That translates to GFLOP/second/second. Data throughput is a measure of speed. Speed requires units of velocity, not acceleration. GFLOPs is a plural noun. Occasionally, I use GFLOPs to specify the number of floating-point operations required for a linear algebra operation.  Pay close attention to the capitalization of `s`. Source code in this repository should use `numGFLOP(S|s)` to clarify the difference, while respecting the convention of camel case.

TODO: Explain O(kn^3), uncertainty in computational complexity, universal measure of time-to-solution (agnostic of precision or algorithm), visualize GFLOPS/k like the slope of a line, GFLOPS/0.25k for complex-valued operations to compare ALU utilization

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
