//
//  Helpers.swift
//  AMXBenchmarks
//
//  Created by Philip Turner on 3/24/23.
//

import Foundation
import Accelerate
import ComplexModule
#if os(macOS)
import PythonKit
#endif

// TODO: Ask GPT-4 to rewrite this. Complex numbers have a different interface
// than real numbers, requires different Swift code.

protocol BLASLAPACKScalar {
  associatedtype BLASLAPACKMutablePointer
  associatedtype BLASLAPACKPointer
  associatedtype BLASLAPACKType
  associatedtype BLASLAPACKRealType
}

extension Float: BLASLAPACKScalar {
  typealias BLASLAPACKMutablePointer = UnsafeMutablePointer<Float32>
  typealias BLASLAPACKPointer = UnsafePointer<Float32>
  typealias BLASLAPACKType = Float32
  typealias BLASLAPACKRealType = Float32
}

extension Double: BLASLAPACKScalar {
  typealias BLASLAPACKMutablePointer = UnsafeMutablePointer<Float64>
  typealias BLASLAPACKPointer = UnsafePointer<Float64>
  typealias BLASLAPACKType = Float64
  typealias BLASLAPACKRealType = Float64
}

extension Complex<Double>: BLASLAPACKScalar {
  typealias BLASLAPACKMutablePointer = OpaquePointer
  typealias BLASLAPACKPointer = OpaquePointer
  typealias BLASLAPACKType = OpaquePointer
  typealias BLASLAPACKRealType = Float64
}

// define a struct that contains closures for BLAS/LAPACK functions
struct BLASLAPACKFunctions<T: BLASLAPACKScalar> {
    // define the function types for each BLAS/LAPACK function
  typealias GEMMFunction = (CBLAS_ORDER, CBLAS_TRANSPOSE, CBLAS_TRANSPOSE, Int, Int, Int, T.BLASLAPACKType, T.BLASLAPACKPointer, Int, T.BLASLAPACKPointer, Int, T.BLASLAPACKType, T.BLASLAPACKMutablePointer, Int) -> Void
  typealias SYEVFunction = (UnsafePointer<Int8>, UnsafePointer<Int8>, UnsafePointer<Int>, UnsafeMutablePointer<T.BLASLAPACKRealType>, UnsafePointer<Int>, T.BLASLAPACKMutablePointer, UnsafeMutablePointer<T.BLASLAPACKRealType>, UnsafePointer<Int>, UnsafeMutablePointer<Int>) -> Void
    typealias GESVFunction = (UnsafePointer<Int>, UnsafePointer<Int>, T.BLASLAPACKMutablePointer, UnsafePointer<Int>, UnsafeMutablePointer<Int>, T.BLASLAPACKMutablePointer, UnsafePointer<Int>, UnsafeMutablePointer<Int>) -> Void
    typealias POTRFFunction = (UnsafePointer<Int8>, UnsafePointer<Int>, T.BLASLAPACKMutablePointer, UnsafePointer<Int>, UnsafeMutablePointer<Int>) -> Void
  typealias TRSMFunction = (CBLAS_ORDER, CBLAS_SIDE, CBLAS_UPLO, CBLAS_TRANSPOSE, CBLAS_DIAG, Int, Int, T.BLASLAPACKType, T.BLASLAPACKPointer, Int, T.BLASLAPACKMutablePointer, Int) -> Void
    
    // define the properties that store the closures for each BLAS/LAPACK function
    let gemm: GEMMFunction
    let syev: SYEVFunction
    let gesv: GESVFunction
    let potrf: POTRFFunction
    let trsm: TRSMFunction
  
  // '(UnsafePointer<Int8>, UnsafePointer<Int8>, UnsafePointer<Int>, Optional<UnsafeMutablePointer<Float>>, UnsafePointer<Int>, Optional<UnsafeMutablePointer<Float>>, UnsafeMutablePointer<Float>, UnsafePointer<Int>, UnsafeMutablePointer<Int>) -> ()')
  // '(UnsafePointer<Int8>, UnsafePointer<Int8>, UnsafePointer<Int>, UnsafeMutablePointer<Float>, UnsafeMutablePointer<Int>, UnsafeMutablePointer<Float>, UnsafeMutablePointer<Float>, UnsafeMutablePointer<Int>, UnsafeMutablePointer<Int>) -> Void'
  
  // '(UnsafePointer<Int8>, UnsafePointer<Int>, Optional<UnsafeMutablePointer<Float>>, UnsafePointer<Int>, UnsafeMutablePointer<Int>) -> ()')
  // '(UnsafeMutablePointer<Int8>, UnsafeMutablePointer<Int>, UnsafeMutablePointer<Float>, UnsafeMutablePointer<Int>, UnsafeMutablePointer<Int>) -> Void'
    
    // initialize the struct with the closures for each BLAS/LAPACK function
    init(gemm: @escaping GEMMFunction,
         syev: @escaping SYEVFunction,
         gesv: @escaping GESVFunction,
         potrf: @escaping POTRFFunction,
         trsm: @escaping TRSMFunction) {
        self.gemm = gemm
        self.syev = syev
        self.gesv = gesv
        self.potrf = potrf
        self.trsm = trsm
    }
}

// extend Float to store a static variable that returns the set of BLAS/LAPACK functions for Float
extension Float {
    static var blaslapackFunctions: BLASLAPACKFunctions<Float> {
        return BLASLAPACKFunctions<Float>(
            gemm: cblas_sgemm,
            syev: ssyev_,
            gesv: sgesv_,
            potrf: spotrf_,
            trsm: cblas_strsm)
    }
}

// extend Double to store a static variable that returns the set of BLAS/LAPACK functions for Double
extension Double {
    static var blaslapackFunctions: BLASLAPACKFunctions<Double> {
        return BLASLAPACKFunctions<Double>(
            gemm: cblas_dgemm,
            syev: dsyev_,
            gesv: dgesv_,
            potrf: dpotrf_,
            trsm: cblas_dtrsm)
    }
}

// aka '(UnsafePointer<Int8>, UnsafePointer<Int8>, UnsafePointer<Int>, Optional<OpaquePointer>, UnsafePointer<Int>, Optional<UnsafeMutablePointer<Double>>, OpaquePointer, UnsafePointer<Int>, Optional<UnsafeMutablePointer<Double>>, UnsafeMutablePointer<Int>) -> ()')



// aka '(UnsafePointer<Int8>, UnsafePointer<Int8>, UnsafePointer<Int>, UnsafeMutablePointer<Double>, UnsafePointer<Int>, OpaquePointer, UnsafeMutablePointer<Double>, UnsafePointer<Int>, UnsafeMutablePointer<Int>) -> ()')

// extend Complex<Double> to store a static variable that returns the set of BLAS/LAPACK functions for Complex<Double>
extension Complex where RealType == Double {
    static var blaslapackFunctions: BLASLAPACKFunctions<Complex<Double>> {
        return BLASLAPACKFunctions<Complex<Double>>(
            gemm: cblas_zgemm,
            syev: zheev_,
            gesv: zgesv_,
            potrf: zpotrf_,
            trsm: cblas_ztrsm)
    }
}
