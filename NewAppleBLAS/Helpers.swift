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

// sgemm: (UnsafePointer<CChar>, UnsafePointer<CChar>, UnsafePointer<__LAPACK_int>, UnsafePointer<__LAPACK_int>, UnsafePointer<__LAPACK_int>, UnsafePointer<Float>, UnsafePointer<Float>?, UnsafePointer<__LAPACK_int>, UnsafePointer<Float>?, UnsafePointer<__LAPACK_int>, UnsafePointer<Float>, UnsafeMutablePointer<Float>?, UnsafePointer<__LAPACK_int>) -> Void
// cgemm: (UnsafePointer<CChar>, UnsafePointer<CChar>, UnsafePointer<__LAPACK_int>, UnsafePointer<__LAPACK_int>, UnsafePointer<__LAPACK_int>, OpaquePointer, OpaquePointer?, UnsafePointer<__LAPACK_int>, OpaquePointer?, UnsafePointer<__LAPACK_int>, OpaquePointer, OpaquePointer?, UnsafePointer<__LAPACK_int>) -> Void
// ssyev: (UnsafePointer<CChar>, UnsafePointer<CChar>, UnsafePointer<__LAPACK_int>, UnsafeMutablePointer<Float>?, UnsafePointer<__LAPACK_int>, UnsafeMutablePointer<Float>?, UnsafeMutablePointer<Float>, UnsafePointer<__LAPACK_int>, UnsafeMutablePointer<__LAPACK_int>) -> Void
// cheev: (UnsafePointer<CChar>, UnsafePointer<CChar>, UnsafePointer<__LAPACK_int>, OpaquePointer?, UnsafePointer<__LAPACK_int>, UnsafeMutablePointer<Float>?, OpaquePointer, UnsafePointer<__LAPACK_int>, UnsafeMutablePointer<Float>?, UnsafeMutablePointer<__LAPACK_int>) -> Void
// sgesv: (UnsafePointer<__LAPACK_int>, UnsafePointer<__LAPACK_int>, UnsafeMutablePointer<Float>?, UnsafePointer<__LAPACK_int>, UnsafeMutablePointer<__LAPACK_int>?, UnsafeMutablePointer<Float>?, UnsafePointer<__LAPACK_int>, UnsafeMutablePointer<__LAPACK_int>) -> Void
// cgesv: (UnsafePointer<__LAPACK_int>, UnsafePointer<__LAPACK_int>, OpaquePointer?, UnsafePointer<__LAPACK_int>, UnsafeMutablePointer<__LAPACK_int>?, OpaquePointer?, UnsafePointer<__LAPACK_int>, UnsafeMutablePointer<__LAPACK_int>) -> Void
// spotrf: (UnsafePointer<CChar>, UnsafePointer<__LAPACK_int>, UnsafeMutablePointer<Float>?, UnsafePointer<__LAPACK_int>, UnsafeMutablePointer<__LAPACK_int>) -> Void
// cpotrf: (UnsafePointer<CChar>, UnsafePointer<__LAPACK_int>, OpaquePointer?, UnsafePointer<__LAPACK_int>, UnsafeMutablePointer<__LAPACK_int>) -> Void
// strsm: (UnsafePointer<CChar>, UnsafePointer<CChar>, UnsafePointer<CChar>, UnsafePointer<CChar>, UnsafePointer<__LAPACK_int>, UnsafePointer<__LAPACK_int>, UnsafePointer<Float>, UnsafePointer<Float>?, UnsafePointer<__LAPACK_int>, UnsafeMutablePointer<Float>?, UnsafePointer<__LAPACK_int>) -> Void
// ctrsm: (UnsafePointer<CChar>, UnsafePointer<CChar>, UnsafePointer<CChar>, UnsafePointer<CChar>, UnsafePointer<__LAPACK_int>, UnsafePointer<__LAPACK_int>, OpaquePointer, OpaquePointer?, UnsafePointer<__LAPACK_int>, OpaquePointer?, UnsafePointer<__LAPACK_int>) -> Void

//sgemm: (
//  UnsafePointer<CChar>, UnsafePointer<CChar>, UnsafePointer<__LAPACK_int>,
//  UnsafePointer<__LAPACK_int>, UnsafePointer<__LAPACK_int>, PointerSelf,
//  PointerSelf?, UnsafePointer<__LAPACK_int>, PointerSelf?,
//  UnsafePointer<__LAPACK_int>, PointerSelf, MutablePointerSelf?,
//  UnsafePointer<__LAPACK_int>) -> Void
//cgemm: (
//  UnsafePointer<CChar>, UnsafePointer<CChar>, UnsafePointer<__LAPACK_int>,
//  UnsafePointer<__LAPACK_int>, UnsafePointer<__LAPACK_int>, PointerSelf,
//  PointerSelf?, UnsafePointer<__LAPACK_int>, PointerSelf?,
//  UnsafePointer<__LAPACK_int>, PointerSelf, MutablePointerSelf?,
//  UnsafePointer<__LAPACK_int>) -> Void

//ssyev: (
//  UnsafePointer<CChar>, UnsafePointer<CChar>, UnsafePointer<__LAPACK_int>,
//  MutablePointerSelf?, UnsafePointer<__LAPACK_int>, MutablePointerReal?,
//  MutablePointerSelf, UnsafePointer<__LAPACK_int>, OptionalMutablePointerReal?,
//  UnsafeMutablePointer<__LAPACK_int>) -> Void
//cheev: (
//  UnsafePointer<CChar>, UnsafePointer<CChar>, UnsafePointer<__LAPACK_int>,
//  MutablePointerSelf?, UnsafePointer<__LAPACK_int>, MutablePointerReal?,
//  MutablePointerSelf, UnsafePointer<__LAPACK_int>, OptionalMutablePointerReal?,
//  UnsafeMutablePointer<__LAPACK_int>) -> Void

//sgesv: (
//  UnsafePointer<__LAPACK_int>, UnsafePointer<__LAPACK_int>, MutablePointerSelf,
//  UnsafePointer<__LAPACK_int>, UnsafeMutablePointer<__LAPACK_int>?,
//  MutablePointerSelf?, UnsafePointer<__LAPACK_int>,
//  UnsafeMutablePointer<__LAPACK_int>) -> Void
//cgesv: (
//  UnsafePointer<__LAPACK_int>, UnsafePointer<__LAPACK_int>, MutablePointerSelf?,
//  UnsafePointer<__LAPACK_int>, UnsafeMutablePointer<__LAPACK_int>?,
//  MutablePointerSelf?, UnsafePointer<__LAPACK_int>,
//  UnsafeMutablePointer<__LAPACK_int>) -> Void

//spotrf: (
//  UnsafePointer<CChar>, UnsafePointer<__LAPACK_int>, MutablePointerSelf?,
//  UnsafePointer<__LAPACK_int>, UnsafeMutablePointer<__LAPACK_int>) -> Void
//cpotrf: (
//  UnsafePointer<CChar>, UnsafePointer<__LAPACK_int>, MutablePointerSelf?,
//  UnsafePointer<__LAPACK_int>, UnsafeMutablePointer<__LAPACK_int>) -> Void

//strsm: (
//  UnsafePointer<CChar>, UnsafePointer<CChar>, UnsafePointer<CChar>,
//  UnsafePointer<CChar>, UnsafePointer<__LAPACK_int>,
//  UnsafePointer<__LAPACK_int>, PointerSelf, PointerSelf?,
//  UnsafePointer<__LAPACK_int>, MutablePointerSelf?,
//  UnsafePointer<__LAPACK_int>) -> Void
//ctrsm: (
//  UnsafePointer<CChar>, UnsafePointer<CChar>, UnsafePointer<CChar>,
//  UnsafePointer<CChar>, UnsafePointer<__LAPACK_int>,
//  UnsafePointer<__LAPACK_int>, PointerSelf, PointerSelf?,
//  UnsafePointer<__LAPACK_int>, MutablePointerSelf?,
//  UnsafePointer<__LAPACK_int>) -> Void

protocol BLASLAPACKScalar {
  associatedtype PointerSelf
  associatedtype MutablePointerSelf
  associatedtype MutablePointerReal
}

extension Float: BLASLAPACKScalar {
  typealias PointerSelf = UnsafePointer<Float>
  typealias MutablePointerSelf = UnsafeMutablePointer<Float>
  typealias MutablePointerReal = UnsafeMutablePointer<Float>
}

extension Double: BLASLAPACKScalar {
  typealias PointerSelf = UnsafePointer<Double>
  typealias MutablePointerSelf = UnsafeMutablePointer<Double>
  typealias MutablePointerReal = UnsafeMutablePointer<Double>
}

extension Complex<Double>: BLASLAPACKScalar {
  typealias PointerSelf = OpaquePointer
  typealias MutablePointerSelf = OpaquePointer
  typealias MutablePointerReal = UnsafeMutablePointer<Double>
}

// define a struct that contains closures for BLAS/LAPACK functions
struct BLASLAPACKFunctions<T: BLASLAPACKScalar> {
  // define the function types for each BLAS/LAPACK function
  typealias GEMMFunction = (
    UnsafePointer<CChar>, UnsafePointer<CChar>, UnsafePointer<__LAPACK_int>,
    UnsafePointer<__LAPACK_int>, UnsafePointer<__LAPACK_int>, T.PointerSelf,
    T.PointerSelf?, UnsafePointer<__LAPACK_int>, T.PointerSelf?,
    UnsafePointer<__LAPACK_int>, T.PointerSelf, T.MutablePointerSelf?,
    UnsafePointer<__LAPACK_int>) -> Void
  
  typealias SYEVFunction = (
    UnsafePointer<CChar>, UnsafePointer<CChar>, UnsafePointer<__LAPACK_int>,
    T.MutablePointerSelf?, UnsafePointer<__LAPACK_int>, T.MutablePointerReal?,
    T.MutablePointerSelf, UnsafePointer<__LAPACK_int>, T.MutablePointerReal?,
    UnsafeMutablePointer<__LAPACK_int>) -> Void
  
  typealias GESVFunction = (
    UnsafePointer<__LAPACK_int>, UnsafePointer<__LAPACK_int>,
    T.MutablePointerSelf, UnsafePointer<__LAPACK_int>,
    UnsafeMutablePointer<__LAPACK_int>?, T.MutablePointerSelf?,
    UnsafePointer<__LAPACK_int>, UnsafeMutablePointer<__LAPACK_int>) -> Void
  
  typealias POTRFFunction = (
    UnsafePointer<CChar>, UnsafePointer<__LAPACK_int>, T.MutablePointerSelf?,
    UnsafePointer<__LAPACK_int>, UnsafeMutablePointer<__LAPACK_int>) -> Void
  
  typealias TRSMFunction = (
    UnsafePointer<CChar>, UnsafePointer<CChar>, UnsafePointer<CChar>,
    UnsafePointer<CChar>, UnsafePointer<__LAPACK_int>,
    UnsafePointer<__LAPACK_int>, T.PointerSelf, T.PointerSelf?,
    UnsafePointer<__LAPACK_int>, T.MutablePointerSelf?,
    UnsafePointer<__LAPACK_int>) -> Void
  
  // define the properties that store the closures for each BLAS/LAPACK function
  let gemm: GEMMFunction
  let syev: SYEVFunction
  let gesv: GESVFunction
  let potrf: POTRFFunction
  let trsm: TRSMFunction
  
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

@inline(__always)
func wrap_syev<T: BLASLAPACKScalar>(
  type: T.Type,
  _ syev: @escaping (
    UnsafePointer<CChar>, UnsafePointer<CChar>, UnsafePointer<__LAPACK_int>,
    T.MutablePointerSelf?, UnsafePointer<__LAPACK_int>, T.MutablePointerReal?,
    T.MutablePointerSelf, UnsafePointer<__LAPACK_int>,
    UnsafeMutablePointer<__LAPACK_int>
  ) -> Void
) -> (
  UnsafePointer<CChar>, UnsafePointer<CChar>, UnsafePointer<__LAPACK_int>,
  T.MutablePointerSelf?, UnsafePointer<__LAPACK_int>, T.MutablePointerReal?,
  T.MutablePointerSelf, UnsafePointer<__LAPACK_int>, T.MutablePointerReal?,
  UnsafeMutablePointer<__LAPACK_int>
) -> Void {
  return {
    return syev($0, $1, $2, $3, $4, $5, $6, $7, $9)
  } as (
    UnsafePointer<CChar>, UnsafePointer<CChar>, UnsafePointer<__LAPACK_int>,
    T.MutablePointerSelf?, UnsafePointer<__LAPACK_int>, T.MutablePointerReal?,
    T.MutablePointerSelf, UnsafePointer<__LAPACK_int>, T.MutablePointerReal?,
    UnsafeMutablePointer<__LAPACK_int>
  ) -> Void
}

// extend Float to store a static variable that returns the set of BLAS/LAPACK functions for Float
extension Float {
  static var blaslapackFunctions: BLASLAPACKFunctions<Float> {
    return BLASLAPACKFunctions<Float>(
      gemm: sgemm_,
      syev: wrap_syev(type: Float.self, ssyev_2stage_),
      gesv: sgesv_,
      potrf: spotrf_,
      trsm: strsm_)
  }
}

// extend Double to store a static variable that returns the set of BLAS/LAPACK functions for Double
extension Double {
  static var blaslapackFunctions: BLASLAPACKFunctions<Double> {
    return BLASLAPACKFunctions<Double>(
      gemm: dgemm_,
      syev: wrap_syev(type: Double.self, dsyev_2stage_),
      gesv: dgesv_,
      potrf: dpotrf_,
      trsm: dtrsm_)
  }
}

// extend Complex<Double> to store a static variable that returns the set of BLAS/LAPACK functions for Complex<Double>
extension Complex where RealType == Double {
  static var blaslapackFunctions: BLASLAPACKFunctions<Complex<Double>> {
    return BLASLAPACKFunctions<Complex<Double>>(
      gemm: zgemm_,
      syev: zheev_2stage_,
      gesv: zgesv_,
      potrf: zpotrf_,
      trsm: ztrsm_)
  }
}
