//
//  Helpers.swift
//  AMXBenchmarks
//
//  Created by Philip Turner on 3/24/23.
//

import Foundation
import Accelerate
import RealModule
import ComplexModule
#if os(macOS)
// import the PythonKit framework to access the PythonObject type
import PythonKit
fileprivate let np = Python.import("numpy")
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

protocol LinearAlgebraScalar: Numeric {
  associatedtype PointerSelf
  associatedtype MutablePointerSelf
  associatedtype MutablePointerReal
  
  static var linearAlgebraFunctions: LinearAlgebraFunctions<Self> { get }
  
  static var one: Self { get }
}

extension Float: LinearAlgebraScalar {
  typealias PointerSelf = UnsafePointer<Float>
  typealias MutablePointerSelf = UnsafeMutablePointer<Float>
  typealias MutablePointerReal = UnsafeMutablePointer<Float>
  
  static let one: Float = 1
}

extension Double: LinearAlgebraScalar {
  typealias PointerSelf = UnsafePointer<Double>
  typealias MutablePointerSelf = UnsafeMutablePointer<Double>
  typealias MutablePointerReal = UnsafeMutablePointer<Double>
  
  static let one: Double = 1
}

extension Complex<Double>: LinearAlgebraScalar {
  typealias PointerSelf = OpaquePointer
  typealias MutablePointerSelf = OpaquePointer
  typealias MutablePointerReal = UnsafeMutablePointer<Double>
  
  static let one: Complex<Double> = .init(1, 0)
}

// define a struct that contains closures for BLAS/LAPACK functions
struct LinearAlgebraFunctions<T: LinearAlgebraScalar> {
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
func wrap_syev<T: LinearAlgebraScalar>(
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
  static var linearAlgebraFunctions: LinearAlgebraFunctions<Float> {
    return LinearAlgebraFunctions<Float>(
      gemm: sgemm_,
      syev: wrap_syev(type: Float.self, ssyev_), // try ssyevd_, _2stage
      gesv: sgesv_,
      potrf: spotrf_,
      trsm: strsm_)
  }
}

// extend Double to store a static variable that returns the set of BLAS/LAPACK functions for Double
extension Double {
  static var linearAlgebraFunctions: LinearAlgebraFunctions<Double> {
    return LinearAlgebraFunctions<Double>(
      gemm: dgemm_,
      syev: wrap_syev(type: Double.self, dsyev_), // try dsyevd_, _2stage
      gesv: dgesv_,
      potrf: dpotrf_,
      trsm: dtrsm_)
  }
}

// extend Complex<Double> to store a static variable that returns the set of BLAS/LAPACK functions for Complex<Double>
extension Complex where RealType == Double {
  static var linearAlgebraFunctions: LinearAlgebraFunctions<Complex<Double>> {
    return LinearAlgebraFunctions<Complex<Double>>(
      gemm: zgemm_,
      syev: zheev_, // try zheevd_, _2stage
      gesv: zgesv_,
      potrf: zpotrf_,
      trsm: ztrsm_)
  }
}

struct Matrix<T: LinearAlgebraScalar> {
  // the internal 1D array
  private var storage: [T]
  
  // the dimension of the square matrix
  let dimension: Int
  
  // initialize the matrix with a given dimension and a default value
  init(dimension: Int, defaultValue: T) {
    self.dimension = dimension
    self.storage = Array(repeating: defaultValue, count: dimension * dimension)
  }
  
  // get or set the element at the given row and column indices
  subscript(row: Int, column: Int) -> T {
    get {
      precondition(row >= 0 && row < dimension, "Row index out of range")
      precondition(column >= 0 && column < dimension, "Column index out of range")
      return storage[row * dimension + column]
    }
    set {
      precondition(row >= 0 && row < dimension, "Row index out of range")
      precondition(column >= 0 && column < dimension, "Column index out of range")
      storage[row * dimension + column] = newValue
    }
  }
}


#if os(macOS)
// define a struct that represents a matrix using a PythonObject
struct PythonMatrix<T: LinearAlgebraScalar & PythonConvertible & ConvertibleFromPython> {
  // the internal PythonObject that stores the matrix data
  private var storage: PythonObject
  
  // the dimension of the square matrix
  let dimension: Int
  
  // initialize the matrix with a given dimension and a default value
  init(dimension: Int, defaultValue: T) {
    self.dimension = dimension
    // create a NumPy array with the default value and reshape it to a square matrix
    self.storage = np.full(dimension * dimension, defaultValue)
      .reshape(dimension, dimension)
    
    if T.self == Float.self {
      storage = storage.astype("float32")
    } else if T.self == Complex<Double>.self {
      storage = storage.astype("complex128")
    }
  }
  
  // get or set the element at the given row and column indices
  subscript(row: Int, column: Int) -> T {
    get {
      precondition(row >= 0 && row < dimension, "Row index out of range")
      precondition(column >= 0 && column < dimension, "Column index out of range")
      // use the subscript operator of the PythonObject to access the element
      return T(storage[row, column])!
    }
    set {
      precondition(row >= 0 && row < dimension, "Row index out of range")
      precondition(column >= 0 && column < dimension, "Column index out of range")
      // use the subscript operator of the PythonObject to modify the element
      storage[row, column] = newValue.pythonObject
    }
  }
}

extension Complex<Double>: PythonConvertible & ConvertibleFromPython {
  public init?(_ object: PythonKit.PythonObject) {
    guard let py_real = object.checking.real,
          let py_imaginary = object.checking.imag,
          let real = Double(py_real),
          let imaginary = Double(py_imaginary) else {
      return nil
    }
    self.init(real, imaginary)
  }
  
  public init?(pythonObject: PythonKit.PythonObject) {
    self.init(pythonObject)
  }
  
  public var pythonObject: PythonObject {
    Python.complex(self.real, self.imaginary)
  }
}
#endif

// define a protocol that requires instance methods for matrix operations
protocol MatrixOperations: CustomStringConvertible {
  // define the associated type for the scalar element
  associatedtype Scalar: LinearAlgebraScalar
  
  // define the instance methods for matrix operations
  // use descriptive names that indicate the corresponding BLAS/LAPACK function
  // use inout parameters for the return values
  func matrixMultiply(by other: Self, into result: inout Self) // corresponds to GEMM
  func eigenDecomposition(into values: inout [Scalar], vectors: inout Self) // corresponds to SYEV
  func solveLinearSystem(with rhs: Self, into solution: inout Self) // corresponds to GESV
  func choleskyFactorization(into factor: inout Self) // corresponds to POTRF
  func triangularSolve(with rhs: Self, into solution: inout Self) // corresponds to TRSM
  
  var dimension: Int { get }
  
  // "Matrix \(dataType) \(library)"
  var description: String { get }
}

struct AnyMatrixOperations {
  var value: any MatrixOperations
  
  init(_ value: any MatrixOperations) {
    self.value = value
  }
  
  func matrixMultiply(by other: any MatrixOperations, into result: inout any MatrixOperations) {
    func bypassTypeChecking<T: MatrixOperations>(result: inout T) {
      (value as! T).matrixMultiply(by: other as! T, into: &result)
    }
    bypassTypeChecking(result: &result)
  }
}

func checkDimensions<
  T: MatrixOperations, U: MatrixOperations
>(
  _ lhs: T, _ rhs: U
) {
  precondition(
    lhs.dimension == rhs.dimension,
    "Incompatible dimensions: lhs (\(lhs.dimension)) != rhs (\(rhs.dimension))")
}

func checkDimensions<
  T: MatrixOperations, U: MatrixOperations, V: MatrixOperations
>(
  _ lhs: T, _ mhs: U, _ rhs: V
) {
  checkDimensions(lhs, mhs)
  checkDimensions(lhs, rhs)
}

// extend Matrix to conform to MatrixOperations protocol
extension Matrix: MatrixOperations {
  var description: String {
    if Scalar.self == Complex<Double>.self {
      return "Matrix Complex Accelerate"
    } else {
      return "Matrix \(Scalar.self) Accelerate"
    }
  }
  
  func matrixMultiply(by other: Matrix<T>, into result: inout Matrix<T>) {
    // implement matrix multiplication using BLAS/LAPACK functions
    // store the result in the inout parameter
    checkDimensions(self, other, result)
    
    var dim: Int = self.dimension
    let alpha = T.one
    let beta = T.zero
    self.storage.withUnsafeBufferPointer { pointerA in
      let A = unsafeBitCast(pointerA.baseAddress, to: T.PointerSelf?.self)
      
      other.storage.withUnsafeBufferPointer { pointerB in
        let B = unsafeBitCast(pointerB.baseAddress, to: T.PointerSelf?.self)
        
        result.storage.withUnsafeMutableBufferPointer { pointerC in
          let C = unsafeBitCast(pointerC.baseAddress, to: T.MutablePointerSelf?.self)
          
          withUnsafePointer(to: alpha) { pointerAlpha in
            let alpha = unsafeBitCast(pointerAlpha, to: T.PointerSelf.self)
            
            withUnsafePointer(to: beta) { pointerBeta in
              let beta = unsafeBitCast(pointerBeta, to: T.PointerSelf.self)
              
              Scalar.linearAlgebraFunctions.gemm(
                "N", "N", &dim, &dim, &dim, alpha, A, &dim,
                B, &dim, beta, C, &dim)
            }
          }
        }
      }
    }
    
  }
  
  func eigenDecomposition(into values: inout [T], vectors: inout Matrix<T>) {
    // implement eigenvalue decomposition using BLAS/LAPACK functions
    // store the values and vectors in the inout parameters
  }
  
  func solveLinearSystem(with rhs: Matrix<T>, into solution: inout Matrix<T>) {
    // implement linear system solver using BLAS/LAPACK functions
    // store the solution in the inout parameter
  }
  
  func choleskyFactorization(into factor: inout Matrix<T>) {
    // implement Cholesky factorization using BLAS/LAPACK functions
    // store the factor in the inout parameter
  }
  
  func triangularSolve(with rhs: Matrix<T>, into solution: inout Matrix<T>) {
    // implement triangular solver using BLAS/LAPACK functions
    // store the solution in the inout parameter
  }
}

#if os(macOS)
// extend PythonMatrix to conform to MatrixOperations protocol
extension PythonMatrix: MatrixOperations {
  var description: String {
    if Scalar.self == Complex<Double>.self {
      return "Matrix Complex OpenBLAS"
    } else {
      return "Matrix \(Scalar.self) OpenBLAS"
    }
  }
  
  func matrixMultiply(by other: PythonMatrix<T>, into result: inout PythonMatrix<T>) {
    // implement matrix multiplication using NumPy functions
    // store the result in the inout parameter
    checkDimensions(self, other, result)
    
    np.matmul(self.storage, other.storage, out: result.storage)
  }
  
  func eigenDecomposition(into values: inout [T], vectors: inout PythonMatrix<T>) {
    // implement eigenvalue decomposition using NumPy functions
    // store the values and vectors in the inout parameters
  }
  
  func solveLinearSystem(with rhs: PythonMatrix<T>, into solution: inout PythonMatrix<T>) {
    // implement linear system solver using NumPy functions
    // store the solution in the inout parameter
  }
  
  func choleskyFactorization(into factor: inout PythonMatrix<T>) {
    // implement Cholesky factorization using NumPy functions
    // store the factor in the inout parameter
  }
  
  func triangularSolve(with rhs: PythonMatrix<T>, into solution: inout PythonMatrix<T>) {
    // implement triangular solver using NumPy functions
    // store the solution in the inout parameter
  }
}
#endif
