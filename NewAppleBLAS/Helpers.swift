#if true

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

fileprivate let scratch1 = malloc(256 * 1024 * 1024)!
fileprivate let scratch2 = malloc(256 * 1024 * 1024)!
fileprivate let scratch3 = malloc(256 * 1024 * 1024)!
fileprivate let scratch4 = malloc(256 * 1024 * 1024)!
fileprivate let scratch5 = malloc(256 * 1024 * 1024)!
fileprivate let scratch6 = malloc(256 * 1024 * 1024)!

protocol LinearAlgebraScalar: Numeric {
  associatedtype PointerSelf
  associatedtype MutablePointerSelf
  associatedtype RealType: LinearAlgebraScalar
  
  static var linearAlgebraFunctions: LinearAlgebraFunctions<Self> { get }
  
  static var one: Self { get }
}

#if os(macOS)
protocol PythonLinearAlgebraScalar: LinearAlgebraScalar & PythonConvertible & ConvertibleFromPython where RealType: PythonLinearAlgebraScalar {
  
}

extension Float: PythonLinearAlgebraScalar {}
extension Double: PythonLinearAlgebraScalar {}
extension Complex<Double>: PythonLinearAlgebraScalar {}
#endif

extension LinearAlgebraScalar {
  typealias MutablePointerReal = UnsafeMutablePointer<RealType>
  typealias PointerReal = UnsafePointer<RealType>
}

extension Float: LinearAlgebraScalar {
  typealias PointerSelf = UnsafePointer<Float>
  typealias MutablePointerSelf = UnsafeMutablePointer<Float>
  typealias RealType = Float
  
  static let one: Float = 1
}

extension Double: LinearAlgebraScalar {
  typealias PointerSelf = UnsafePointer<Double>
  typealias MutablePointerSelf = UnsafeMutablePointer<Double>
  typealias RealType = Double
  
  static let one: Double = 1
}

extension Complex<Double>: LinearAlgebraScalar {
  typealias PointerSelf = OpaquePointer
  typealias MutablePointerSelf = OpaquePointer
  typealias RealType = Double
  
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
    _ JOBZ: UnsafePointer<CChar>,
    _ UPLO: UnsafePointer<CChar>,
    _ N: UnsafePointer<__LAPACK_int>,
    _ A: T.MutablePointerSelf?,
    _ LDA: UnsafePointer<__LAPACK_int>,
    _ W: T.MutablePointerReal?,
    _ WORK: T.MutablePointerSelf,
    _ LWORK: UnsafePointer<__LAPACK_int>,
    _ RWORK: T.MutablePointerReal?,
    _ INFO: UnsafeMutablePointer<__LAPACK_int>) -> Void
  
  typealias SYEVDFunction = (
    _ JOBZ: UnsafePointer<CChar>,
    _ UPLO: UnsafePointer<CChar>,
    _ N: UnsafePointer<__LAPACK_int>,
    _ A: T.MutablePointerSelf?,
    _ LDA: UnsafePointer<__LAPACK_int>,
    _ W: T.MutablePointerReal?,
    _ WORK: T.MutablePointerSelf,
    _ LWORK: UnsafePointer<__LAPACK_int>,
    _ RWORK: T.MutablePointerReal?,
    _ LRWORK: UnsafePointer<__LAPACK_int>,
    _ IWORK: UnsafeMutablePointer<Int>?,
    _ LIWORK: UnsafePointer<__LAPACK_int>,
    _ INFO: UnsafeMutablePointer<__LAPACK_int>) -> Void
  
  typealias SYEVRFunction = (
    _ JOBZ: UnsafePointer<CChar>,
    _ RANGE: UnsafePointer<CChar>,
    _ UPLO: UnsafePointer<CChar>,
    _ N: UnsafePointer<__LAPACK_int>,
    _ A: T.MutablePointerSelf?,
    _ LDA: UnsafePointer<__LAPACK_int>,
    _ VL: T.PointerReal,
    _ VU: T.PointerReal,
    _ IL: UnsafePointer<__LAPACK_int>,
    _ IU: UnsafePointer<__LAPACK_int>,
    _ ABSTOL: T.PointerReal,
    _ M: UnsafeMutablePointer<__LAPACK_int>,
    _ W: T.MutablePointerReal?,
    _ Z: T.MutablePointerSelf,
    _ LDZ: UnsafePointer<__LAPACK_int>,
    _ ISUPPZ: UnsafeMutablePointer<__LAPACK_int>?,
    _ WORK: T.MutablePointerSelf,
    _ LWORK: UnsafePointer<__LAPACK_int>,
    _ RWORK: T.MutablePointerReal?,
    _ LRWORK: UnsafePointer<__LAPACK_int>,
    _ IWORK: UnsafeMutablePointer<Int>?,
    _ LIWORK: UnsafePointer<__LAPACK_int>,
    _ INFO: UnsafeMutablePointer<__LAPACK_int>) -> Void
  
  typealias POTRFFunction = (
    UnsafePointer<CChar>, UnsafePointer<__LAPACK_int>, T.MutablePointerSelf?,
    UnsafePointer<__LAPACK_int>, UnsafeMutablePointer<__LAPACK_int>) -> Void
  
  typealias TRSMFunction = (
    UnsafePointer<CChar>, UnsafePointer<CChar>, UnsafePointer<CChar>,
    UnsafePointer<CChar>, UnsafePointer<__LAPACK_int>,
    UnsafePointer<__LAPACK_int>, T.PointerSelf, T.PointerSelf?,
    UnsafePointer<__LAPACK_int>, T.MutablePointerSelf?,
    UnsafePointer<__LAPACK_int>) -> Void
  
  typealias TRTTPFunction = (
    UnsafePointer<CChar>, UnsafePointer<__LAPACK_int>, T.PointerSelf?,
    UnsafePointer<__LAPACK_int>, T.MutablePointerSelf?,
    UnsafeMutablePointer<__LAPACK_int>) -> Void
  
  // define the properties that store the closures for each BLAS/LAPACK function
  let gemm: GEMMFunction
  
  let syev: SYEVFunction
  let syevd: SYEVDFunction
  let syevr: SYEVRFunction
  
  let potrf: POTRFFunction
  let trsm: TRSMFunction
  let trttp: TRTTPFunction
  
  // initialize the struct with the closures for each BLAS/LAPACK function
  init(gemm: @escaping GEMMFunction,
       syev: @escaping SYEVFunction,
       syevd: @escaping SYEVDFunction,
       syevr: @escaping SYEVRFunction,
       potrf: @escaping POTRFFunction,
       trsm: @escaping TRSMFunction,
       trttp: @escaping TRTTPFunction) {
    self.gemm = gemm
    
    self.syev = syev
    self.syevd = syevd
    self.syevr = syevr
    
    self.potrf = potrf
    self.trsm = trsm
    self.trttp = trttp
  }
}

@inline(__always)
func wrap_syev<T: LinearAlgebraScalar>(
  type: T.Type,
  _ syev: @escaping (
    _ JOBZ: UnsafePointer<CChar>,
    _ UPLO: UnsafePointer<CChar>,
    _ N: UnsafePointer<__LAPACK_int>,
    _ A: T.MutablePointerSelf?,
    _ LDA: UnsafePointer<__LAPACK_int>,
    _ W: T.MutablePointerReal?,
    _ WORK: T.MutablePointerSelf,
    _ LWORK: UnsafePointer<__LAPACK_int>,
    _ INFO: UnsafeMutablePointer<__LAPACK_int>
  ) -> Void
) -> LinearAlgebraFunctions<T>.SYEVFunction {
  return {
    return syev($0, $1, $2, $3, $4, $5, $6, $7, $9)
  } as LinearAlgebraFunctions<T>.SYEVFunction
}

@inline(__always)
func wrap_syevd<T: LinearAlgebraScalar>(
  type: T.Type,
  _ syev: @escaping (
    _ JOBZ: UnsafePointer<CChar>,
    _ UPLO: UnsafePointer<CChar>,
    _ N: UnsafePointer<__LAPACK_int>,
    _ A: T.MutablePointerSelf?,
    _ LDA: UnsafePointer<__LAPACK_int>,
    _ W: T.MutablePointerReal?,
    _ WORK: T.MutablePointerSelf,
    _ LWORK: UnsafePointer<__LAPACK_int>,
    _ IWORK: UnsafeMutablePointer<Int>?,
    _ LIWORK: UnsafePointer<__LAPACK_int>,
    _ INFO: UnsafeMutablePointer<__LAPACK_int>
  ) -> Void
) -> LinearAlgebraFunctions<T>.SYEVDFunction {
  return {
    return syev($0, $1, $2, $3, $4, $5, $6, $7, $10, $11, $12)
  } as LinearAlgebraFunctions<T>.SYEVDFunction
}

@inline(__always)
func wrap_syevr<T: LinearAlgebraScalar>(
  type: T.Type,
  _ syev: @escaping (
    _ JOBZ: UnsafePointer<CChar>,
    _ RANGE: UnsafePointer<CChar>,
    _ UPLO: UnsafePointer<CChar>,
    _ N: UnsafePointer<__LAPACK_int>,
    _ A: T.MutablePointerSelf?,
    _ LDA: UnsafePointer<__LAPACK_int>,
    _ VL: T.PointerReal,
    _ VU: T.PointerReal,
    _ IL: UnsafePointer<__LAPACK_int>,
    _ IU: UnsafePointer<__LAPACK_int>,
    _ ABSTOL: T.PointerReal,
    _ M: UnsafeMutablePointer<__LAPACK_int>,
    _ W: T.MutablePointerReal?,
    _ Z: T.MutablePointerSelf,
    _ LDZ: UnsafePointer<__LAPACK_int>,
    _ ISUPPZ: UnsafeMutablePointer<__LAPACK_int>?,
    _ WORK: T.MutablePointerSelf,
    _ LWORK: UnsafePointer<__LAPACK_int>,
    _ IWORK: UnsafeMutablePointer<Int>?,
    _ LIWORK: UnsafePointer<__LAPACK_int>,
    _ INFO: UnsafeMutablePointer<__LAPACK_int>) -> Void
) -> LinearAlgebraFunctions<T>.SYEVRFunction {
  return {
    return syev($0, $1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11,
    $12, $13, $14, $15, $16, $17, $20, $21, $22)
  } as LinearAlgebraFunctions<T>.SYEVRFunction
}

// extend Float to store a static variable that returns the set of BLAS/LAPACK functions for Float
extension Float {
  static var linearAlgebraFunctions: LinearAlgebraFunctions<Float> {
    return LinearAlgebraFunctions<Float>(
      gemm: sgemm_,
      syev: wrap_syev(type: Float.self, ssyev_),
      syevd: wrap_syevd(type: Float.self, ssyevd_),
      syevr: wrap_syevr(type: Float.self, ssyevr_),
      potrf: spotrf_,
      trsm: strsm_,
      trttp: strttp_)
  }
}

// extend Double to store a static variable that returns the set of BLAS/LAPACK functions for Double
extension Double {
  static var linearAlgebraFunctions: LinearAlgebraFunctions<Double> {
    return LinearAlgebraFunctions<Double>(
      gemm: dgemm_,
      syev: wrap_syev(type: Double.self, dsyev_),
      syevd: wrap_syevd(type: Double.self, dsyevd_),
      syevr: wrap_syevr(type: Double.self, dsyevr_),
      potrf: dpotrf_,
      trsm: dtrsm_,
      trttp: dtrttp_)
  }
}

// extend Complex<Double> to store a static variable that returns the set of BLAS/LAPACK functions for Complex<Double>
extension Complex where RealType == Double {
  static var linearAlgebraFunctions: LinearAlgebraFunctions<Complex<Double>> {
    return LinearAlgebraFunctions<Complex<Double>>(
      gemm: zgemm_,
      syev: zheev_,
      syevd: zheevd_,
      syevr: zheevr_,
      potrf: zpotrf_,
      trsm: ztrsm_,
      trttp: ztrttp_)
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

// define a struct that represents a vector using a 1D array
struct Vector<T: LinearAlgebraScalar> {
  // the internal 1D array that stores the vector data
  var storage: [T]
  
  // the dimension of the vector
  let dimension: Int
  
  // initialize the vector with a given dimension and a default value
  init(dimension: Int, defaultValue: T) {
    self.dimension = dimension
    self.storage = Array(repeating: defaultValue, count: dimension)
  }
  
  // get or set the element at the given index
  subscript(index: Int) -> T {
    get {
      precondition(index >= 0 && index < dimension, "Index out of range")
      return storage[index]
    }
    set {
      precondition(index >= 0 && index < dimension, "Index out of range")
      storage[index] = newValue
    }
  }
}

#if os(macOS)
//typealias PythonLinearAlgebraScalar =
//  LinearAlgebraScalar & PythonConvertible & ConvertibleFromPython

// define a struct that represents a matrix using a PythonObject
struct PythonMatrix<T: PythonLinearAlgebraScalar> {
  // the internal PythonObject that stores the matrix data
  var storage: PythonObject
  
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

// define a struct that represents a vector using a PythonObject
struct PythonVector<T: PythonLinearAlgebraScalar> {
  // the internal PythonObject that stores the vector data
  var storage: PythonObject
  
  // the dimension of the vector
  let dimension: Int
  
  // initialize the vector with a given dimension and a default value
  init(dimension: Int, defaultValue: T) {
    self.dimension = dimension
    // create a NumPy array with the default value and reshape it to a vector
    let np = Python.import("numpy")
    self.storage = np.full(dimension, defaultValue).reshape(dimension)
  }
  
  // get or set the element at the given index
  subscript(index: Int) -> T {
    get {
      precondition(index >= 0 && index < dimension, "Index out of range")
      // use the subscript operator of the PythonObject to access the element
      return T(storage[index])!
    }
    set {
      precondition(index >= 0 && index < dimension, "Index out of range")
      // use the subscript operator of the PythonObject to modify the element
      storage[index] = newValue.pythonObject
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
  
  // define the associated type for the vector element
  associatedtype RealVector: VectorOperations where RealVector.Scalar == Scalar.RealType
  
  // define the instance methods for matrix operations
  // use descriptive names that indicate the corresponding BLAS/LAPACK function
  // use inout parameters for the return values
  func matrixMultiply(by other: Self, into result: inout Self) // corresponds to GEMM
  func eigenDecomposition(into values: inout RealVector, vectors: inout Self) // corresponds to SYEV
  func solveLinearSystem(with rhs: Self, into solution: inout Self) // corresponds to GESV
  func choleskyFactorization(into factor: inout Self) // corresponds to POTRF
  func triangularSolve(with rhs: Self, into solution: inout Self) // corresponds to TRSM
  
  var dimension: Int { get }
  
  // "Matrix \(dataType) \(library)"
  var description: String { get }
  
  subscript(row: Int, column: Int) -> Scalar { get set }
  
  init(dimension: Int, defaultValue: Scalar)
}

protocol VectorOperations: CustomStringConvertible {
  // define the associated type for the scalar element
  associatedtype Scalar: LinearAlgebraScalar
  
  var dimension: Int { get }
  
  // "Matrix \(dataType) \(library)"
  var description: String { get }
  
  init(dimension: Int, defaultValue: Scalar)
  
  subscript(index: Int) -> Scalar { get set }
}

struct AnyMatrixOperations {
  var value: any MatrixOperations
  
  var dimension: Int { value.dimension }
  
  init(_ value: any MatrixOperations) {
    self.value = value
  }
  
  func makeRealVector() -> any VectorOperations {
    func bypassTypeChecking<T: MatrixOperations>(value: T) -> any VectorOperations {
      return T.RealVector(dimension: value.dimension, defaultValue: .zero)
    }
    return bypassTypeChecking(value: value)
  }
  
  func makeMatrix() -> any MatrixOperations {
    func bypassTypeChecking<T: MatrixOperations>(value: T) -> any MatrixOperations {
      return T(dimension: value.dimension, defaultValue: .zero)
    }
    return bypassTypeChecking(value: value)
  }
  
  func matrixMultiply(by other: any MatrixOperations, into result: inout any MatrixOperations) {
    func bypassTypeChecking<T: MatrixOperations>(result: inout T) {
      (value as! T).matrixMultiply(by: other as! T, into: &result)
    }
    bypassTypeChecking(result: &result)
  }
  
  func eigenDecomposition(into values: inout any VectorOperations, vectors: inout any MatrixOperations) {
    func bypassTypeChecking<T: MatrixOperations>(vectors: inout T) {
      var valuesCopy = values as! T.RealVector
      (value as! T).eigenDecomposition(into: &valuesCopy, vectors: &vectors)
      values = valuesCopy
    }
    bypassTypeChecking(vectors: &vectors)
  }
  
  
  subscript(row: Int, column: Int) -> any LinearAlgebraScalar {
    get {
      (value[row, column] as any LinearAlgebraScalar)
    }
    set {
      func bypassTypeChecking<T: MatrixOperations>(value: inout T) {
        value[row, column] = newValue as! T.Scalar
      }
      bypassTypeChecking(value: &value)
    }
  }
}

struct AnyVectorOperations {
  var value: any VectorOperations
  
  var dimension: Int { value.dimension }
  
  init(_ value: any VectorOperations) {
    self.value = value
  }
  
  subscript(index: Int) -> any LinearAlgebraScalar {
    get {
      (value[index] as any LinearAlgebraScalar)
    }
    set {
      func bypassTypeChecking<T: VectorOperations>(value: inout T) {
        value[index] = newValue as! T.Scalar
      }
      bypassTypeChecking(value: &value)
    }
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

func checkDimensions<
  T: MatrixOperations, U: VectorOperations, V: MatrixOperations
>(
  _ lhs: T, _ mhs: U, _ rhs: V
) {
  precondition(
    lhs.dimension == mhs.dimension,
    "Incompatible dimensions: lhs (\(lhs.dimension)) != rhs (\(mhs.dimension))")
  precondition(
    lhs.dimension == rhs.dimension,
    "Incompatible dimensions: lhs (\(lhs.dimension)) != rhs (\(rhs.dimension))")
}

func checkLAPACKError(
  _ error: Int,
  _ file: StaticString = #file,
  _ line: UInt = #line
) {
  if _slowPath(error != 0) {
    let message = """
      Found LAPACK error in \(file):\(line):
      Error code = \(error)
      """
    print(message)
    fatalError(message, file: file, line: line)
  }
}

// extend Matrix to conform to MatrixOperations protocol
extension Matrix: MatrixOperations {
  typealias Scalar = T
  
  typealias RealVector = Vector<T.RealType>
  
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
  
  func eigenDecomposition(into values: inout RealVector, vectors: inout Matrix<T>) {
    // implement eigenvalue decomposition using BLAS/LAPACK functions
    // store the values and vectors in the inout parameters
    checkDimensions(self, values, vectors)
    
    // First copy the input to the output in packed format, then overwrite the
    // output with eigendecomposition.
    var dim: Int = self.dimension
    var error: Int = 0
//    var A_copied = UnsafeMutablePointer<T>.allocate(capacity: dim * dim)
//    defer { A_copied.deallocate() }
    self.storage.withUnsafeBufferPointer { pointerA in
      let A = unsafeBitCast(pointerA.baseAddress, to: T.PointerSelf?.self)
//      
//      memcpy(A_copied, unsafeBitCast(A, to: UnsafeMutableRawPointer.self), dim * dim * MemoryLayout<T>.stride)
      
      vectors.storage.withUnsafeMutableBufferPointer { pointerAP in
        let AP = unsafeBitCast(pointerAP.baseAddress, to: T.MutablePointerSelf?.self)
        
        memcpy(unsafeBitCast(AP, to: UnsafeMutableRawPointer.self), unsafeBitCast(A, to: UnsafeMutableRawPointer.self), dim * dim * MemoryLayout<T>.stride)
      }
    }
    
    var lworkSize: Int = -1
    var rworkSize: Int = -1
    var iworkSize: Int = -1
    
    precondition(dim == self.dimension)
    var lwork: [T] = [.zero]
    var rwork: [T.RealType] = [.zero]
    var iwork: [Int] = [.zero]
//    var isuppz: [Int] = Array(repeating: .zero, count: 2 * max(1, dim))
    
    var garbage_v: Scalar.RealType = .zero
    var garbage_i: Int = .zero
    var dim_copy = dim
    
    var abstol: T.RealType = .zero
//    var lamch_str = "S"
//    if T.self == Float.self {
//      abstol = slamch_(&lamch_str) as! T.RealType
//    } else {
//      abstol = dlamch_(&lamch_str) as! T.RealType
//    }
    
    precondition(dim == self.dimension)
    dim_copy = self.dimension
    
    values.storage.withUnsafeMutableBufferPointer { pointerW in
      let W = pointerW.baseAddress
      
      vectors.storage.withUnsafeMutableBufferPointer { pointerZ in
        let Z = unsafeBitCast(pointerZ.baseAddress, to: T.MutablePointerSelf?.self)
        
        lwork.withUnsafeMutableBufferPointer { pointerLWORK in
          let LWORK = unsafeBitCast(pointerLWORK.baseAddress, to: T.MutablePointerSelf.self)
          
          Scalar.linearAlgebraFunctions.syevd(
            "V",
            "U",
            &dim,
            nil,
            &dim,
            nil,
            LWORK,
            &lworkSize,
            &rwork,
            //
            &rworkSize,
            &iwork,
            &iworkSize,
            //
            &error)
          
          precondition(dim == self.dimension)
          precondition(dim_copy == self.dimension)
//          Scalar.linearAlgebraFunctions.syevr(
//            "V",
//            "A",
//            "U",
//            &dim,
//            unsafeBitCast(A_copied, to: T.MutablePointerSelf?.self),
//            &dim,
//            &garbage_v,
//            &garbage_v,
//            &garbage_i,
//            &garbage_i,
//            &abstol,
//            &dim_copy,
//            nil,
//            Z!,
//            &dim,
//            &isuppz,
//            LWORK,
//            &lworkSize,
//            &rwork,
//            //
//            &rworkSize,
//            &iwork,
//            &iworkSize,
//            //
//            &error)
          checkLAPACKError(error)
        }
        
        precondition(dim == self.dimension)
        precondition(dim_copy == self.dimension)
        
        if T.self == Float.self {
          lworkSize = Int(lwork[0] as! Float)
          rworkSize = Int(rwork[0] as! Float)
        } else if T.self == Double.self {
          lworkSize = Int(lwork[0] as! Double)
          rworkSize = Int(rwork[0] as! Double)
        } else if T.self == Complex<Double>.self {
          lworkSize = Int((lwork[0] as! Complex<Double>).real)
          rworkSize = Int(rwork[0] as! Double)
        }
        iworkSize = iwork[0]
//        rworkSize = max(1, 3 * dim - 2)
        
//        lwork = Array(repeating: .zero, count: lworkSize)
//        rwork = Array(repeating: .zero, count: rworkSize)
//        iwork = Array(repeating: .zero, count: iworkSize)
        
        lwork.withUnsafeMutableBufferPointer { pointerLWORK in
          let LWORK = unsafeBitCast(pointerLWORK.baseAddress, to: T.MutablePointerSelf.self)
          
//          rwork.withUnsafeMutableBufferPointer { RWORK in
//            iwork.withUnsafeMutableBufferPointer { IWORK in
              Scalar.linearAlgebraFunctions.syevd(
                "V",
                "U",
                &dim,
                Z,
                &dim,
                W,
                unsafeBitCast(scratch1, to: T.MutablePointerSelf.self),
                &lworkSize,
                unsafeBitCast(scratch2, to: UnsafeMutablePointer<T.RealType>.self),
                //
                &rworkSize,
                unsafeBitCast(scratch3, to: UnsafeMutablePointer<Int>.self),
                &iworkSize,
                //
                &error)
          
              precondition(dim == self.dimension)
              precondition(dim_copy == self.dimension)
//              Scalar.linearAlgebraFunctions.syevr(
//                "V",
//                "A",
//                "U",
//                &dim,
//                unsafeBitCast(A_copied, to: T.MutablePointerSelf?.self),
//                &dim,
//                &garbage_v,
//                &garbage_v,
//                &garbage_i,
//                &garbage_i,
//                &abstol,
//                &dim_copy,
//                W,
//                Z!,
//                &dim,
//                &isuppz,
//                LWORK,
//                &lworkSize,
//                &rwork,
//                //
//                &rworkSize,
//                &iwork,
//                &iworkSize,
//                //
//                &error)
              checkLAPACKError(error)
//            }
//          }
        }
      }
    }
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

extension Vector: VectorOperations {
  typealias Scalar = T
  
  var description: String {
    if Scalar.self == Complex<Double>.self {
      return "Vector Complex Accelerate"
    } else {
      return "Vector \(Scalar.self) Accelerate"
    }
  }
}

#if os(macOS)
// extend PythonMatrix to conform to MatrixOperations protocol
extension PythonMatrix: MatrixOperations {
  typealias Scalar = T
  
  typealias RealVector = PythonVector<T.RealType>
  
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
  
  func eigenDecomposition(into values: inout RealVector, vectors: inout PythonMatrix<T>) {
    // implement eigenvalue decomposition using NumPy functions
    // store the values and vectors in the inout parameters
    checkDimensions(self, values, vectors)
    
    let (a1, a2) = np.linalg.eigh(self.storage, UPLO: "U").tuple2
    np.copyto(values.storage, a1)
    np.copyto(vectors.storage, a2)
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

extension PythonVector: VectorOperations {
  typealias Scalar = T
  
  var description: String {
    if Scalar.self == Complex<Double>.self {
      return "Vector Complex OpenBLAS"
    } else {
      return "Vector \(Scalar.self) OpenBLAS"
    }
  }
}
#endif


#else

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

fileprivate let scratch1 = malloc(256 * 1024 * 1024)!
fileprivate let scratch2 = malloc(256 * 1024 * 1024)!
fileprivate let scratch3 = malloc(256 * 1024 * 1024)!
fileprivate let scratch4 = malloc(256 * 1024 * 1024)!
fileprivate let scratch5 = malloc(256 * 1024 * 1024)!
fileprivate let scratch6 = malloc(256 * 1024 * 1024)!

protocol LinearAlgebraScalar: Numeric {
  associatedtype PointerSelf
  associatedtype MutablePointerSelf
  associatedtype RealType: LinearAlgebraScalar
  
  static var linearAlgebraFunctions: LinearAlgebraFunctions<Self> { get }
  
  static var one: Self { get }
}

#if os(macOS)
protocol PythonLinearAlgebraScalar: LinearAlgebraScalar & PythonConvertible & ConvertibleFromPython where RealType: PythonLinearAlgebraScalar {
  
}

extension Float: PythonLinearAlgebraScalar {}
extension Double: PythonLinearAlgebraScalar {}
extension Complex<Double>: PythonLinearAlgebraScalar {}
#endif

extension LinearAlgebraScalar {
  typealias MutablePointerReal = UnsafeMutablePointer<RealType>
  typealias PointerReal = UnsafePointer<RealType>
}

extension Float: LinearAlgebraScalar {
  typealias PointerSelf = UnsafePointer<Float>
  typealias MutablePointerSelf = UnsafeMutablePointer<Float>
  typealias RealType = Float
  
  static let one: Float = 1
}

extension Double: LinearAlgebraScalar {
  typealias PointerSelf = UnsafePointer<Double>
  typealias MutablePointerSelf = UnsafeMutablePointer<Double>
  typealias RealType = Double
  
  static let one: Double = 1
}

extension Complex<Double>: LinearAlgebraScalar {
  typealias PointerSelf = OpaquePointer
  typealias MutablePointerSelf = OpaquePointer
  typealias RealType = Double
  
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
    _ JOBZ: UnsafePointer<CChar>,
    _ UPLO: UnsafePointer<CChar>,
    _ N: UnsafePointer<__LAPACK_int>,
    _ A: T.MutablePointerSelf?,
    _ LDA: UnsafePointer<__LAPACK_int>,
    _ W: T.MutablePointerReal?,
    _ WORK: T.MutablePointerSelf,
    _ LWORK: UnsafePointer<__LAPACK_int>,
    _ RWORK: T.MutablePointerReal?,
    _ INFO: UnsafeMutablePointer<__LAPACK_int>) -> Void
  
  typealias SYEVDFunction = (
    _ JOBZ: UnsafePointer<CChar>,
    _ UPLO: UnsafePointer<CChar>,
    _ N: UnsafePointer<__LAPACK_int>,
    _ A: T.MutablePointerSelf?,
    _ LDA: UnsafePointer<__LAPACK_int>,
    _ W: T.MutablePointerReal?,
    _ WORK: T.MutablePointerSelf,
    _ LWORK: UnsafePointer<__LAPACK_int>,
    _ RWORK: T.MutablePointerReal?,
    _ LRWORK: UnsafePointer<__LAPACK_int>,
    _ IWORK: UnsafeMutablePointer<Int>?,
    _ LIWORK: UnsafePointer<__LAPACK_int>,
    _ INFO: UnsafeMutablePointer<__LAPACK_int>) -> Void
  
  typealias SYEVRFunction = (
    _ JOBZ: UnsafePointer<CChar>,
    _ RANGE: UnsafePointer<CChar>,
    _ UPLO: UnsafePointer<CChar>,
    _ N: UnsafePointer<__LAPACK_int>,
    _ A: T.MutablePointerSelf?,
    _ LDA: UnsafePointer<__LAPACK_int>,
    _ VL: T.PointerReal,
    _ VU: T.PointerReal,
    _ IL: UnsafePointer<__LAPACK_int>,
    _ IU: UnsafePointer<__LAPACK_int>,
    _ ABSTOL: T.PointerReal,
    _ M: UnsafeMutablePointer<__LAPACK_int>,
    _ W: T.MutablePointerReal?,
    _ Z: T.MutablePointerSelf,
    _ LDZ: UnsafePointer<__LAPACK_int>,
    _ ISUPPZ: UnsafeMutablePointer<__LAPACK_int>?,
    _ WORK: T.MutablePointerSelf,
    _ LWORK: UnsafePointer<__LAPACK_int>,
    _ RWORK: T.MutablePointerReal?,
    _ LRWORK: UnsafePointer<__LAPACK_int>,
    _ IWORK: UnsafeMutablePointer<Int>?,
    _ LIWORK: UnsafePointer<__LAPACK_int>,
    _ INFO: UnsafeMutablePointer<__LAPACK_int>) -> Void
  
  typealias POTRFFunction = (
    UnsafePointer<CChar>, UnsafePointer<__LAPACK_int>, T.MutablePointerSelf?,
    UnsafePointer<__LAPACK_int>, UnsafeMutablePointer<__LAPACK_int>) -> Void
  
  typealias TRSMFunction = (
    UnsafePointer<CChar>, UnsafePointer<CChar>, UnsafePointer<CChar>,
    UnsafePointer<CChar>, UnsafePointer<__LAPACK_int>,
    UnsafePointer<__LAPACK_int>, T.PointerSelf, T.PointerSelf?,
    UnsafePointer<__LAPACK_int>, T.MutablePointerSelf?,
    UnsafePointer<__LAPACK_int>) -> Void
  
  typealias TRTTPFunction = (
    UnsafePointer<CChar>, UnsafePointer<__LAPACK_int>, T.PointerSelf?,
    UnsafePointer<__LAPACK_int>, T.MutablePointerSelf?,
    UnsafeMutablePointer<__LAPACK_int>) -> Void
  
  // define the properties that store the closures for each BLAS/LAPACK function
  let gemm: GEMMFunction
  
  let syev: SYEVFunction
  let syevd: SYEVDFunction
  let syevr: SYEVRFunction
  
  let potrf: POTRFFunction
  let trsm: TRSMFunction
  let trttp: TRTTPFunction
  
  // initialize the struct with the closures for each BLAS/LAPACK function
  init(gemm: @escaping GEMMFunction,
       syev: @escaping SYEVFunction,
       syevd: @escaping SYEVDFunction,
       syevr: @escaping SYEVRFunction,
       potrf: @escaping POTRFFunction,
       trsm: @escaping TRSMFunction,
       trttp: @escaping TRTTPFunction) {
    self.gemm = gemm
    
    self.syev = syev
    self.syevd = syevd
    self.syevr = syevr
    
    self.potrf = potrf
    self.trsm = trsm
    self.trttp = trttp
  }
}

@inline(__always)
func wrap_syev<T: LinearAlgebraScalar>(
  type: T.Type,
  _ syev: @escaping (
    _ JOBZ: UnsafePointer<CChar>,
    _ UPLO: UnsafePointer<CChar>,
    _ N: UnsafePointer<__LAPACK_int>,
    _ A: T.MutablePointerSelf?,
    _ LDA: UnsafePointer<__LAPACK_int>,
    _ W: T.MutablePointerReal?,
    _ WORK: T.MutablePointerSelf,
    _ LWORK: UnsafePointer<__LAPACK_int>,
    _ INFO: UnsafeMutablePointer<__LAPACK_int>
  ) -> Void
) -> LinearAlgebraFunctions<T>.SYEVFunction {
  return {
    return syev($0, $1, $2, $3, $4, $5, $6, $7, $9)
  } as LinearAlgebraFunctions<T>.SYEVFunction
}

@inline(__always)
func wrap_syevd<T: LinearAlgebraScalar>(
  type: T.Type,
  _ syev: @escaping (
    _ JOBZ: UnsafePointer<CChar>,
    _ UPLO: UnsafePointer<CChar>,
    _ N: UnsafePointer<__LAPACK_int>,
    _ A: T.MutablePointerSelf?,
    _ LDA: UnsafePointer<__LAPACK_int>,
    _ W: T.MutablePointerReal?,
    _ WORK: T.MutablePointerSelf,
    _ LWORK: UnsafePointer<__LAPACK_int>,
    _ IWORK: UnsafeMutablePointer<Int>?,
    _ LIWORK: UnsafePointer<__LAPACK_int>,
    _ INFO: UnsafeMutablePointer<__LAPACK_int>
  ) -> Void
) -> LinearAlgebraFunctions<T>.SYEVDFunction {
  return {
    return syev($0, $1, $2, $3, $4, $5, $6, $7, $10, $11, $12)
  } as LinearAlgebraFunctions<T>.SYEVDFunction
}

@inline(__always)
func wrap_syevr<T: LinearAlgebraScalar>(
  type: T.Type,
  _ syev: @escaping (
    _ JOBZ: UnsafePointer<CChar>,
    _ RANGE: UnsafePointer<CChar>,
    _ UPLO: UnsafePointer<CChar>,
    _ N: UnsafePointer<__LAPACK_int>,
    _ A: T.MutablePointerSelf?,
    _ LDA: UnsafePointer<__LAPACK_int>,
    _ VL: T.PointerReal,
    _ VU: T.PointerReal,
    _ IL: UnsafePointer<__LAPACK_int>,
    _ IU: UnsafePointer<__LAPACK_int>,
    _ ABSTOL: T.PointerReal,
    _ M: UnsafeMutablePointer<__LAPACK_int>,
    _ W: T.MutablePointerReal?,
    _ Z: T.MutablePointerSelf,
    _ LDZ: UnsafePointer<__LAPACK_int>,
    _ ISUPPZ: UnsafeMutablePointer<__LAPACK_int>?,
    _ WORK: T.MutablePointerSelf,
    _ LWORK: UnsafePointer<__LAPACK_int>,
    _ IWORK: UnsafeMutablePointer<Int>?,
    _ LIWORK: UnsafePointer<__LAPACK_int>,
    _ INFO: UnsafeMutablePointer<__LAPACK_int>) -> Void
) -> LinearAlgebraFunctions<T>.SYEVRFunction {
  return {
    return syev($0, $1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11,
    $12, $13, $14, $15, $16, $17, $20, $21, $22)
  } as LinearAlgebraFunctions<T>.SYEVRFunction
}

// extend Float to store a static variable that returns the set of BLAS/LAPACK functions for Float
extension Float {
  static var linearAlgebraFunctions: LinearAlgebraFunctions<Float> {
    return LinearAlgebraFunctions<Float>(
      gemm: sgemm_,
      syev: wrap_syev(type: Float.self, ssyev_),
      syevd: wrap_syevd(type: Float.self, ssyevd_),
      syevr: wrap_syevr(type: Float.self, ssyevr_),
      potrf: spotrf_,
      trsm: strsm_,
      trttp: strttp_)
  }
}

// extend Double to store a static variable that returns the set of BLAS/LAPACK functions for Double
extension Double {
  static var linearAlgebraFunctions: LinearAlgebraFunctions<Double> {
    return LinearAlgebraFunctions<Double>(
      gemm: dgemm_,
      syev: wrap_syev(type: Double.self, dsyev_),
      syevd: wrap_syevd(type: Double.self, dsyevd_),
      syevr: wrap_syevr(type: Double.self, dsyevr_),
      potrf: dpotrf_,
      trsm: dtrsm_,
      trttp: dtrttp_)
  }
}

// extend Complex<Double> to store a static variable that returns the set of BLAS/LAPACK functions for Complex<Double>
extension Complex where RealType == Double {
  static var linearAlgebraFunctions: LinearAlgebraFunctions<Complex<Double>> {
    return LinearAlgebraFunctions<Complex<Double>>(
      gemm: zgemm_,
      syev: zheev_,
      syevd: zheevd_,
      syevr: zheevr_,
      potrf: zpotrf_,
      trsm: ztrsm_,
      trttp: ztrttp_)
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

// define a struct that represents a vector using a 1D array
struct Vector<T: LinearAlgebraScalar> {
  // the internal 1D array that stores the vector data
  var storage: [T]
  
  // the dimension of the vector
  let dimension: Int
  
  // initialize the vector with a given dimension and a default value
  init(dimension: Int, defaultValue: T) {
    self.dimension = dimension
    self.storage = Array(repeating: defaultValue, count: dimension)
  }
  
  // get or set the element at the given index
  subscript(index: Int) -> T {
    get {
      precondition(index >= 0 && index < dimension, "Index out of range")
      return storage[index]
    }
    set {
      precondition(index >= 0 && index < dimension, "Index out of range")
      storage[index] = newValue
    }
  }
}

#if os(macOS)
//typealias PythonLinearAlgebraScalar =
//  LinearAlgebraScalar & PythonConvertible & ConvertibleFromPython

// define a struct that represents a matrix using a PythonObject
struct PythonMatrix<T: PythonLinearAlgebraScalar> {
  // the internal PythonObject that stores the matrix data
  var storage: PythonObject
  
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

// define a struct that represents a vector using a PythonObject
struct PythonVector<T: PythonLinearAlgebraScalar> {
  // the internal PythonObject that stores the vector data
  var storage: PythonObject
  
  // the dimension of the vector
  let dimension: Int
  
  // initialize the vector with a given dimension and a default value
  init(dimension: Int, defaultValue: T) {
    self.dimension = dimension
    // create a NumPy array with the default value and reshape it to a vector
    let np = Python.import("numpy")
    self.storage = np.full(dimension, defaultValue).reshape(dimension)
  }
  
  // get or set the element at the given index
  subscript(index: Int) -> T {
    get {
      precondition(index >= 0 && index < dimension, "Index out of range")
      // use the subscript operator of the PythonObject to access the element
      return T(storage[index])!
    }
    set {
      precondition(index >= 0 && index < dimension, "Index out of range")
      // use the subscript operator of the PythonObject to modify the element
      storage[index] = newValue.pythonObject
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
  
  // define the associated type for the vector element
  associatedtype RealVector: VectorOperations where RealVector.Scalar == Scalar.RealType
  
  // define the instance methods for matrix operations
  // use descriptive names that indicate the corresponding BLAS/LAPACK function
  // use inout parameters for the return values
  func matrixMultiply(by other: Self, into result: inout Self) // corresponds to GEMM
  func eigenDecomposition(into values: inout RealVector, vectors: inout Self) // corresponds to SYEV
  func solveLinearSystem(with rhs: Self, into solution: inout Self) // corresponds to GESV
  func choleskyFactorization(into factor: inout Self) // corresponds to POTRF
  func triangularSolve(with rhs: Self, into solution: inout Self) // corresponds to TRSM
  
  var dimension: Int { get }
  
  // "Matrix \(dataType) \(library)"
  var description: String { get }
  
  subscript(row: Int, column: Int) -> Scalar { get set }
  
  init(dimension: Int, defaultValue: Scalar)
}

protocol VectorOperations: CustomStringConvertible {
  // define the associated type for the scalar element
  associatedtype Scalar: LinearAlgebraScalar
  
  var dimension: Int { get }
  
  // "Matrix \(dataType) \(library)"
  var description: String { get }
  
  init(dimension: Int, defaultValue: Scalar)
  
  subscript(index: Int) -> Scalar { get set }
}

struct AnyMatrixOperations {
  var value: any MatrixOperations
  
  var dimension: Int { value.dimension }
  
  init(_ value: any MatrixOperations) {
    self.value = value
  }
  
  func makeRealVector() -> any VectorOperations {
    func bypassTypeChecking<T: MatrixOperations>(value: T) -> any VectorOperations {
      return T.RealVector(dimension: value.dimension, defaultValue: .zero)
    }
    return bypassTypeChecking(value: value)
  }
  
  func makeMatrix() -> any MatrixOperations {
    func bypassTypeChecking<T: MatrixOperations>(value: T) -> any MatrixOperations {
      return T(dimension: value.dimension, defaultValue: .zero)
    }
    return bypassTypeChecking(value: value)
  }
  
  func matrixMultiply(by other: any MatrixOperations, into result: inout any MatrixOperations) {
    func bypassTypeChecking<T: MatrixOperations>(result: inout T) {
      (value as! T).matrixMultiply(by: other as! T, into: &result)
    }
    bypassTypeChecking(result: &result)
  }
  
  func eigenDecomposition(into values: inout any VectorOperations, vectors: inout any MatrixOperations) {
    func bypassTypeChecking<T: MatrixOperations>(vectors: inout T) {
      var valuesCopy = values as! T.RealVector
      (value as! T).eigenDecomposition(into: &valuesCopy, vectors: &vectors)
      values = valuesCopy
    }
    bypassTypeChecking(vectors: &vectors)
  }
  
  
  subscript(row: Int, column: Int) -> any LinearAlgebraScalar {
    get {
      (value[row, column] as any LinearAlgebraScalar)
    }
    set {
      func bypassTypeChecking<T: MatrixOperations>(value: inout T) {
        value[row, column] = newValue as! T.Scalar
      }
      bypassTypeChecking(value: &value)
    }
  }
}

struct AnyVectorOperations {
  var value: any VectorOperations
  
  var dimension: Int { value.dimension }
  
  init(_ value: any VectorOperations) {
    self.value = value
  }
  
  subscript(index: Int) -> any LinearAlgebraScalar {
    get {
      (value[index] as any LinearAlgebraScalar)
    }
    set {
      func bypassTypeChecking<T: VectorOperations>(value: inout T) {
        value[index] = newValue as! T.Scalar
      }
      bypassTypeChecking(value: &value)
    }
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

func checkDimensions<
  T: MatrixOperations, U: VectorOperations, V: MatrixOperations
>(
  _ lhs: T, _ mhs: U, _ rhs: V
) {
  precondition(
    lhs.dimension == mhs.dimension,
    "Incompatible dimensions: lhs (\(lhs.dimension)) != rhs (\(mhs.dimension))")
  precondition(
    lhs.dimension == rhs.dimension,
    "Incompatible dimensions: lhs (\(lhs.dimension)) != rhs (\(rhs.dimension))")
}

func checkLAPACKError(
  _ error: Int,
  _ file: StaticString = #file,
  _ line: UInt = #line
) {
  if _slowPath(error != 0) {
    let message = """
      Found LAPACK error in \(file):\(line):
      Error code = \(error)
      """
    print(message)
    fatalError(message, file: file, line: line)
  }
}

// extend Matrix to conform to MatrixOperations protocol
extension Matrix: MatrixOperations {
  typealias Scalar = T
  
  typealias RealVector = Vector<T.RealType>
  
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
  
  func eigenDecomposition(into values: inout RealVector, vectors: inout Matrix<T>) {
    // implement eigenvalue decomposition using BLAS/LAPACK functions
    // store the values and vectors in the inout parameters
    checkDimensions(self, values, vectors)
    
    // First copy the input to the output in packed format, then overwrite the
    // output with eigendecomposition.
    var dim: Int = self.dimension
    var error: Int = 0
    var A_copied = UnsafeMutablePointer<T>.allocate(capacity: dim * dim)
    defer { A_copied.deallocate() }
    self.storage.withUnsafeBufferPointer { pointerA in
      let A = unsafeBitCast(pointerA.baseAddress, to: T.PointerSelf?.self)
      
      memcpy(A_copied, unsafeBitCast(A, to: UnsafeMutableRawPointer.self), dim * dim * MemoryLayout<T>.stride)
      
//      vectors.storage.withUnsafeMutableBufferPointer { pointerAP in
//        let AP = unsafeBitCast(pointerAP.baseAddress, to: T.MutablePointerSelf?.self)
//
//        memcpy(unsafeBitCast(AP, to: UnsafeMutableRawPointer.self), unsafeBitCast(A, to: UnsafeMutableRawPointer.self), dim * dim * MemoryLayout<T>.stride)
//      }
    }
    
    var lworkSize: Int = -1
    var rworkSize: Int = -1
    var iworkSize: Int = -1
    
    precondition(dim == self.dimension)
    var lwork: [T] = [.zero]
    var rwork: [T.RealType] = [.zero]
    var iwork: [Int] = [.zero]
    var isuppz: [Int] = Array(repeating: .zero, count: 2 * max(1, dim))
    
    var garbage_v: Scalar.RealType = .zero
    var garbage_i: Int = .zero
    var dim_copy = dim
    
    var abstol: T.RealType = .zero
//    var lamch_str = "S"
//    if T.self == Float.self {
//      abstol = slamch_(&lamch_str) as! T.RealType
//    } else {
//      abstol = dlamch_(&lamch_str) as! T.RealType
//    }
    
    precondition(dim == self.dimension)
    dim_copy = self.dimension
    
    values.storage.withUnsafeMutableBufferPointer { pointerW in
      let W = pointerW.baseAddress
      
      vectors.storage.withUnsafeMutableBufferPointer { pointerZ in
        let Z = unsafeBitCast(pointerZ.baseAddress, to: T.MutablePointerSelf?.self)
        
        lwork.withUnsafeMutableBufferPointer { pointerLWORK in
          let LWORK = unsafeBitCast(pointerLWORK.baseAddress, to: T.MutablePointerSelf.self)
          
          precondition(dim == self.dimension)
          precondition(dim_copy == self.dimension)
          Scalar.linearAlgebraFunctions.syevr(
            "V",
            "A",
            "U",
            &dim,
            unsafeBitCast(A_copied, to: T.MutablePointerSelf?.self),
            &dim,
            &garbage_v,
            &garbage_v,
            &garbage_i,
            &garbage_i,
            &abstol,
            &dim_copy,
            nil,
            Z!,
            &dim,
            &isuppz,
            LWORK,
            &lworkSize,
            &rwork,
            //
            &rworkSize,
            &iwork,
            &iworkSize,
            //
            &error)
          checkLAPACKError(error)
        }
        
        precondition(dim == self.dimension)
        precondition(dim_copy == self.dimension)
        
        if T.self == Float.self {
          lworkSize = Int(lwork[0] as! Float)
          rworkSize = Int(rwork[0] as! Float)
        } else if T.self == Double.self {
          lworkSize = Int(lwork[0] as! Double)
          rworkSize = Int(rwork[0] as! Double)
        } else if T.self == Complex<Double>.self {
          lworkSize = Int((lwork[0] as! Complex<Double>).real)
          rworkSize = Int(rwork[0] as! Double)
        }
        iworkSize = iwork[0]
//        rworkSize = max(1, 3 * dim - 2)
        
//        lwork = Array(repeating: .zero, count: lworkSize)
//        rwork = Array(repeating: .zero, count: rworkSize)
//        iwork = Array(repeating: .zero, count: iworkSize)
        
        lwork.withUnsafeMutableBufferPointer { pointerLWORK in
          let LWORK = unsafeBitCast(pointerLWORK.baseAddress, to: T.MutablePointerSelf.self)
          
//          rwork.withUnsafeMutableBufferPointer { RWORK in
//            iwork.withUnsafeMutableBufferPointer { IWORK in
//              Scalar.linearAlgebraFunctions.syevd(
//                "V",
//                "U",
//                &dim,
//                &A_copied,
//                &dim,
//                W,
//                LWORK,
//                &lworkSize,
//                RWORK.baseAddress,
//                //
//                &rworkSize,
//                IWORK.baseAddress,
//                &iworkSize,
//                //
//                &error)
          
              precondition(dim == self.dimension)
              precondition(dim_copy == self.dimension)
              Scalar.linearAlgebraFunctions.syevr(
                "V",
                "A",
                "U",
                &dim,
                unsafeBitCast(A_copied, to: T.MutablePointerSelf?.self),
                &dim,
                &garbage_v,
                &garbage_v,
                &garbage_i,
                &garbage_i,
                &abstol,
                &dim_copy,
                W,
                Z!,
                &dim,
                &isuppz,
                unsafeBitCast(scratch1, to: T.MutablePointerSelf.self),
                &lworkSize,
                unsafeBitCast(scratch2, to: UnsafeMutablePointer<T.RealType>.self),
                //
                &rworkSize,
                unsafeBitCast(scratch3, to: UnsafeMutablePointer<Int>.self),
                &iworkSize,
                //
                &error)
              checkLAPACKError(error)
//            }
//          }
        }
      }
    }
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

extension Vector: VectorOperations {
  typealias Scalar = T
  
  var description: String {
    if Scalar.self == Complex<Double>.self {
      return "Vector Complex Accelerate"
    } else {
      return "Vector \(Scalar.self) Accelerate"
    }
  }
}

#if os(macOS)
// extend PythonMatrix to conform to MatrixOperations protocol
extension PythonMatrix: MatrixOperations {
  typealias Scalar = T
  
  typealias RealVector = PythonVector<T.RealType>
  
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
  
  func eigenDecomposition(into values: inout RealVector, vectors: inout PythonMatrix<T>) {
    // implement eigenvalue decomposition using NumPy functions
    // store the values and vectors in the inout parameters
    checkDimensions(self, values, vectors)
    
    let (a1, a2) = np.linalg.eigh(self.storage, UPLO: "U").tuple2
    np.copyto(values.storage, a1)
    np.copyto(vectors.storage, a2)
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

extension PythonVector: VectorOperations {
  typealias Scalar = T
  
  var description: String {
    if Scalar.self == Complex<Double>.self {
      return "Vector Complex OpenBLAS"
    } else {
      return "Vector \(Scalar.self) OpenBLAS"
    }
  }
}
#endif


#endif
