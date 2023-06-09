//
//  MainFile.swift
//  AMXBenchmarks
//
//  Created by Philip Turner on 3/24/23.
//

import Foundation
#if os(macOS)
import PythonKit
#endif
import RealModule
import ComplexModule

func mainFunc() {
  #if os(macOS)
  let downloadsURL = FileManager.default.urls(
    for: .downloadsDirectory, in: .userDomainMask)[0]
  let homePath = downloadsURL.deletingLastPathComponent().relativePath
  PythonLibrary.useLibrary(at: homePath + "/miniforge3/bin/python")
  print(Python.version)
  print()
  
  // TODO: Query number of performance cores on machines besides Mx Pro/Max.
  setenv("NUM_THREADS", "8", 0)
  setenv("OMP_NUM_THREADS", "8", 0)
  setenv("OPENBLAS_NUM_THREADS", "8", 0)
  #endif
  
  // This actually works! Setting to 1 thread decreases GEMM performance.
  setenv("VECLIB_MAXIMUM_THREADS", "8", 0)
  
  // Run initial tests that everything works.
  boilerplateLikeCode()
  
  // define a constant for the number of repetitions
  let REPS = 50
  
  // define a constant for the dimension
  let N = 512
  
  // define a function that takes two matrices and performs matrix multiplication on them
  // use generic parameters that conform to MatrixOperations protocol
  // use an inout parameter for the result matrix
  func matrixMultiply<T: MatrixOperations>(lhs: T, rhs: T, into result: inout T) {
    // perform matrix multiplication using the protocol method
    lhs.matrixMultiply(by: rhs, into: &result)
  }
  
  // define a function that takes a matrix and performs eigenvalue decomposition on it
  // use a generic parameter that conforms to MatrixOperations protocol
  // use inout parameters for the eigenvalues and eigenvectors arrays
  func eigenDecompose<T: MatrixOperations>(matrix: T, into values: inout T.RealVector, vectors: inout T) {
    // perform eigenvalue decomposition using the protocol method
    matrix.eigenDecomposition(into: &values, vectors: &vectors)
  }
  
  // define a function that takes two matrices and solves a linear system on them
  // use generic parameters that conform to MatrixOperations protocol
  // use an inout parameter for the solution matrix
  func solveLinearSystem<T: MatrixOperations>(lhs: T, rhs: T, into solution: inout T) {
    // solve the linear system using the protocol method
    lhs.solveLinearSystem(with: rhs, into: &solution)
  }
  
  // define a function that takes a matrix and performs Cholesky factorization on it
  // use a generic parameter that conforms to MatrixOperations protocol
  // use an inout parameter for the factor matrix
  func choleskyFactorize<T: MatrixOperations>(matrix: T, into factor: inout T) {
    // perform Cholesky factorization using the protocol method
    matrix.choleskyFactorization(into: &factor)
  }
  
  // define a function that takes two matrices and solves a triangular system on them
  // use generic parameters that conform to MatrixOperations protocol
  // use an inout parameter for the solution matrix
  func triangularSolve<T: MatrixOperations>(lhs: T, rhs: T, into solution: inout T) {
    // solve the triangular system using the protocol method
    lhs.triangularSolve(with: rhs, into: &solution)
  }
  
  // TODO: Make these randomized before passing into anything besides GEMM.
  
  // create some matrices of different data types and libraries
  let matrixFloat = Matrix<Float>(dimension: N, defaultValue: 0) // a Float matrix using Accelerate
  let matrixDouble = Matrix<Double>(dimension: N, defaultValue: 0) // a Double matrix using Accelerate
  let matrixComplex = Matrix<Complex<Double>>(dimension: N, defaultValue: 0) // a Complex<Double> matrix using Accelerate
  #if os(macOS)
  let pythonMatrixFloat = PythonMatrix<Float>(dimension: N, defaultValue: 0) // a Float matrix using NumPy
  let pythonMatrixDouble = PythonMatrix<Double>(dimension: N, defaultValue: 0) // a Double matrix using NumPy
  let pythonMatrixComplex = PythonMatrix<Complex<Double>>(dimension: N, defaultValue: 0) // a Complex<Double> matrix using NumPy
  #endif
  
  // create an array of matrices as type-erased wrappers
  var matrices: [AnyMatrixOperations] = [
    AnyMatrixOperations(matrixFloat),
    AnyMatrixOperations(matrixDouble),
    AnyMatrixOperations(matrixComplex)
  ]
  #if os(macOS)
  matrices += [
    AnyMatrixOperations(pythonMatrixFloat),
    AnyMatrixOperations(pythonMatrixDouble),
    AnyMatrixOperations(pythonMatrixComplex)
  ]
  #endif
  
  // create a dictionary to store the minimum elapsed time for each function and data type combination
  var minElapsedTimes = [String : Double]()
  
  func convert_GFLOPS_k(time: Double) -> Double {
    return Double(N * N * N) / time / 1e9
  }
  
  for matrix in matrices {
    loopBody(matrix.value)
    
    func loopBody<T: MatrixOperations>(_ matrix: T) {
      // get the data type and library name from the matrix description
      let dataType = String(matrix.description.split(separator: " ")[1])
      let libraryName = String(matrix.description.split(separator: " ")[2])
      
      // create some arrays and matrices to store the inputs and outputs of each function call
      // create some arrays and matrices to store the inputs and outputs of each function call
      var lhs = matrix // the left-hand side matrix for multiplication and linear system
      var rhs = matrix // the right-hand side matrix for multiplication and linear system
      var result = matrix // the result matrix for multiplication, linear system, and triangular system
      var values = [T.Scalar](repeating: .zero, count: matrix.dimension) // the eigenvalues array for eigenvalue decomposition
      var vectors = matrix // the eigenvectors matrix for eigenvalue decomposition
      var factor = matrix // the factor matrix for Cholesky factorization
      
      // create a variable to store the minimum elapsed time for each function call
      var minElapsed = Double.infinity
      
      // repeat the matrix multiplication N times
      for _ in 1...REPS {
        // get the current time before calling the function
        let start = DispatchTime.now()
        // perform matrix multiplication using the generic function
        matrixMultiply(lhs: lhs, rhs: rhs, into: &result)
        // get the current time after calling the function
        let end = DispatchTime.now()
        // calculate the elapsed time in seconds
        let elapsed = Double(end.uptimeNanoseconds - start.uptimeNanoseconds) / 1_000_000_000
        // update the minimum elapsed time if needed
        minElapsed = min(minElapsed, elapsed)
      }
      // store the minimum elapsed time for matrix multiplication in the dictionary
      minElapsedTimes["\(libraryName)_\(dataType)_gemm"] =
        convert_GFLOPS_k(time: minElapsed)
    }
  }
  
  // print the dictionary of minimum elapsed times for each function and data type combination
//  print(minElapsedTimes)
  
  for (key, value) in minElapsedTimes.enumerated()
    .sorted(by: { $0.element.key < $1.element.key })
  {
    print(String(format: "%.1f", value.value), "\t\t", value.key)
  }
}

// Stuff that I should wrap in an iteration over the generic type.
func boilerplateLikeCode() {
  do {
    // create a 3x3 matrix of numbers with default value 0
    var m = Matrix<Float>(dimension: 3, defaultValue: 0)
    
    // set some elements using the subscript operator
    m[0, 0] = 1
    m[1, 1] = 2
    m[2, 2] = 3
    
    // print the matrix using a nested loop
    for i in 0..<m.dimension {
      for j in 0..<m.dimension {
        print(m[i, j], terminator: " ")
      }
      print()
    }
    
    var out = m
    m.matrixMultiply(by: m, into: &out)
    print(out)
  }
  
  do {
    // create a 3x3 matrix of numbers with default value 0
    var m = Matrix<Double>(dimension: 3, defaultValue: 0)
    
    // set some elements using the subscript operator
    m[0, 0] = 1
    m[1, 1] = 2
    m[2, 2] = 3
    
    // print the matrix using a nested loop
    for i in 0..<m.dimension {
      for j in 0..<m.dimension {
        print(m[i, j], terminator: " ")
      }
      print()
    }
    
    var out = m
    m.matrixMultiply(by: m, into: &out)
    print(out)
  }
  
  do {
    // create a 3x3 matrix of numbers with default value 0
    var m = Matrix<Complex<Double>>(dimension: 3, defaultValue: 0)
    
    // set some elements using the subscript operator
    m[0, 0] = 1
    m[1, 1] = 2
    m[2, 2] = 3
    
    // print the matrix using a nested loop
    for i in 0..<m.dimension {
      for j in 0..<m.dimension {
        print(m[i, j], terminator: " ")
      }
      print()
    }
    
    var out = m
    m.matrixMultiply(by: m, into: &out)
    print(out)
  }
  
  #if os(macOS)
  do {
    // create a 3x3 matrix of numbers with default value 0
    var m = PythonMatrix<Float>(dimension: 3, defaultValue: 0)

    // set some elements using the subscript operator
    m[0, 0] = 1
    m[1, 1] = 2
    m[2, 2] = 3

    // print the matrix using a nested loop
    for i in 0..<m.dimension {
        for j in 0..<m.dimension {
            print(m[i, j], terminator: " ")
        }
        print()
    }
    
    var out = m
    m.matrixMultiply(by: m, into: &out)
    print(out)
  }
  
  do {
    // create a 3x3 matrix of numbers with default value 0
    var m = PythonMatrix<Double>(dimension: 3, defaultValue: 0)

    // set some elements using the subscript operator
    m[0, 0] = 1
    m[1, 1] = 2
    m[2, 2] = 3

    // print the matrix using a nested loop
    for i in 0..<m.dimension {
        for j in 0..<m.dimension {
            print(m[i, j], terminator: " ")
        }
        print()
    }
    
    var out = m
    m.matrixMultiply(by: m, into: &out)
    print(out)
  }
  
  do {
    // create a 3x3 matrix of numbers with default value 0
    var m = PythonMatrix<Complex<Double>>(dimension: 3, defaultValue: 0)

    // set some elements using the subscript operator
    m[0, 0] = 1
    m[1, 1] = 2
    m[2, 2] = 3

    // print the matrix using a nested loop
    for i in 0..<m.dimension {
        for j in 0..<m.dimension {
            print(m[i, j], terminator: " ")
        }
        print()
    }
    
    var out = m
    m.matrixMultiply(by: m, into: &out)
    print(out)
  }
  #endif
}
