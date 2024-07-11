// $ cd ust-dialect/build
// $ ./bin/ust-translate ../test/SparseTensor/examples/csr_spmv.mlir -emit-vivado-hls \ 
//   > ../test/SparseTensor/codegen_output/csr_spmv_codegen.cpp

module {
  func.func @matvec(%arg0: tensor<32x64xf64, #sparse_tensor.encoding<{ lvlTypes = [ "dense", "compressed" ] }>>, %arg1: tensor<64xf64>, %arg2: tensor<32xf64>) -> tensor<32xf64> {
    %c0 = arith.constant 0 : index
    %c8 = arith.constant 8 : index
    %c32 = arith.constant 32 : index
    %c0_0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %0 = sparse_tensor.positions %arg0 {level = 1 : index} : tensor<32x64xf64, #sparse_tensor.encoding<{ lvlTypes = [ "dense", "compressed" ] }>> to memref<?xindex>
    %1 = sparse_tensor.coordinates %arg0 {level = 1 : index} : tensor<32x64xf64, #sparse_tensor.encoding<{ lvlTypes = [ "dense", "compressed" ] }>> to memref<?xindex>
    %2 = sparse_tensor.values %arg0 : tensor<32x64xf64, #sparse_tensor.encoding<{ lvlTypes = [ "dense", "compressed" ] }>> to memref<?xf64>
    %3 = bufferization.to_memref %arg1 : memref<64xf64>
    %4 = bufferization.to_memref %arg2 : memref<32xf64>
    scf.for %arg3 = %c0_0 to %c32 step %c1 {
      %6 = memref.load %4[%arg3] : memref<32xf64>
      %7 = memref.load %0[%arg3] : memref<?xindex>
      %8 = arith.addi %arg3, %c1 : index
      %9 = memref.load %0[%8] : memref<?xindex>
      %10 = scf.while (%arg4 = %7) : (index) -> index {
        %11 = arith.cmpi slt, %7, %9 : index
        scf.condition(%11) %arg4 : index
      } do {
      ^bb0(%arg4: index):
        %11 = scf.for %arg5 = %c0 to %c8 step %c1 iter_args(%arg6 = %6) -> (f64) {
          %13 = arith.addi %arg5, %7 : index
          %14 = memref.load %1[%13] : memref<?xindex>
          %15 = memref.load %2[%13] : memref<?xf64>
          %16 = memref.load %3[%14] : memref<64xf64>
          %17 = arith.mulf %15, %16 : f64
          %18 = arith.addf %arg6, %17 : f64
          scf.yield %18 : f64
        } {"Emitted from" = "linalg.generic"}
        memref.store %11, %4[%arg3] : memref<32xf64>
        %12 = arith.addi %7, %c8 : index
        scf.yield %12 : index
      } attributes {"Emitted from" = "linalg.generic"}
    } {"Emitted from" = "linalg.generic"}
    %5 = bufferization.to_tensor %4 : memref<32xf64>
    return %5 : tensor<32xf64>
  }
}

