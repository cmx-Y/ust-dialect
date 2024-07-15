from ust_mlir.ir import Context, Module

def test_codegen():
    mlir_code = """
    module {
        func.func @matvec(%arg0: tensor<32x64xf64, #sparse_tensor.encoding<{ lvlTypes = [ "dense", "compressed" ] }>>, %arg1: tensor<64xf64>, %arg2: tensor<32xf64>) -> tensor<32xf64> {
            %c32 = arith.constant 32 : index
            %c0 = arith.constant 0 : index
            %c1 = arith.constant 1 : index
            %0 = sparse_tensor.positions %arg0 {level = 1 : index} : tensor<32x64xf64, #sparse_tensor.encoding<{ lvlTypes = [ "dense", "compressed" ] }>> to memref<?xindex>
            %1 = sparse_tensor.coordinates %arg0 {level = 1 : index} : tensor<32x64xf64, #sparse_tensor.encoding<{ lvlTypes = [ "dense", "compressed" ] }>> to memref<?xindex>
            %2 = sparse_tensor.values %arg0 : tensor<32x64xf64, #sparse_tensor.encoding<{ lvlTypes = [ "dense", "compressed" ] }>> to memref<?xf64>
            %3 = bufferization.to_memref %arg1 : memref<64xf64>
            %4 = bufferization.to_memref %arg2 : memref<32xf64>
            scf.for %arg3 = %c0 to %c32 step %c1 {
                %6 = memref.load %4[%arg3] : memref<32xf64>
                %7 = memref.load %0[%arg3] : memref<?xindex>
                %8 = arith.addi %arg3, %c1 : index
                %9 = memref.load %0[%8] : memref<?xindex>
                %10 = scf.for %arg4 = %7 to %9 step %c1 iter_args(%arg5 = %6) -> (f64) {
                    %11 = memref.load %1[%arg4] : memref<?xindex>
                    %12 = memref.load %2[%arg4] : memref<?xf64>
                    %13 = memref.load %3[%11] : memref<64xf64>
                    %14 = arith.mulf %12, %13 : f64
                    %15 = arith.addf %arg5, %14 : f64
                    scf.yield %15 : f64
                } {"Emitted from" = "linalg.generic"}
                memref.store %10, %4[%arg3] : memref<32xf64>
            } {"Emitted from" = "linalg.generic"}
            %5 = bufferization.to_tensor %4 : memref<32xf64>
            return %5 : tensor<32xf64>
        }
    }
    """
    ctx = Context()
    mod = Module.parse(mlir_code, ctx)
    mod.dump()


if __name__ == "__main__":
    test_codegen()