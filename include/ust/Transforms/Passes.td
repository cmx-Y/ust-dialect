/*
 * @author: Xingyan Chen
 * @date: 2024/06/16
 * @description: This file defines the passes for the UST dialect.
*/

#ifndef UST_MLIR_PASSES
#define UST_MLIR_PASSES

include "mlir/Pass/PassBase.td"

def LoopBoundConstant : Pass<"loop-bound-const", "ModuleOp"> {
  let summary = "Transform loop bound to constant pass";
  let constructor = "mlir::ust::createLoopBoundConstantPass()";
}

#endif // UST_MLIR_PASSES