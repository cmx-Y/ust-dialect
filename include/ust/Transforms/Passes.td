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

def SetSparseInfo : Pass<"set-sparse-info", "ModuleOp"> {
  let summary = "Set sparse info pass";
  let constructor = "mlir::ust::createSetSparseInfoPass()";

  let options = [
    Option<"posSize", "pos-size", "unsigned", /*default=*/"1",
           "Positive number: position array size">,
    Option<"crdSize", "crd-size", "unsigned", /*default=*/"1",
           "Positive number: coordinate array size">,
    Option<"valSize", "val-size", "unsigned", /*default=*/"1",
            "Positive number: value array size">,
  ];
}

#endif // UST_MLIR_PASSES