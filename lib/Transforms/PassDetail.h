/*
* @author: Xingyan Chen
* @date: 2024/06/16
*/

#ifndef UST_MLIR_PASSDETAIL_H
#define UST_MLIR_PASSDETAIL_H

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"

namespace mlir {
namespace ust {

#define GEN_PASS_CLASSES
#include "ust/Transforms/Passes.h.inc"

} // namespace ust
} // end namespace mlir

#endif // UST_MLIR_PASSDETAIL_H
