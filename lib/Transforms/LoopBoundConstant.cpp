/*
* @author: Xingyan Chen
* @date: 2024/06/16
* @description: This file is used to implement the LoopBoundConstant pass.
*/

#include "PassDetail.h"
#include "ust/Transforms/Passes.h"

#include "mlir/Pass/Pass.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"

using namespace mlir;
using namespace ust;

namespace mlir {
namespace ust {

/// Pass entry point
bool applyUSTLoopBoundConstant(ModuleOp &module) {
  for (func::FuncOp func : module.getOps<func::FuncOp>()) {
    if (func.getBlocks().empty()) {
      continue;
    }
  }
  return true;
}

}   // namespace ust
}   // namespace mlir

namespace {
struct USTLoopBoundConstantTransformation
    : public LoopBoundConstantBase<USTLoopBoundConstantTransformation> {
  void runOnOperation() override {
    auto mod = getOperation();
    if (!applyUSTLoopBoundConstant(mod)) {
      signalPassFailure();
    }
  }
};
} // namespace

namespace mlir {
namespace ust {
std::unique_ptr<OperationPass<ModuleOp>> createLoopBoundConstantPass() {
  return std::make_unique<USTLoopBoundConstantTransformation>();
}
} // namespace ust
} // namespace mlir