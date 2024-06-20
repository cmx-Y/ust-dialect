/*
* @author: Xingyan Chen
* @date: 2024/06/16
* @description: This file is used to implement the LoopBoundConstant pass.
*/

#include <iostream>

#include "PassDetail.h"
#include "ust/Transforms/Passes.h"

#include "mlir/Pass/Pass.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"

using namespace mlir;
using namespace ust;

namespace mlir {
namespace ust {

void LoopBoundConstant(func::FuncOp &func) {
  SmallVector<Operation *, 8> loopOps;
  // Get all loop operations
  func.walk([&](Operation *op) {
    if (auto loopOp = dyn_cast<scf::ForOp>(op)) {
      loopOps.push_back(loopOp);
    }
  });

  for (auto op : loopOps) {
    auto loopOp = cast<scf::ForOp>(op);
    auto lb = loopOp.getLowerBound();
    auto ub = loopOp.getUpperBound();
    auto step = loopOp.getStep();
    
    // If the lower bound and upper bound are both constants, skip
    if (lb.getDefiningOp<arith::ConstantOp>() && ub.getDefiningOp<arith::ConstantOp>()) {
      continue;
    }
    else {
      OpBuilder ifBuilder(op);
      //auto ifOp = ifBuilder.create<scf::IfOp>(op->getLoc(), lb , false);

      // ::mlir::Region &getBody();
      // Block &front();
      // Operation &front();
      Operation& funcFirstOp = func.getBody().front().front();
      OpBuilder constBuilder(&funcFirstOp);
      auto lbConstOp = constBuilder.create<arith::ConstantIndexOp>(funcFirstOp.getLoc(), 0);
      auto ubConstOp = constBuilder.create<arith::ConstantIndexOp>(funcFirstOp.getLoc(), 8);
      // loopOp.setLowerBound(lbConstOp);
      // loopOp.setUpperBound(ubConstOp);
    }
  }

}

/// Pass entry point
bool applyLoopBoundConstant(ModuleOp &module) {
  for (func::FuncOp func : module.getOps<func::FuncOp>()) {
    if (func.getBlocks().empty()) {
      continue;
    }
    else {
      LoopBoundConstant(func);
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
    if (!applyLoopBoundConstant(mod)) {
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