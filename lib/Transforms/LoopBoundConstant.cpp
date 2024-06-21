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
      // ::mlir::Region &getBody();
      // Block &front();
      // Operation &front();
      Operation& funcFirstOp = func.getBody().front().front();
      OpBuilder constBuilder(&funcFirstOp);
      auto lbConstOp = constBuilder.create<arith::ConstantIndexOp>(funcFirstOp.getLoc(), 0);
      auto ubConstOp = constBuilder.create<arith::ConstantIndexOp>(funcFirstOp.getLoc(), 8);
      // loopOp.setLowerBound(lbConstOp);
      // loopOp.setUpperBound(ubConstOp);

      // Arguments for scf::WhileOp
      SmallVector<Type> lcvTypes;
      SmallVector<Location> lcvLocs;
      lcvTypes.push_back(lb.getType());
      lcvLocs.push_back(lb.getLoc());
      // for (Value value : loopOp.getInitArgs()) {
      //   lcvTypes.push_back(value.getType());
      //   lcvLocs.push_back(value.getLoc());
      // }

      SmallVector<Value> initArgs;
      initArgs.push_back(loopOp.getLowerBound());
      //llvm::append_range(initArgs, loopOp.getInitArgs());

      OpBuilder whileBuilder(op);
      auto whileOp = whileBuilder.create<scf::WhileOp>(loopOp.getLoc(), lcvTypes, initArgs, loopOp->getAttrs());

      // 'before' region contains the loop condition and forwarding of iteration
      // arguments to the 'after' region.
      auto *beforeBlock = whileBuilder.createBlock(&whileOp.getBefore(), whileOp.getBefore().begin(), lcvTypes, lcvLocs);
      whileBuilder.setInsertionPointToStart(&whileOp.getBefore().front());

      OpBuilder beforeBuilder = OpBuilder::atBlockBegin(&whileOp.getBefore().front());
      auto cmpOp = beforeBuilder.create<arith::CmpIOp>(whileOp.getLoc(), arith::CmpIPredicate::slt, lb, ub);
      whileBuilder.create<scf::ConditionOp>(whileOp.getLoc(), cmpOp.getResult(), beforeBlock->getArguments());

      // 'after' region contains the loop body and the loop increment
      auto *afterBlock = whileBuilder.createBlock(&whileOp.getAfter(), whileOp.getAfter().begin(), lcvTypes, lcvLocs);
      whileBuilder.setInsertionPointToEnd(afterBlock);
      auto ivIncOp = whileBuilder.create<arith::AddIOp>(
        whileOp.getLoc(), loopOp.getLowerBound(), ubConstOp.getResult());
      
      SmallVector<Value> yieldArgs;
      yieldArgs.push_back(ivIncOp.getResult());
      auto yieleOp = whileBuilder.create<scf::YieldOp>(whileOp->getLoc(), yieldArgs);

      // Test OpBuilder
      // OpBuilder builder(op);
      // auto addiOp = builder.create<arith::AddIOp>(
      //   loopOp.getLoc(), loopOp.getLowerBound(), ubConstOp.getResult());
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