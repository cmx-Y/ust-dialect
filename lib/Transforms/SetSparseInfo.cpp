/*
* @author: Xingyan Chen
* @date: 2024/06/16
* @description: This file is used to implement the LoopBoundConstant pass.
*/

#include <iostream>

#include "ust/Transforms/Passes.h"
#include "ust/Dialect/USTDialect.h"

#include "mlir/Pass/Pass.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/SparseTensor/IR/SparseTensor.h"

namespace mlir {
#define GEN_PASS_DEF_SETSPARSEINFO
#include "ust/Transforms/Passes.h.inc"
} // namespace mlir



using namespace mlir;
using namespace ust;

namespace mlir {
namespace ust {

void SetSparseInfo(func::FuncOp &func, unsigned iPosSize, 
                                       unsigned iCrdSize, unsigned iValSize) {
  func.walk([&](Operation *op) {
    if (auto toPositionOp = dyn_cast<sparse_tensor::ToPositionsOp>(op)) {
      setSparseInfo(op, iPosSize, iCrdSize, iValSize);
    }
  });
}

/// Pass entry point
bool applySetSparseInfo(ModuleOp &module, unsigned iPosSize, 
                        unsigned iCrdSize, unsigned iValSize) {
  for (func::FuncOp func : module.getOps<func::FuncOp>()) {
    if (func.getBlocks().empty()) {
      continue;
    }
    else {
      SetSparseInfo(func, iPosSize, iCrdSize, iValSize);
    }
  }
  return true;
}

}   // namespace ust
}   // namespace mlir

namespace {
struct USTSetSparseInfoTransformation
    : public impl::SetSparseInfoBase<USTSetSparseInfoTransformation> {

  USTSetSparseInfoTransformation() = default;
  explicit USTSetSparseInfoTransformation(unsigned iPosSize, unsigned iCrdSize, unsigned iValSize) {
    this->posSize = iPosSize;
    this->crdSize = iCrdSize;
    this->valSize = iValSize;
  }
  
  void runOnOperation() override {
    printf("%u", this->posSize.getValue());
    auto mod = getOperation();
    if (!applySetSparseInfo(mod, this->posSize.getValue(), 
                                 this->crdSize.getValue(), this->valSize.getValue())) {
      signalPassFailure();
    }
  }
};
} // namespace

namespace mlir {
namespace ust {
std::unique_ptr<OperationPass<ModuleOp>> createSetSparseInfoPass(unsigned iPosSize, unsigned iCrdSize, unsigned iValSize) {
  return std::make_unique<USTSetSparseInfoTransformation>(iPosSize, iCrdSize, iValSize);
}

} // namespace ust
} // namespace mlir