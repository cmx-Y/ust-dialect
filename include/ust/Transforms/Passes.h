#ifndef UST_TRANSFORMS_PASSES_H
#define UST_TRANSFORMS_PASSES_H

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"

namespace mlir {
#define GEN_PASS_DECL
#include "ust/Transforms/Passes.h.inc"
}

namespace mlir {
namespace ust {

std::unique_ptr<OperationPass<ModuleOp>> createLoopBoundConstantPass();
std::unique_ptr<OperationPass<ModuleOp>> createSetSparseInfoPass(unsigned iPosSize=1, unsigned iCrdSize=1, unsigned iValSize=1);

void registerUSTPipeline();

bool applySetSparseInfo(ModuleOp &module);
bool applyLoopBoundConstant(ModuleOp &module);

/// Registers all UST transformation passes
void registerUSTPasses();



} // namespace ust
} // namespace mlir

namespace mlir {
#define GEN_PASS_REGISTRATION
#include "ust/Transforms/Passes.h.inc"
}

#endif // UST_TRANSFORMS_PASSES_H