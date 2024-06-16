#ifndef UST_TRANSFORMS_PASSES_H
#define UST_TRANSFORMS_PASSES_H

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"

namespace mlir {
namespace ust {

std::unique_ptr<OperationPass<ModuleOp>> createLoopBoundConstantPass();

bool applyUSTLoopBoundConstant(ModuleOp &module);

/// Registers all UST transformation passes
void registerUSTPasses();

} // namespace ust
} // namespace mlir

#endif // UST_TRANSFORMS_PASSES_H