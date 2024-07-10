#ifndef UST_ANALYSIS_UTILS_H
#define UST_ANALYSIS_UTILS_H

#include "mlir/Dialect/Affine/Analysis/LoopAnalysis.h"
#include "mlir/Dialect/Affine/Analysis/Utils.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Affine/IR/AffineValueMap.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"

using namespace mlir::affine;

namespace mlir {
namespace ust {

//===----------------------------------------------------------------------===//
// HLSCpp attribute parsing utils
//===----------------------------------------------------------------------===//

std::vector<std::string> split_names(const std::string &arg_names);

} // namespace ust
} // namespace mlir

#endif // UST_ANALYSIS_UTILS_H
