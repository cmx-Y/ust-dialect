
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/Transforms/InliningUtils.h"
#include "llvm/ADT/StringExtras.h"

#include "llvm/ADT/TypeSwitch.h"

#include "ust/Dialect/USTDialect.h"

using namespace mlir;
using namespace mlir::ust;

#include "ust/Dialect/USTDialect.cpp.inc"

#define GET_ATTRDEF_CLASSES
#include "ust/Dialect/USTAttrs.cpp.inc"


//===----------------------------------------------------------------------===//
// Dialect initialize method.
//===----------------------------------------------------------------------===//
void USTDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "ust/Dialect/USTOps.cpp.inc"
      >();
  addAttributes< 
#define GET_ATTRDEF_LIST
#include "ust/Dialect/USTAttrs.cpp.inc"
      >();
}


void ust::setSparseInfo(Operation *op, SparseInfoAttr sparseInfo) {
  op->setAttr("sparse_info", sparseInfo);
}

void ust::setSparseInfo(Operation *op, int64_t posSize) {
  auto sparseInfo =
      SparseInfoAttr::get(op->getContext(), posSize);
  setSparseInfo(op, sparseInfo);
}
