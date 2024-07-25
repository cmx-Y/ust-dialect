
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/Transforms/InliningUtils.h"
#include "llvm/ADT/StringExtras.h"

#include "llvm/ADT/TypeSwitch.h"

#include "ust/Dialect/USTDialect.h"
#include "ust/Dialect/USTTypes.h"

using namespace mlir;
using namespace mlir::ust;

#include "ust/Dialect/USTDialect.cpp.inc"

#define GET_ATTRDEF_CLASSES
#include "ust/Dialect/USTAttrs.cpp.inc"

#define GET_TYPEDEF_CLASSES
#include "ust/Dialect/USTTypes.cpp.inc"


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

Attribute SparseInfoAttr::parse(AsmParser &p, Type type) {
  StringRef posSizeKw;
  int64_t posSize;
  if (p.parseLess() || p.parseKeyword(&posSizeKw) || p.parseEqual() ||
      p.parseInteger(posSize) || p.parseComma() ||
      p.parseGreater())
    return Attribute();

  if (posSizeKw != "posSize")
    return Attribute();

  return SparseInfoAttr::get(p.getContext(), posSize);
}

void SparseInfoAttr::print(AsmPrinter &p) const {
  p << "<posSize=" << getPosSize() << ">";
}

#define GET_OP_CLASSES
#include "ust/Dialect/USTOps.cpp.inc"