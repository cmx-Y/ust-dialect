
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
  addTypes<
#define GET_TYPEDEF_LIST
#include "ust/Dialect/USTTypes.cpp.inc"
      >(); 
  addAttributes< 
#define GET_ATTRDEF_LIST
#include "ust/Dialect/USTAttrs.cpp.inc"
      >();
}


void ust::setSparseInfo(Operation *op, SparseInfoAttr sparseInfo) {
  op->setAttr("sparse_info", sparseInfo);
}

void ust::setSparseInfo(Operation *op, int64_t posSize, int64_t crdSize, int64_t valSize) {
  auto sparseInfo =
      SparseInfoAttr::get(op->getContext(), posSize, crdSize, valSize);
  setSparseInfo(op, sparseInfo);
}

Attribute SparseInfoAttr::parse(AsmParser &p, Type type) {
  StringRef posSizeKw, crdSizeKw, valSizeKw;
  int64_t posSize, crdSize, valSize;
  if (p.parseLess() || p.parseKeyword(&posSizeKw) || p.parseEqual() ||
      p.parseInteger(posSize) || p.parseComma() ||
      p.parseKeyword(&crdSizeKw) || p.parseEqual() ||
      p.parseInteger(crdSize) || p.parseComma() ||
      p.parseKeyword(&valSizeKw) || p.parseEqual() ||
      p.parseInteger(valSize) || p.parseComma() ||
      p.parseGreater())
    return Attribute();

  if (posSizeKw != "posSize" ||
      crdSizeKw != "crdSize" ||
      valSizeKw != "valSize")
    return Attribute();

  return SparseInfoAttr::get(p.getContext(), posSize, crdSize, valSize);
}

void SparseInfoAttr::print(AsmPrinter &p) const {
  p << "<posSize=" << getPosSize() << ", "
    << "crdSize=" << getCrdSize() << ", " << "valSize=" << getValSize() << ">";
}

#define GET_OP_CLASSES
#include "ust/Dialect/USTOps.cpp.inc"