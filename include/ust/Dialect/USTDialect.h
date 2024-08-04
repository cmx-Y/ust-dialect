
#ifndef UST_DIALECT_H_
#define UST_DIALECT_H_

#include "mlir/Bytecode/BytecodeOpInterface.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Interfaces/CallInterfaces.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"


#include "ust/Dialect/USTDialect.h.inc"

#define GET_ATTRDEF_CLASSES
#include "ust/Dialect/USTAttrs.h.inc"

#define GET_TYPEDEF_CLASSES
#include "ust/Dialect/USTTypes.h.inc"

#define GET_OP_CLASSES
#include "ust/Dialect/USTOps.h.inc"

namespace mlir {
namespace ust {
void setSparseInfo(Operation *op, SparseInfoAttr sparseInfo);
void setSparseInfo(Operation *op, int64_t posSize, int64_t crdSize, int64_t valSize);
}
}

#endif // UST_DIALECT_H_
