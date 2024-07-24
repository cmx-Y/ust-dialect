
#ifndef UST_DIALECT_H_
#define UST_DIALECT_H_

#include "mlir/Bytecode/BytecodeOpInterface.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Interfaces/CallInterfaces.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

/// Include the auto-generated header file containing the declaration of the toy
/// dialect.
#include "ust/Dialect/USTDialect.h.inc"

/// Include the auto-generated header file containing the declarations of the
/// toy operations.
#define GET_OP_CLASSES
#include "ust/Dialect/USTOps.h.inc"

#define GET_ATTRDEF_CLASSES
#include "ust/Dialect/USTAttrs.h.inc"

namespace mlir {
namespace ust {
void setSparseInfo(Operation *op, SparseInfoAttr sparseInfo);
void setSparseInfo(Operation *op, int64_t posSize);
}
}

#endif // UST_DIALECT_H_
