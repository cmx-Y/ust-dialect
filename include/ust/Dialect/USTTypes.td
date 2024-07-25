

#ifndef UST_TYPES
#define UST_TYPES

include "ust/Dialect/USTOps.td"
include "mlir/IR/BuiltinTypeInterfaces.td"

def Type : UST_Type<"Type", [MemRefElementTypeInterface]> {
  let summary = "generic type";
  let mnemonic = "Type";
}

#endif // UST_TYPES