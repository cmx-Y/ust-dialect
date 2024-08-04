

#ifndef UST_TYPES
#define UST_TYPES

include "ust/Dialect/USTDialect.td"
include "mlir/IR/BuiltinTypeInterfaces.td"

class UST_Type<string mnemonic, list<Trait> traits = [], 
            string baseCppClass = "::mlir::Type"> 
    : TypeDef<UST_Dialect, mnemonic, traits, baseCppClass>;

def Type : UST_Type<"Type", [MemRefElementTypeInterface]> {
  let summary = "generic type";
  let mnemonic = "Type";
}

def OpHandle : UST_Type<"OpHandle"> {
  let summary = "op handle";
  let mnemonic = "OpHandle";
}

#endif // UST_TYPES