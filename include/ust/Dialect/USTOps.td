
#ifndef UST_OPS
#define UST_OPS

include "mlir/IR/OpBase.td"
include "mlir/Interfaces/FunctionInterfaces.td"
include "mlir/IR/SymbolInterfaces.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

def UST_Dialect : Dialect {
  let name = "ust";
  let useDefaultTypePrinterParser = 1;
  let useDefaultAttributePrinterParser = 1;
  let cppNamespace = "::mlir::ust";
}

class UST_Op<string mnemonic, list<Trait> traits = []> :
    Op<UST_Dialect, mnemonic, traits>;

class UST_Type<string mnemonic, list<Trait> traits = []> : 
    TypeDef<UST_Dialect, mnemonic, traits>;

class UST_Attr<string mnemonic, list<Trait> traits = []> : 
    AttrDef<UST_Dialect, mnemonic, traits>;

//===----------------------------------------------------------------------===//
// Toy Operations
//===----------------------------------------------------------------------===//


#endif // UST_OPS
