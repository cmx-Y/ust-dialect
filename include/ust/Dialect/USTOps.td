
#ifndef UST_OPS
#define UST_OPS

include "mlir/IR/OpBase.td"
include "mlir/Interfaces/FunctionInterfaces.td"
include "mlir/IR/SymbolInterfaces.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

def UST_Dialect : Dialect {
  let name = "ust";
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

//===----------------------------------------------------------------------===//
// PrintOp
//===----------------------------------------------------------------------===//

def PrintOp : UST_Op<"print"> {
  let summary = "print operation";
  let description = [{
    The "print" builtin operation prints a given input tensor, and produces
    no results.
  }];

  // The print operation takes an input tensor to print.
  let arguments = (ins F64Tensor:$input);

  let assemblyFormat = "$input attr-dict `:` type($input)";
}

#endif // UST_OPS
