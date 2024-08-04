
#ifndef UST_OPS
#define UST_OPS

include "ust/Dialect/USTDialect.td"
include "ust/Dialect/USTTypes.td"
include "ust/Dialect/USTAttrs.td"

include "mlir/IR/BuiltinTypes.td"
include "mlir/Interfaces/FunctionInterfaces.td"
include "mlir/IR/SymbolInterfaces.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

class UST_Op<string mnemonic, list<Trait> traits = []> :
    Op<UST_Dialect, mnemonic, traits>;

def UST_CreateOpHandleOp : UST_Op<"create_op_handle"> 
{
    let summary = "op handle";
    let description = [{
        Create a op handle

        Parameters
        op_name (String) - The name of a op
    }];

    let arguments = (ins StrAttr:$op_name);
    let results = (outs OpHandle:$result);
    let assemblyFormat = [{
       $op_name attr-dict
    }];
}



#endif // UST_OPS
