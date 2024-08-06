
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

def UST_CreateLoopHandleOp : UST_Op<"create_loop_handle"> 
{
    let summary = "loop handle";
    let description = [{
        Create a loop handle

        Parameters
        loop_name (String) - The name of a loop
    }];

    let arguments = (ins OpHandle:$op, StrAttr:$loop_name);
    let results = (outs LoopHandle:$result);
    let assemblyFormat = [{
       $op `,` $loop_name attr-dict
    }];
}

def UST_UnrollOp : UST_Op<"unroll"> 
{
    let summary = "unroll";
    let description = [{
        ust.unroll(var, factor=0)

        Unroll the iteration.

        Parameters
        * var (IterVar) - The iteration to be unrolled.
        * factor (Expr) - The unroll factor. Default value 0 means full unroll.
    }];

    let arguments = (ins LoopHandle:$loop, OptionalAttr<DefaultValuedAttr<UI32Attr,"0">>:$factor);
    let results = (outs );
    let assemblyFormat = [{
        `(` $loop (`,` $factor^)? `)` attr-dict
    }];
}

def UST_PipelineOp : UST_Op<"pipeline"> 
{
    let summary = "pipeline";
    let description = [{
        ust.pipeline(var, initiation_interval=1)

        Pipeline the iteration.

        Parameters
        * var (IterVar) - The iteration to be pipelined.
        * initiation_interval (Expr) - The initiation interval in pipeline schedule. Default value is 1.
    }];

    let arguments = (ins LoopHandle:$loop, OptionalAttr<DefaultValuedAttr<UI32Attr, "1">>:$ii);
    let results = (outs LoopHandle:$result);
    let assemblyFormat = [{
        `(` $loop (`,` $ii^)? `)` attr-dict
    }];
    let builders = [
        OpBuilder<(ins "ust::LoopHandleType":$loop,
                   CArg<"uint64_t", "1">:$ii)>
    ];
}



#endif // UST_OPS
