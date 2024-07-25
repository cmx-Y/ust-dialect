
#ifndef UST_ATTRS
#define UST_ATTRS

include "ust/Dialect/USTOps.td"
include "mlir/IR/EnumAttr.td"


def SparseInfo : UST_Attr<"SparseInfo"> {
  let summary = "Loop information";
  let description = [{
    This attribute includes loop information used by estimator and optimizer,
    including flattened trip count, iteration latency, and minimum initial
    interval (II).
  }];

  let hasCustomAssemblyFormat = 1;
  let mnemonic = "sparse_info";
  let parameters = (ins "int64_t":$posSize, "int64_t":$crdSize, "int64_t":$valSize);
}

#endif // UST_ATTRS