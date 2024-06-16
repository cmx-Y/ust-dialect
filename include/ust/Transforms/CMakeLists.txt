
set(LLVM_TARGET_DEFINITIONS Passes.td)
mlir_tablegen(Passes.h.inc -gen-pass-decls)
add_public_tablegen_target(MLIRUSTPassesIncGen)

add_mlir_doc(USTPass USTPass UST/ -gen-pass-doc)