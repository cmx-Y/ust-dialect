
set(LLVM_LINK_COMPONENTS
  Core
  Support
  nativecodegen
  OrcJIT
  )
get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)
set(LIBS
        ${dialect_libs}
        ${conversion_libs}
        MLIRExecutionEngine
        MLIRLLVMCommonConversion
        MLIRLLVMDialect
        MLIRLLVMToLLVMIRTranslation
        MLIRTargetLLVMIRExport
        MLIRIR
        MLIROptLib
        MLIRUSTPasses
        )
add_llvm_executable(ust-opt ust-opt.cpp)

llvm_update_compile_flags(ust-opt)
target_link_libraries(ust-opt PRIVATE ${LIBS})

mlir_check_all_link_libraries(ust-opt)
