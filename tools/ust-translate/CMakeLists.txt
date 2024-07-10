set(LLVM_LINK_COMPONENTS
  Support
  )

get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(translation_libs GLOBAL PROPERTY MLIR_TRANSLATION_LIBS)

add_llvm_executable(ust-translate
  ust-translate.cpp
  )
llvm_update_compile_flags(ust-translate)

target_link_libraries(ust-translate
  PRIVATE
  ${dialect_libs}
  ${translation_libs}
  MLIRIR
  MLIRParser
  MLIRPass
  MLIRSPIRVDialect
  MLIRTranslateLib
  MLIRSupport
  MLIRUSTEmitHLSCpp
  MLIRUSTSupport
  )


mlir_check_link_libraries(ust-translate)
