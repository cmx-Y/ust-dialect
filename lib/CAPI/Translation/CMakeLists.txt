
add_mlir_public_c_api_library(MLIRUSTCAPIEmitHLSCpp
  EmitVivadoHLS.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir-c

  LINK_LIBS PUBLIC
  MLIRCAPIIR
  MLIRUSTEmitHLSCpp
  )
