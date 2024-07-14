
add_mlir_public_c_api_library(MLIRUSTCAPI
  Registration.cpp
  ${PROJECT_SOURCE_DIR}/lib/Transforms/Passes.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir-c

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRCAPIIR
  MLIRSupport
  MLIRUSTPasses
  )
