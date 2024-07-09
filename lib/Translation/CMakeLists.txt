# Copyright HeteroCL authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

file(GLOB globbed *.cpp)

add_mlir_library(MLIRUSTEmitHLSCpp
  ${globbed}

  LINK_LIBS PUBLIC
  MLIRMemRefDialect
  MLIRAnalysis
)