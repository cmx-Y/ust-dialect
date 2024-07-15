
include(AddMLIRPython)

find_package(pybind11 CONFIG REQUIRED)

# The directory at which the Python import tree begins.
# See documentation for `declare_mlir_python_sources`'s ROOT_DIR
# argument.
set(UST_MLIR_PYTHON_ROOT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/ust")
set(UST_MLIR_PYTHON_PACKAGES_DIR "${PROJECT_BINARY_DIR}/tools/ust/python_packages")
set(MLIR_PYTHON_SOURCE_DIR "${MLIR_MAIN_SRC_DIR}/lib/Bindings/Python")
set(UST_PYTHON_SOURCE_DIR "${PROJECT_SOURCE_DIR}/lib/Bindings/Python")

# Debug print
message(STATUS "UST_MLIR_PYTHON_ROOT_DIR: ${UST_MLIR_PYTHON_ROOT_DIR}")
message(STATUS "UST_MLIR_PYTHON_PACKAGES_DIR: ${UST_MLIR_PYTHON_PACKAGES_DIR}")
message(STATUS "MLIR_PYTHON_SOURCE_DIR: ${MLIR_PYTHON_SOURCE_DIR}")
message(STATUS "UST_PYTHON_SOURCE_DIR: ${UST_PYTHON_SOURCE_DIR}")

include_directories(${MLIR_PYTHON_SOURCE_DIR})

# We vendor our own MLIR instance in the `ust_mlir` namespace.
add_compile_definitions("MLIR_PYTHON_PACKAGE_PREFIX=ust_mlir.")

################################################################################
# Sources
################################################################################

declare_mlir_python_sources(USTMLIRPythonSources)
declare_mlir_python_sources(USTMLIRPythonExtensions)

declare_mlir_python_sources(USTMLIRPythonSources.Dialects
  ROOT_DIR "${UST_MLIR_PYTHON_ROOT_DIR}"
  ADD_TO_PARENT USTMLIRPythonSources
)

declare_mlir_dialect_python_bindings(
  ADD_TO_PARENT USTMLIRPythonSources.Dialects
  ROOT_DIR "${UST_MLIR_PYTHON_ROOT_DIR}"
  # TD_FILE dialects/HeteroCLBinding.td
  SOURCES
    dialects/ust.py
    build_ir.py
    exceptions.py
    __init__.py
  DIALECT_NAME ust
)

################################################################################
# Extensions
################################################################################

declare_mlir_python_extension(USTMLIRPythonExtensions.Main
  MODULE_NAME _ust
  ADD_TO_PARENT USTMLIRPythonExtensions
  ROOT_DIR "/"
  SOURCES
    ${UST_PYTHON_SOURCE_DIR}/USTModule.cpp
  EMBED_CAPI_LINK_LIBS
    MLIRCAPIIR
    MLIRCAPIDebug
    MLIRUSTCAPI
    MLIRUSTCAPIEmitHLSCpp
  PRIVATE_LINK_LIBS
    MLIRPass
    MLIRUSTPasses
    LLVMSupport
)

################################################################################
# Generate packages and shared library
# Downstreams typically will not use these, but they are useful for local
# testing.
################################################################################

set(_source_components
  USTMLIRPythonSources
  USTMLIRPythonExtensions
  MLIRPythonSources
  MLIRPythonExtension.RegisterEverything
)

add_mlir_python_common_capi_library(USTMLIRAggregateCAPI
  INSTALL_COMPONENT USTMLIRPythonModules
  INSTALL_DESTINATION python_packages/ust_core/ust_mlir/_mlir_libs
  OUTPUT_DIRECTORY "${UST_MLIR_PYTHON_PACKAGES_DIR}/ust_core/ust_mlir/_mlir_libs"
  RELATIVE_INSTALL_ROOT "../../../.."
  DECLARED_HEADERS
    MLIRPythonCAPI.HeaderSources
  DECLARED_SOURCES
    ${_source_components}
)

add_mlir_python_modules(USTMLIRPythonModules
  ROOT_PREFIX "${UST_MLIR_PYTHON_PACKAGES_DIR}/ust_core/ust_mlir"
  INSTALL_PREFIX "python_packages/ust_core/ust_mlir"
  DECLARED_SOURCES ${_source_components}
  COMMON_CAPI_LINK_LIBS
    USTMLIRAggregateCAPI
  )

################################################################################
# Add setup script
################################################################################

add_custom_command(
          TARGET USTMLIRPythonModules POST_BUILD
          COMMAND ${CMAKE_COMMAND} -E copy
          ${UST_MLIR_PYTHON_ROOT_DIR}/../setup.py
          ${UST_MLIR_PYTHON_PACKAGES_DIR}/ust_core)
