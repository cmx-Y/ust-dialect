add_mlir_library(MLIRUSTPasses
    LoopBoundConstant.cpp
    SetSparseInfo.cpp
    Passes.cpp

    ADDITIONAL_HEADER_DIRS
    ${PROJECT_SOURCE_DIR}/include/ust

    DEPENDS
    MLIRUSTPassesIncGen
    MLIRUSTAttrsIncGen
    MLIRUSTOpsIncGen

    LINK_COMPONENTS
    Core

    LINK_LIBS PUBLIC
    MLIRUST
    MLIRIR
    MLIRPass
)