add_mlir_library(MLIRUSTPasses
    LoopBoundConstant.cpp
    Passes.cpp

    ADDITIONAL_HEADER_DIRS
    ${PROJECT_SOURCE_DIR}/include/ust

    DEPENDS
    MLIRUSTPassesIncGen

    LINK_COMPONENTS
    Core

    LINK_LIBS PUBLIC
    MLIRIR
    MLIRPass
)