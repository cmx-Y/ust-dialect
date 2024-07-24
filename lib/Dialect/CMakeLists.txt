
add_mlir_dialect_library(MLIRUST
        USTDialect.cpp

        ADDITIONAL_HEADER_DIRS
        ${PROJECT_SOURCE_DIR}/include/ust

        DEPENDS
        MLIRUSTOpsIncGen
        MLIRUSTAttrsIncGen

	LINK_LIBS PUBLIC
	MLIRIR
	)

