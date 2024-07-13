import setuptools

def setup():
    setuptools.setup(
        name="ust_mlir",
        description="UST-MLIR: UST Dialect for Heterogeneous Computing",
        version="0.1",
        author="UST-MLIR Developers",
        setup_requires=[],
        install_requires=[],
        packages=setuptools.find_packages(),
        url="https://github.com/cmx-Y/ust-dialect.git",
        python_requires=">=3.6",
        classifiers=[
            "Programming Language :: Python :: 3.6",
            "Programming Language :: Python :: 3.7",
            "Programming Language :: Python :: 3.8",
            "Programming Language :: Python :: 3.9",
            "Programming Language :: Python :: 3.10",
            "Topic :: Scientific/Engineering :: Accelerator Design",
            "Operating System :: OS Independent",
        ],
        zip_safe=True,
    )


if __name__ == "__main__":
    setup()
