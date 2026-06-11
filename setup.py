from setuptools import setup, find_packages

long_description = ""
try:
    with open("README.md") as f:
        long_description = f.read()
except FileNotFoundError:
    pass

setup(
    name="dynare-lsp",
    version="0.3.1",
    description="Language Server Protocol implementation for the Dynare modeling language",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="LLMacro",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "pygls>=1.0.0",
        "lsprotocol>=2023.0.0",
    ],
    extras_require={
        "solver": [
            "numpy>=1.20.0",
            "scipy>=1.7.0",
            "sympy>=1.12",
        ],
        "mcp": [
            "mcp>=0.1.0",
        ],
        "all": [
            "numpy>=1.20.0",
            "scipy>=1.7.0",
            "sympy>=1.12",
            "mcp>=0.1.0",
        ],
        "dev": [
            "pytest>=7.0",
            "pytest-cov",
            "numpy>=1.20.0",
            "scipy>=1.7.0",
            "sympy>=1.12",
        ],
    },
    entry_points={
        "console_scripts": [
            "dynare-lsp=dynare_lsp.__main__:main",
            "dynare-mcp=dynare_lsp.mcp_server:main",
        ],
    },
    package_data={
        "dynare_lsp": [
            "tests/fixtures/*.mod",
            "bin/*.exe",
            "matlab/*.m",
            "oracle/*.m",
        ],
    },
    include_package_data=True,
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Mathematics",
        "Topic :: Text Editors :: Integrated Development Environments (IDE)",
        "Programming Language :: Python :: 3",
    ],
)
