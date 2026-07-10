from pathlib import Path

from setuptools import find_packages, setup

package_root = Path(__file__).resolve().parent
try:
    long_description = (package_root / "README.md").read_text(encoding="utf-8")
except FileNotFoundError:
    long_description = ""

setup(
    name="dynare-lsp",
    version="0.4.0",
    description="Language Server Protocol implementation for the Dynare modeling language",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="LLMacro",
    license="GPL-3.0-or-later",
    url="https://github.com/pdwhoward/LLMacro-Dynare-LSP",
    project_urls={
        "Issues": "https://github.com/pdwhoward/LLMacro-Dynare-LSP/issues",
    },
    packages=find_packages(),
    python_requires=">=3.11",
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
            "bin/*",
            "matlab/*.m",
            "oracle/*.m",
        ],
    },
    include_package_data=True,
    classifiers=[
        "Development Status :: 3 - Alpha",
        "License :: OSI Approved :: GNU General Public License v3 or later (GPLv3+)",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Mathematics",
        "Topic :: Text Editors :: Integrated Development Environments (IDE)",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Programming Language :: Python :: 3.13",
        "Programming Language :: Python :: 3.14",
    ],
)
