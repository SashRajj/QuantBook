"""
Build script for the qbexec_cpp C++ extension module.

The extension is an additive optimisation: the Python code falls back
to a pure-Python implementation if the module is not present, so
`pip install -e .` is optional for development. The setuptools route
(no cmake) is the cleanest way to handle macOS clang and Linux gcc
uniformly via pybind11.setup_helpers.

Usage:
    pip install pybind11
    pip install -e .
"""

from pathlib import Path

from pybind11.setup_helpers import Pybind11Extension, build_ext
from setuptools import setup

HERE = Path(__file__).parent
CPP_DIR = HERE / "cpp"

ext_modules = [
    Pybind11Extension(
        "qbexec_cpp",
        sources=[
            str(CPP_DIR / "bindings.cpp"),
            str(CPP_DIR / "paper_broker.cpp"),
            str(CPP_DIR / "risk_gate.cpp"),
        ],
        include_dirs=[str(CPP_DIR)],
        cxx_std=17,
    ),
]

setup(
    name="qbexec_cpp",
    version="0.1.0",
    description="C++ core for quantresearch execution layer",
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
    zip_safe=False,
    python_requires=">=3.9",
)
