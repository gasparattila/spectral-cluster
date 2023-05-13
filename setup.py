from setuptools import setup
from pybind11.setup_helpers import Pybind11Extension

setup(
    py_modules=["spectral_cluster"],
    ext_modules=[
        Pybind11Extension(
            "_spectral_cluster_impl",
            ["_spectral_cluster_impl.cpp"],
            include_dirs=["lib/eigen", "lib/spectra/include"],
            cxx_std=17,
        )]
)
