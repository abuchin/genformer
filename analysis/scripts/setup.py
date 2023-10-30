from setuptools import Extension, setup, Distribution
from Cython.Build import cythonize
import numpy as np

Distribution().fetch_build_eggs(['Cython>=0.15.1', 'numpy>=1.10'])

ext_modules = [
    Extension(
        "cython_fxn",
        ["cython_fxn.pyx"],
        extra_compile_args=['-fopenmp'],
        extra_link_args=['-fopenmp'],
    )
]

setup(
    ext_modules = cythonize(ext_modules, annotate=True),
    include_dirs=[np.get_include()],
)