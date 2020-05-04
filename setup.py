from setuptools import setup
from Cython.Build import cythonize

setup(
    ext_modules = cythonize("cylibs/qvalues.pyx",
                            build_dir="build")
)
