import os
from setuptools import setup, Extension
from Cython.Build import cythonize

# os.environ["CC"] = "g++"

setup(
    ext_modules = cythonize("cylibs/qvalues.pyx",
                            build_dir="build")
)
# setup(
#     ext_modules = cythonize(Extension("qvalues",
#                                       sources=["cylibs/qvalues.pyx"],
#                                       language="c++",
#                                       extra_compile_args=["-O3"]), 
#                             build_dir="build")
# )
