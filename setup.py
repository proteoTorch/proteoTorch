#!/usr/bin/env python3
import sys

if (sys.version_info[0] == 3 and sys.version_info[1] < 0):
    print("PRISM requires Python version 3.0 or later")
    sys.exit(1)

from setuptools import setup, find_packages
from Cython.Build import cythonize
from os import path
import subprocess

DISTNAME = 'ProteoTorch'
VERSION = '0.1.0'
DESCRIPTION = 'Deep semi-supervised learning for identification of shotgun proteomics data'
# with open('README.md') as f_in:
#     LONG_DESCRIPTION = f_in.read()
AUTHOR = 'John T. Halloran, Gregor Urban'
AUTHOR_EMAIL = 'johnhalloran321@gmail.com, gur9000@outlook.com'
URL = 'https://github.com/johnhalloran321/proteoTorch'
LICENSE='OSL-3.0'

CLASSIFIERS = ["Natural Language :: English",
               "Development Status :: 3 - Alpha",
               "Intended Audience :: Science/Research",
               "License :: OSI Approved :: The Open Software License 3.0 "
               "(OSL-3.0)",
               "Topic :: Scientific/Engineering :: Bio-Informatics",
               "Operating System :: MacOS",
               "Operating System :: Microsoft :: Windows",
               "Operating System :: Unix",
               "Programming Language :: Python :: 3 :: Only"]

def build_solvers():
    """ Check if the solver library has been built.  If not, run
        make in the solvers directory
    """
    if not path.exists(path.join('proteoTorch_solvers', 'libssl.so')):
        try:
            subprocess.check_call(['make'], cwd='proteoTorch_solvers')
        except:
            print("Could not build ProteoTorch SVM solver library")

def main():
    build_solvers()
    setup(
        name=DISTNAME,
        version=VERSION,
        author=AUTHOR,
        author_email=AUTHOR_EMAIL,
        description=DESCRIPTION,
        license=LICENSE,
        packages=find_packages(include=["proteoTorch", "proteoTorch.*, proteoTorch_solvers"]),
        url=URL,
        platforms=['any'],
        # install_requires=[
        #     'numpy', 
        #     'sklearn', 
        #     'torch'
        # ],
        classifiers=CLASSIFIERS,
        ext_modules = cythonize("proteoTorch/cylibs/proteoTorch_qvalues.pyx",
                                build_dir="build")
    )

if __name__ == "__main__":
    main()
