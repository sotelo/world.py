import numpy
from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
import os

ext_modules = [Extension(
    name="wrap1",
    sources=["wrap1.pyx"],
    include_dirs = [numpy.get_include(),
                    os.path.join(os.getcwd(), 'lib/world/src/')],
    library_dirs = [os.path.join(os.getcwd(), 'lib/world/build/src/')],
    libraries = ['world'],
    language="c++",
    )]

setup(
    name = 'f',
    cmdclass = {'build_ext': build_ext},
    ext_modules = ext_modules,
    )
