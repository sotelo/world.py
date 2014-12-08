import numpy
from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext

ext_modules = [Extension(
    name="wrap1",
    sources=["wrap1.pyx"],
    include_dirs = [numpy.get_include()],
    language="c++",
         libraries= ["world"]
    )]

setup(
    name = 'f',
    cmdclass = {'build_ext': build_ext},
    ext_modules = ext_modules,
    )

