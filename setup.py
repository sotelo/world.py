import numpy
from setuptools import setup, Extension
from Cython.Distutils import build_ext
import os

ext_modules = [Extension(
    name="world",
    sources=["world.pyx"],
    include_dirs = [numpy.get_include(),
                    os.path.join(os.getcwd(), 'lib/world/src/')],
    library_dirs = [os.path.join(os.getcwd(), 'lib/world/build/src/')],
    libraries = ['world'],
    extra_compile_args=['-fpermissive'],
    language="c++",
    )]

setup(
    name = 'world',
    cmdclass = {'build_ext': build_ext},
    ext_modules = ext_modules,
    )
