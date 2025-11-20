from pybind11.setup_helpers import Pybind11Extension, build_ext
from setuptools import setup
import sys

class BuildExt(build_ext):
    def build_extensions(self):
        compiler_type = self.compiler.compiler_type
        opts = []
        link_opts = []
        
        if compiler_type == 'msvc':
            opts = ['/O2', '/openmp', '/wd4244', '/wd4267', '/wd4018']
        else:
            opts = ['-O3', '-fopenmp']
            link_opts = ['-fopenmp']
            
        for ext in self.extensions:
            ext.extra_compile_args.extend(opts)
            ext.extra_link_args.extend(link_opts)
            
        build_ext.build_extensions(self)

ext_modules = [
    Pybind11Extension(
        "gbrs.core",
        ["python/bindings.cpp"], 
        include_dirs=[
            "inst/include",
            "third_party/eigen"
        ],
        cxx_std=17,
    ),
]

setup(
    name="gbrs",
    version="0.1.0",
    packages=["gbrs"],   
    package_dir={"": "python"}, 
    ext_modules=ext_modules,
    cmdclass={"build_ext": BuildExt},
    zip_safe=False,
)