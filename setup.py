from pybind11.setup_helpers import Pybind11Extension, build_ext
from setuptools import setup

ext_modules = [
    Pybind11Extension(
        "gbrs.core",
        ["python/bindings.cpp"], 
        include_dirs=[
            "inst/include",
            "third_party/eigen"
        ],
        extra_compile_args=["-O3", "-fopenmp"],
        extra_link_args=["-fopenmp"], 
        cxx_std=17,
    ),
]

setup(
    name="gbrs",
    version="0.1",
    packages=["gbrs"],   
    package_dir={"": "python"}, 
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
    zip_safe=False,
)