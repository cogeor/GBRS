from pybind11.setup_helpers import Pybind11Extension, build_ext
from setuptools import setup

ext_modules = [
    Pybind11Extension(
        "risk_score",
        ["python/bindings.cpp"],  # or wherever your cpp file is
        include_dirs=[
            "inst/include",
            "third_party/eigen"
        ],
        extra_compile_args=["-O3", "-fopenmp"],
        extra_link_args=["-fopenmp"],  # 👈 Required to link the OpenMP runtime
        cxx_std=17,
    ),
]

setup(
    name="risk_score",
    version="0.1",
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
    zip_safe=False,
)