import numpy
from Cython.Build import cythonize
from setuptools import Extension, setup

extensions = [
    Extension(
        "msca.integrate.cython_utils",
        ["src/msca/integrate/cython_utils.pyx"],
        include_dirs=[numpy.get_include()],
    )
]

setup(
    name="msca",
    ext_modules=cythonize(
        extensions,
        compiler_directives={"language_level": 3, "profile": False},
    ),
)
