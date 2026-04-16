from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext
import sys
import setuptools

class get_pybind_include(object):
    def __init__(self, user=False):
        self.user = user

    def __str__(self):
        import pybind11
        return pybind11.get_include(self.user)

ext_modules = [
    Extension(
        'fasttok.fasttok_core',
        ['src/main.cpp'],
        include_dirs=[
            get_pybind_include(),
            get_pybind_include(user=True),
            'include/'
        ],
        extra_compile_args=['/O2', '/openmp', '/arch:AVX2', '/std:c++17'] if sys.platform == 'win32' else 
                           (['-O3', '-mavx2', '-std=c++17'] + (['-fopenmp'] if sys.platform == 'linux' else [])),
        extra_link_args=['/openmp'] if sys.platform == 'win32' else 
                        (['-fopenmp'] if sys.platform == 'linux' else []),
        language='c++'
    ),
]

setup(
    name='fasttok',
    version='0.1.1',
    author='Anique',
    author_email='muhammadanique81@gmail.com',
    description='A high-performance C++ powered tokenizer and compressor for LLMs',
    packages=['fasttok'],
    ext_modules=ext_modules,
    install_requires=['pybind11>=2.4'],
    setup_requires=['pybind11>=2.4'],
    zip_safe=False,
)
