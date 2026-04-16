import sys
import platform
from setuptools import setup, Extension

# Determine if the current platform is macOS on ARM
is_mac_arm = sys.platform == 'darwin' and platform.machine() in ['arm64', 'aarch64']

# Base arguments
if sys.platform == 'win32':
    compile_args = ['/O2', '/openmp', '/arch:AVX2', '/std:c++17']
    link_args = ['/openmp']
else:
    compile_args = ['-O3', '-std=c++17']
    # Add AVX2 for non-ARM macOS and Linux
    if not is_mac_arm:
        compile_args.append('-mavx2')
    # OpenMP is typically not available by default on macOS clang
    if sys.platform == 'linux':
        compile_args.append('-fopenmp')
        
    link_args = ['-fopenmp'] if sys.platform == 'linux' else []

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
        extra_compile_args=compile_args,
        extra_link_args=link_args,
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
