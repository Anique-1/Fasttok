from setuptools import setup, Extension
import sys
import platform
import os

class get_pybind_include(object):
    def __init__(self, user=False):
        self.user = user

    def __str__(self):
        import pybind11
        return pybind11.get_include(self.user)

# Determine if the architecture supports AVX2 (x86/x64)
machine = platform.machine().lower()
is_x86 = machine in ['x86_64', 'amd64', 'x86', 'i386', 'i686']

# In macOS cross-compilation environments (like cibuildwheel), ARCHFLAGS dictates the target
archflags = os.environ.get('ARCHFLAGS', '')
if 'arm64' in archflags:
    is_x86 = False

# Configure compile arguments based on platform and architecture
if sys.platform == 'win32':
    compile_args = ['/O2', '/openmp', '/std:c++17']
    link_args = ['/openmp']
    if is_x86:
        compile_args.append('/arch:AVX2')
else:
    compile_args = ['-O3', '-std=c++17']
    link_args = []
    
    # macOS clang doesn't generally need or support -mavx2 well during universal2/arm64 cross-compilation
    if is_x86 and sys.platform != 'darwin':
        compile_args.append('-mavx2')
        
    if sys.platform == 'linux':
        compile_args.append('-fopenmp')
        link_args.append('-fopenmp')

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
    version='0.1.6',
    author='Anique',
    author_email='muhammadanique81@gmail.com',
    description='A high-performance C++ powered tokenizer and compressor for LLMs',
    packages=['fasttok'],
    ext_modules=ext_modules,
    install_requires=['pybind11>=2.4'],
    setup_requires=['pybind11>=2.4'],
    zip_safe=False,
)
