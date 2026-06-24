import os
import sys
import subprocess
import shutil
from setuptools import setup, Extension
import numpy

# Resolve absolute path to CrSDK libs to embed in rpath
project_dir = os.path.abspath(os.path.dirname(__file__))
sdk_lib_dir = os.path.join(project_dir, '3rd_party', 'CrSDK', 'lib')

# Compile src/color_conversion.cu using nvcc if available
nvcc_path = shutil.which('nvcc')
if not nvcc_path:
    for path in ['/usr/local/cuda/bin/nvcc', '/usr/local/cuda-12.6/bin/nvcc']:
        if os.path.exists(path):
            nvcc_path = path
            break

has_cuda = False
extra_objects = []
extra_compile_args = ['-std=c++17', '-Wall', '-Wextra', '-fsigned-char']
libraries = ['Cr_Core', 'raw', 'lcms2', 'pthread']
library_dirs = [sdk_lib_dir]

if nvcc_path:
    cuda_src = os.path.join(project_dir, 'src', 'color_conversion.cu')
    build_dir = os.path.join(project_dir, 'build')
    os.makedirs(build_dir, exist_ok=True)
    cuda_obj = os.path.join(build_dir, 'color_conversion_cuda.o')
    print(f"Compiling CUDA source: {cuda_src} ...")
    cmd = [nvcc_path, '-c', cuda_src, '-o', cuda_obj, '-O3', '-std=c++17', '-Xcompiler', '-fPIC']
    res = subprocess.run(cmd)
    if res.returncode == 0:
        has_cuda = True
        extra_objects.append(cuda_obj)
        extra_compile_args.append('-DHAVE_CUDA=1')
        libraries.append('cudart')
        
        # Resolve target lib paths
        for path in ['/usr/local/cuda/lib64', '/usr/local/cuda/lib', 
                     '/usr/local/cuda-12.6/targets/aarch64-linux/lib',
                     '/usr/local/cuda/targets/aarch64-linux/lib']:
            if os.path.exists(path):
                library_dirs.append(path)
        print("Successfully integrated CUDA into setup.py compilation.")
    else:
        print("Error: CUDA compilation failed.")
else:
    print("CUDA toolchain (nvcc) not found. Building CPU-only pipeline.")

module = Extension(
    'negicc_station',
    sources=[
        'src/python_bindings.cpp',
        'src/image_capture.cpp',
        'src/raw_processor.cpp',
        'src/sony_camera_session.cpp'
    ],
    include_dirs=[
        project_dir,
        os.path.join(project_dir, 'src'),
        os.path.join(project_dir, '3rd_party'),
        os.path.join(project_dir, '3rd_party', 'CrSDK', 'include'),
        '/usr/local/cuda/include',
        '/usr/local/cuda-12.6/include',
        numpy.get_include()
    ],
    library_dirs=library_dirs,
    libraries=libraries,
    extra_compile_args=extra_compile_args,
    extra_objects=extra_objects,
    extra_link_args=[
        '-Wl,-rpath,$ORIGIN',
        f'-Wl,-rpath,{sdk_lib_dir}'
    ]
)

setup(
    name='negicc_station',
    version='0.1',
    ext_modules=[module]
)
