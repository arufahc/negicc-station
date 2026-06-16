import os
import sys
from setuptools import setup, Extension
import numpy

# Resolve absolute path to CrSDK libs to embed in rpath
project_dir = os.path.abspath(os.path.dirname(__file__))
sdk_lib_dir = os.path.join(project_dir, '3rd_party', 'CrSDK', 'lib')

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
        numpy.get_include()
    ],
    library_dirs=[
        sdk_lib_dir
    ],
    libraries=[
        'Cr_Core',
        'raw',
        'lcms2',
        'pthread'
    ],
    extra_compile_args=['-std=c++17', '-Wall', '-Wextra', '-fsigned-char'],
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
