from setuptools import setup, Extension
import sys
import os

python_include = sys.executable.replace(
	'bin/python', 
	'include/python{}.{}'.format(
    sys.version_info.major,
    sys.version_info.minor
    )
)




cpp_extension = Extension(
    name="sample_opencl_lib",
    sources=[
        "src/opencllib/cpp/kernelmanager.cpp",
        "src/opencllib/cpp/operationmanager.cpp",
		"src/opencllib/python/bindings.cpp"
    ],
    include_dirs=[
        python_include,
        "src/opencllib/cpp"
    ],
	libraries=['OpenCL'],
    language="c++",
    extra_compile_args=[
        "-std=c++11",
        "-O3",
        "-Wall",
        "-Wno-unused-function",  # Python macros can generate unused functions
        "-fPIC"  # Position Independent Code is required
    ]
)


setup(
    name="sample_opencl_lib",
    version="0.1",
    ext_modules=[cpp_extension]
)