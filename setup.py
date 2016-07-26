#!/usr/bin/env python3

from distutils.core import setup

setup(
    name='sht',
    version='1.0',
    description='A fast spherical harmonic transform implementation',
    author='Praveen Venkatesh',
    url='https://github.com/praveenv253/sht',
    packages=['sht', ],
    install_requires=['numpy', 'scipy', 'matplotlib', ],
    license='MIT',
)
