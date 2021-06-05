#!/usr/bin/env python

from pathlib import Path
from setuptools import setup
from distutils.extension import Extension

with open(Path(__file__).parent / 'README.rst') as f:
    desc = f.read()

setup(
    name='SISO',
    version='2.1.1',
    description='Convert between different mesh data formats',
    long_description_content_type='text/x-rst',
    long_description=desc,
    maintainer='Eivind Fonn',
    maintainer_email='eivind.fonn@sintef.no',
    packages=['siso', 'siso.writer', 'siso.reader', 'siso.coords'],
    install_requires=[
        'cachetools',
        'click>=8',
        'dataclasses',
        'f90nml',
        'h5py',
        'lrsplines>=1.5',
        'netcdf4',
        'nptyping',
        'numpy',
        'pyerfa',
        'singledispatchmethod',
        'splipy>=1.4',
        'treelog',
        'vtk',
    ],
    extras_require={
        'VTF': ['vtfwriter'],
        'autodiff': ['jax', 'jaxlib'],
        'testing': ['pytest'],
        'deploy': ['twine', 'cibuildwheel==1.1.0'],
    },
    entry_points={
        'console_scripts': [
            'ifem-to-vt=siso.__main__:deprecated',
            'siso=siso.__main__:convert'
        ],
    },
)
