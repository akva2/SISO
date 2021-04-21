#!/usr/bin/env python

from pathlib import Path
from setuptools import setup
from distutils.extension import Extension

with open(Path(__file__).parent / 'README.rst') as f:
    desc = f.read()

setup(
    name='SISO',
    version='2.2.0',
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
        'numpy',
        'Splipy>=1.4',
        'lrsplines>=1.5',
        'h5py',
        'vtk',
        'netcdf4',
        'nptyping',
        'pyerfa',
        'singledispatchmethod',
        'treelog',
    ],
    extras_require={
        'VTF': ['vtfwriter'],
        'autodiff': ['jax', 'jaxlib'],
        'testing': ['pytest'],
        'deploy': ['twine', 'cibuildwheel==1.1.0'],
        'geotiff': ['gdal'],
    },
    entry_points={
        'console_scripts': [
            'ifem-to-vt=siso.__main__:deprecated',
            'siso=siso.__main__:convert'
        ],
    },
)
