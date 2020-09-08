import os

from setuptools import setup, find_packages

REQUIRES = ['dipy==1.1.0', 'nibabel==3.0.0']
setup(name='tractography_file_format',
      packages=find_packages(),
      setup_requires=REQUIRES,
      install_requires=REQUIRES)
