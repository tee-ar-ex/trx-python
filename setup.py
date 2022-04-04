import glob
import os

from setuptools import setup, find_packages

SCRIPTS = glob.glob("scripts/*.py")

REQUIRES = ['dipy>=1.2.0', 'nibabel>=3.0.0', 'PyYAML>=5.3.1']
setup(name='tractography_file_format',
      packages=find_packages(),
      setup_requires=REQUIRES,
      install_requires=REQUIRES,
      scripts=SCRIPTS)
