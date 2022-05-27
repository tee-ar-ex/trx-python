import glob
import os

from setuptools import setup, find_packages

TRX_MINIMAL_INSTALL = os.environ.get('TRX_MINIMAL_INSTALL')
TRX_MINIMAL_INSTALL = False if TRX_MINIMAL_INSTALL is None else \
    int(TRX_MINIMAL_INSTALL)

if TRX_MINIMAL_INSTALL:
    SCRIPTS = None
    REQUIRES = ['numpy>=1.20.*', 'nibabel>=3.*', 'gdown>=4.*',
                'pytest>=7.*', 'pytest-console-scripts>=0.*']
else:
    SCRIPTS = glob.glob("scripts/*.py")
    REQUIRES = ['fury@git+https://github.com/frheault/fury.git@b13f573#egg=fury',
                'dipy@git+https://github.com/frheault/dipy.git@f36f7e8#egg=dipy',
                'gdown>=4.*', 'pytest>=7.*', 'pytest-console-scripts>=0.*']

setup(name='trx',
      packages=find_packages(),
      setup_requires=['packaging>=19.0'],
      install_requires=REQUIRES,
      scripts=SCRIPTS)
