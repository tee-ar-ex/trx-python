import glob

from setuptools import setup, find_packages

SCRIPTS = glob.glob("scripts/*.py")
REQUIRES = ['fury@git+https://github.com/frheault/fury.git@5059a529#egg=fury',
            'dipy@git+https://github.com/frheault/dipy.git@4e192c5c6#egg=dipy']

setup(name='tractography_file_format',
      packages=find_packages(),
      setup_requires=['packaging>=19.0'],
      install_requires=REQUIRES,
      scripts=SCRIPTS)
