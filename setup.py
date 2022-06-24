import glob
from setuptools import setup

setup(name='trx',
      use_scm_version=True,
      scripts=glob.glob("scripts/*.py"))
