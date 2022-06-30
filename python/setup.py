from distutils.core import setup

from setuptools import find_packages

setup(name='multi_label_classification',
      version='1.0',
      description='multi label image classification',
      packages=find_packages("."),
      zip_safe=False,
     )