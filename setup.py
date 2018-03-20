from os import getenv
from setuptools import setup
from setuptools import find_packages


setup(
    name='ml-tools',
    version=getenv("VERSION", "LOCAL"),
    description='Contains tools for ML',
    packages=find_packages()
)
