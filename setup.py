from setuptools import setup

setup(
    name='rlpyt_utils',
    version='1.1',
    description='Some utilities for rlpyt library.',
    author='Vladimir Petrik',
    author_email='vladimir.petrik@cvut.cz',
    packages=['rlpyt_utils', 'rlpyt_utils.samplers'],
    install_requires=['numpy', 'matplotlib'],
)
