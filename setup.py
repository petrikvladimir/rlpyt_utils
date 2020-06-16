from setuptools import setup

setup(
    name='rlpyt_utils',
    version='1.3',
    description='Some utilities for rlpyt library.',
    author='Vladimir Petrik',
    author_email='vladimir.petrik@cvut.cz',
    packages=['rlpyt_utils', 'rlpyt_utils.samplers', 'rlpyt_utils.runners', 'rlpyt_utils.promp'],
    install_requires=['numpy', 'matplotlib'],
)
