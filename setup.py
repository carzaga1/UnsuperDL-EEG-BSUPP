from setuptools import find_packages, setup

setup(
    name='bscarlos',
    package_dir={'':'src'},
    packages=find_packages(where='src'),
    version='0.1.0',
    description='A short description of the project.',
    author='Carlos',
)