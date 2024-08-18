from setuptools import setup

with open('requirements.txt') as f:
    required = f.read().splitlines()

setup(
    name='9DTact',
    version='1.0',
    packages=['data', 'model', 'force_estimation', 'shape_reconstruction'],
    url='https://linchangyi1.github.io/9DTact/',
    license='BSD-3-Clause',
    author='Changyi Lin',
    author_email='changyil@andrew.cmu.edu',
    description='Open source of 9DTact.',
    install_requires=required,
)
