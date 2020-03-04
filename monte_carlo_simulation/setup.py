#!/usr/bin/env python
# coding: utf-8

# In[ ]:
import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

requirements = [
    'arch',
    'pandas',
    'numpy',
    'matplotlib',
    'statsmodels',
    'scipy',
]

setuptools.setup(
    name="mc_simulation", # Replace with your own username
    version="0.0.3",
    author="Carlo Parodi",
    author_email="carlo.parodi91@gmail.com",
    description="A package for monte carlo simulations for time series",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Carlo-Parodi/monte_carlo_simulation",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: Free To Use But Restricted",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3.6"
    ],
    python_requires='>=3.6',
    install_requires = requirements
)

