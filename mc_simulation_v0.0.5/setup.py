#!/usr/bin/env python
# coding: utf-8

# In[ ]:
from distutils.core import setup
setup(
  name = 'mc_simulation',
  packages = ['mc_simulation'],
  version = '0.0.5',
  license='MIT',
  description = 'module to compute Monte Carlo simulations',
  author = 'Carlo Parodi',
  author_email = 'carlo.parodi91@gmail.com',
  url = 'https://github.com/Carlo-Parodi/prova.git',
  download_url = 'https://github.com/Carlo-Parodi/prova.git',
  keywords = ['SOME', 'MEANINGFULL', 'KEYWORDS'],
  install_requires=[
          'pandas',
          'numpy',
          'arch',
          'pandas',
          'numpy',
          'matplotlib',
          'statsmodels',
          'scipy',
      ],
  classifiers=[
    'Development Status :: 3 - Alpha',
    'Intended Audience :: Developers',
    'Topic :: Software Development :: Build Tools',
    'License :: OSI Approved :: MIT License',
    'Programming Language :: Python :: 3',
    'Programming Language :: Python :: 3.4',
    'Programming Language :: Python :: 3.5',
    'Programming Language :: Python :: 3.6',
  ],
)


