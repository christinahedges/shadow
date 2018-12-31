#!/usr/bin/env python
import os
import sys
from setuptools import setup

if "testpublish" in sys.argv[-1]:
    os.system("python setup.py sdist upload -r pypitest")
    sys.exit()
elif "publish" in sys.argv[-1]:
    os.system("python setup.py sdist upload -r pypi")
    sys.exit()

# Load the __version__ variable without importing the package
exec(open('shadow/version.py').read())

entry_points = {'console_scripts':
                ['shadow = shadow.ui:shadow']}

setup(name='shadow',
      version=__version__,
      description="Creates movies if Kepler data based on target names. "
                  "Uses a custom data format.",
      long_description=open('README.rst').read(),
      author='Christina Hedges',
      author_email='christina.l.hedges@nasa.gov',
      license='MIT',
      packages=['shadow'],
      package_data={'shadow': ['data/*.csv']},
      install_requires=['astropy>=0.4',
                        'numpy',
                        'pandas', 'fitsio'],
      entry_points=entry_points,
      classifiers=[
          "Development Status :: 3 - Alpha",
          "License :: OSI Approved :: MIT License",
          "Operating System :: OS Independent",
          "Programming Language :: Python :: 2.7",
          "Programming Language :: Python :: 3",
          "Intended Audience :: Science/Research",
          "Topic :: Scientific/Engineering :: Astronomy",
      ],
      )
