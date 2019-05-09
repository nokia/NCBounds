#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""The setup script."""

from setuptools import setup, find_packages

README = ""
try:
    with open('README.rst') as readme_file:
        README = readme_file.read()
except:
    pass

HISTORY = ""
try:
    with open('HISTORY.rst') as history_file:
        history = history_file.read()
except:
    pass

requirements = ['Click>=6.0', 'numpy', 'cvxopt', 'scipy']

setup_requirements = ['pytest-runner', ]

test_requirements = ['pytest', ]

setup(
    author="Anne Bouillard",
    author_email='anne.bouillard@nokia-bell-labs.com',
    description="Deterministic Performances Bounds with Network Calculus",
    install_requires=requirements,
    license="BSD-3",
    long_description=README + '\n\n' + HISTORY,
    keywords='NCBounds',
    name='NCBounds',
    packages=find_packages(include=['NCBounds']),
    setup_requires=setup_requirements,
    test_suite='tests',
    tests_require=test_requirements,
    url='https://github.com/nokia/NCBounds',
    version='0.1.0',
    zip_safe=False,
)
