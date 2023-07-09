# -*- coding: utf-8 -*-
"""
Created on Mon Oct  3 20:20:34 2022

@author: Clayton
"""

from setuptools import find_packages, setup

setup(
    name='eventlogic',
    packages=find_packages('eventlogic'),
    version='0.1.0',
    description='Performs logical operations on event-style data.',
    author='Clayton Barnes',
    license='MIT',
    install_requires=['numpy','operator'],
    setup_requires=['pytest-runner'],
    tests_require=['pytest==4.4.1'],
    test_suite='tests',
)