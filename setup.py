#!/usr/bin/env python
# -*- encoding: utf-8 -*-
from setuptools import setup, find_packages

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name="semspaces",
    version="0.1.3",
    packages=find_packages(),
    scripts=['bin/w2v2ssm', 'bin/csv_sims', 'bin/subset_space'],

    install_requires=['fs>=0.5.0',
                       'numpy>=1.9.1',
                       'scipy>=0.14.0',
                       'pandas>=0.15.1',
                       'scikit-learn>=0.15.0'],


    # metadata for upload to PyPI
    author="Paweł Mandera",
    author_email="pawel@pawelmandera.com",
    description="Package for working with semantic spaces.",
    long_description_content_type="text/markdown",
    long_description=long_description,
    url="https://github.com/pmandera/semspaces/",
    keywords="semantic space word vectors",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
    ],
)
