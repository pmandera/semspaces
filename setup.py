#!/usr/bin/env python
# -*- encoding: utf-8 -*-
from setuptools import setup, find_packages

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name="semspaces",
    version = "0.1.6",
    packages=find_packages(),
    scripts=['bin/w2v2ssm', 'bin/csv_sims', 'bin/subset_space'],
    install_requires=['fs>=0.5.0',
                       'numpy>=1.22.4',
                       'scipy>=1.5.0',
                       'pandas>=1.4.2',
                       'scikit-learn>=1.1.1'],

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
