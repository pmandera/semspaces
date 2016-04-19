#!/usr/bin/env python
# -*- encoding: utf-8 -*-
from setuptools import setup, find_packages

setup(
    name = "semspaces",
    version = "0.1.2",
    packages = find_packages(),

    install_requires = ['fs>=0.5.0',
                        'numpy>=1.9.1',
                        'scipy>=0.14.0',
                        'pandas>=0.15.1',
                        'scikit-learn>=0.15.0'],

    # metadata for upload to PyPI
    author = "Paweł Mandera",
    author_email = "pawel.mandera@ugent.be",
    description = "This is a package for working with semantic spaces.",
    keywords = "semantic space word vectors",

    scripts = ['bin/w2v2ssm', 'bin/csv_sims', 'bin/subset_space']
)
