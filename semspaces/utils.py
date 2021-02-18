#!/usr/bin/env python3

import numpy as np
import sklearn.decomposition as sd


def subtract_mean_vec(vectors):
    return vectors - vectors.mean(axis=0)


def subtract_top_components(vectors, d=None):
    """Subtract d top PCA components."""
    pca = sd.PCA().fit(vectors)

    for cn in range(d):
        component = pca.components_[cn, :]
        weights = np.array([component.dot(vectors.T)])
        vectors = vectors - weights.T.dot(np.array([component]))

    return vectors


def postprocess(vectors, d=None):
    """
    Postprocess vectors following:

    Jiaqi Mu, Suma Bhat, Pramod Viswanath. 2017.
    All-but-the-Top: Simple and Effective Postpro-cessing for Word
    Representations.https://arxiv.org/abs/1702.01417
    """

    if d is None:
        # this is the default recommended in the paper
        d = int(vectors.shape[1]/100)

    vectors = subtract_mean_vec(vectors)
    vectors = subtract_top_components(vectors, d=d)

    return vectors
