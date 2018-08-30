import unittest

from semspaces.space import SemanticSpace

import numpy as np
import scipy.sparse

dense_space = np.array([[0.61502426,  0.35800892,  0.46591138],
                 [0.00000000,  0.80705953,  0.87805124],
                 [0.18189868,  0.37707662,  0.89973192],
                 [0.32667934,  0.0994168 ,  0.75457225],
                 [0.43300126,  0.17586539,  0.88097073],
                 [0.62085788,  0.29817756,  0.62991792],
                 [0.37163458,  0.86633926,  0.31679958],
                 [0.37416635,  0.82935107,  0.34275204],
                 [0.26996958,  0.57101081,  0.60706083],
                 [0.36690094,  0.70666147,  0.3300295 ],
                 [0.19479401,  0.3334173 ,  0.79296408]])

sparse_space = scipy.sparse.csr_matrix(dense_space)

space = dense_space

rows = ['first', 'second', 'third', 'fourth', 'fifth', 'sixth'
        'seventh', 'eighth', 'ninth', 'tenth', 'eleventh']
columns = ['one', 'two', 'three']
readme_title = 'Random semantic space'
readme_desc = 'This semantic space was genarated for demonstration.'


class TestSemanticSpace(unittest.TestCase):
    def setUp(self):
        self.semspace = SemanticSpace(space, rows, columns, readme_title,
                                      readme_desc)

    def test_defined_at_words(self):
        assert self.semspace.defined_at('first')
        assert not self.semspace.defined_at('twelfth')
        assert not self.semspace.defined_at('one')

    def test_defined_at_seqs(self):
        assert self.semspace.defined_at(['first', 'second'])
        assert not self.semspace.defined_at(['first', 'twelfth'])
        assert not self.semspace.defined_at(['one', 'twelfth'])

    def test_similarity_pairs(self):
        self.assertAlmostEqual(self.semspace.pair_distance('first', 'first'), 0)
        assert self.semspace.pair_distance('first', 'second') > 1e-10

    def test_prenormalization(self):
        semspace = SemanticSpace(space, rows, columns, readme_title,
                                 readme_desc, prenorm=True)
        for row in semspace.vectors:
            row_norm = np.linalg.norm(row)
            print(row_norm, row)
            self.assertAlmostEqual(row_norm, 1.0)

    def test_prenorm_exception_on_non_cosine(self):
        semspace = SemanticSpace(space, rows, columns, readme_title,
                                 readme_desc, prenorm=True)

        # cosine should work on prenormalized space
        semspace.pair_distance('first', 'second', metric='cosine')

        # but not euclidean
        with self.assertRaises(Exception):
            semspace.pair_distance('first', 'second', metric='euclidean')

    def test_cosine_equals_prenorm(self):
        semspace = SemanticSpace(space, rows, columns, readme_title,
                                 readme_desc, prenorm=True)

        cosine_non_prenorm = self.semspace.pair_distance('first', 'second',
                                                         metric='cosine')
        cosine_prenorm = semspace.pair_distance('first', 'second',
                                                metric='cosine')
        self.assertEqual(cosine_non_prenorm, cosine_prenorm)

        pairs = [('first', 'second'), ('third', 'eighth'),
                 ('twelfth', 'first'), ('twelfth', 'thirteenth')]

        pairs_sims_non_prenorm = self.semspace.pair_distances(pairs)
        pairs_sims_prenorm = semspace.pair_distances(pairs)

        self.assertEqual(pairs_sims_non_prenorm, pairs_sims_prenorm)

    def test_subset(self):
        words = ['third', 'second', 'tenth', 'eighth']
        subset = self.semspace.subset(words)
        assert subset.vectors.shape == (4, 3)
        assert not subset.defined_at('first')
        assert subset.defined_at('second')
        assert subset.defined_at('third')
        assert subset.defined_at('eighth')
        assert not subset.defined_at('ninth')
        assert subset.defined_at('tenth')
        self.assertAlmostEqual(
            self.semspace.pair_distance('second', 'third'),
            subset.pair_distance('second', 'third'))
        self.assertAlmostEqual(
            self.semspace.pair_distance('third', 'tenth'),
            subset.pair_distance('third', 'tenth'))

    def test_pair_distances(self):
        pairs = [('first', 'second'), ('third', 'eighth'),
                 ('twelfth', 'first'), ('twelfth', 'thirteenth')]
        pairs_sims = self.semspace.pair_distances(pairs)

        first_second = self.semspace.pair_distance('first', 'second')
        assert pairs_sims[('first', 'second')] == first_second

        assert ('third', 'eighth') in list(pairs_sims.keys())
        assert ('twelfth', 'first') not in list(pairs_sims.keys())
        assert ('twelfth', 'thirteenth') not in list(pairs_sims.keys())

        pairs_sims_nan = self.semspace.pair_distances(pairs, na_val=True)

        assert pairs_sims_nan[('first', 'second')] == first_second

        assert ('third', 'eighth') in list(pairs_sims_nan.keys())
        assert ('twelfth', 'first') in list(pairs_sims_nan.keys())
        assert ('twelfth', 'thirteenth') in list(pairs_sims_nan.keys())

        assert pairs_sims_nan[('twelfth', 'first')] is np.nan
        assert pairs_sims_nan[('twelfth', 'thirteenth')] is np.nan

    def test_vector_entropy(self):
        self.assertAlmostEqual(
            self.semspace.vector_entropy('first'),
            1.5502257500054266)
        self.assertAlmostEqual(
            self.semspace.vector_entropy('fifth'),
            1.3302170534376188)
        self.assertAlmostEqual(
            self.semspace.vector_entropy('second'),
            0.99871934706694587)

    def test_allowed_metrics_when_prenormed(self):
        semspace = SemanticSpace(space, rows, columns, readme_title,
                                 readme_desc, prenorm=False)
        semspace_p = SemanticSpace(space, rows, columns, readme_title,
                                   readme_desc, prenorm=True)
        self.assertIn('cosine', semspace.allowed_metrics())
        self.assertIn('cosine', semspace_p.allowed_metrics())

        self.assertNotIn('manhattan', semspace_p.allowed_metrics())
        self.assertIn('manhattan', semspace.allowed_metrics())

    def test_metrics(self):
        semspace = SemanticSpace(space, rows, columns, readme_title,
                                 readme_desc, prenorm=False)
        for metric in semspace.allowed_metrics():
            print(metric)
            pairs = [('first', 'second'), ('third', 'eighth'),
                     ('twelfth', 'first'), ('twelfth', 'thirteenth')]
            self.semspace.pair_distances(pairs, metric=metric)

    def test_combined_vector(self):
        semspace = SemanticSpace(space, rows, columns, readme_title,
                                 readme_desc, prenorm=False)
        vector = semspace.combined_vector(['first', 'third'])
        assert (space[[0,2], :].sum(0) == vector).all()

        vector = semspace.combined_vector(['second', 'fourth'])
        assert (space[[1,3], :].sum(0) == vector).all()

    def test_combined_vector_prenorm(self):
        semspace = SemanticSpace(space, rows, columns, readme_title,
                                 readme_desc, prenorm=True)
        vector = semspace.combined_vector(['first', 'third'])
        self.assertEqual(np.linalg.norm(vector), 1)

    def test_combined_vector_sparse(self):
        semspace = SemanticSpace(sparse_space, rows, columns, readme_title,
                                 readme_desc, prenorm=False)
        vector = semspace.combined_vector(['first', 'third'])
        assert (space[[0,2], :].sum(0) == vector).all()

        vector = semspace.combined_vector(['second', 'fourth'])
        assert (space[[1,3], :].sum(0) == vector).all()
