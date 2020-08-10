"""
Module for working with semantic spaces in Python.
"""

import numpy as np
import scipy.sparse
import scipy.stats
import sklearn.metrics.pairwise as smp

try:
    import pandas as pd
except ImportError:
    print('Warning: pandas not available. Exporting to pandas will not work.')

from . import space_io

metrics_sklearn = ['cosine', 'euclidean', 'manhattan', 'cityblock', 'l1', 'l2']

metrics = metrics_sklearn


class SemanticSpace(object):
    """
    SemanticSpace is a wrapper for a matrix containing distributional
    semantics vectors.

    It provides a set of convenience methods that allow to compute similarity
    scores for pairs of words, words in semantic neighbourhood, perform
    arithmetic calculations involving word vectors.

    SemanticSpace also implements easy methods for loading and saving the
    semantic spaces.
    """
    def __init__(self, vectors, words, cols=None,
                 readme_title='', readme_desc='', prenorm=False):
        if prenorm:
            self.vectors = smp.normalize(vectors)
        else:
            self.vectors = vectors
        self.prenorm = prenorm
        self.words = words
        self.cols = cols
        self.title = readme_title
        self.readme = readme_desc
        self._incl_words = set(words)
        self.shape = self.vectors.shape

        self.word2id = dict(zip(words, range(len(words))))

    def included_words(self):
        """All included words."""
        return self._incl_words

    def defined_at(self, element):
        """Check if element is in the space."""
        if isinstance(element, str):
            return element in self._incl_words
        elif isinstance(element, list):
            for word in element:
                if word not in self._incl_words:
                    return False
            return True

    def allowed_metrics(self):
        if self.prenorm is False:
            return metrics
        else:
            return ['cosine']

    def subset(self, words):
        """Subset a space to contain only specified words."""

        assert len(words) == len(set(words))

        words = list(words)
        rows = self._row_nums(words)
        new_space = self.vectors[rows, :]
        nrows, ncols = new_space.shape
        orig_nrows, orig_ncols = self.vectors.shape
        readme_title = '%s (subset)' % self.title
        readme_addition = '-- subset %s out of %s original rows--' % (
            nrows,
            orig_nrows)
        readme_desc = '%s\r\n%s' % (self.readme, readme_addition)
        return SemanticSpace(new_space, words, self.cols, readme_title,
                             readme_desc)

    # Input/Output methods

    @classmethod
    def from_ssmarket(cls, fname, prenorm=False):
        """Load a semantic space from semantic space market format."""
        if fname[-4:] == '.zip':
            ssmarket = space_io.ExternalZipSemanticSpaceMarket(fname, 'r')
        else:
            ssmarket = space_io.DirSemanticSpaceMarket(fname, 'r')
        data, rows, cols, r_t, r_d = ssmarket.read_all()
        ssmarket.close()
        return cls(data, rows, cols, r_t, r_d, prenorm=prenorm)

    @classmethod
    def from_csv(cls, fname, prenorm=False, dtype='float64'):
        """Read a semantic space from a CSV file."""
        words, vectors, title, readme = space_io.CSVReader.read(fname,
                                                                dtype=dtype)
        return cls(vectors, words, None, title, readme, prenorm=prenorm)

    def save_ssmarket(self, fname):
        """Save in a semantic space format."""
        ssmarket = space_io.SemanticSpaceMarket(fname, 'w')
        ssmarket.write_all(self.vectors, self.words, self.cols,
                           self.title, self.readme)
        ssmarket.close()

    def save_csv(self, fname, compress=True):
        """Save in a CSV format."""
        space_io.CSVWriter.write(fname, self, compress)

    def to_pandas(self):
        """Return as pandas DataFrame."""
        if scipy.sparse.issparse(self.vectors):
            print('Sparse matrices cannot currently be converted to pandas.')
            return None
        if self.cols is None:
            return pd.DataFrame(self.vectors, self.words)
        else:
            return pd.DataFrame(self.vectors, self.words, self.cols)

    # Similarity methods

    def word_most_similar(self, word, l2=None, n=10, metric='cosine'):
        """Return most similar to a word."""
        sims = self.most_similar([word], l2, n, metric)
        return sims[word]

    def most_similar(self, l1, l2=None, n=10, metric='cosine'):
        """Return distance matrix with distances between pairs of words."""
        if not l1:
            return None

        if l2 is None:
            sims = self.all_distances(l1, metric=metric)
        else:
            sims = self.matrix_distances(l1, l2, metric=metric)

        most_similar = {}
        sim_cols = sims.columns

        for word, neighbours in sims.iterrows():
            neigh_indexes = neighbours.argsort().values[:n]
            neighs = sim_cols[neigh_indexes]
            neighs_dist = [float(d) for d in neighbours.values[neigh_indexes]]
            most_similar[word] = list(zip(neighs, neighs_dist))

        return most_similar

    def pairwise_distances(self, X, Y=None, metric='cosine',
                           n_jobs=1, **kwds):

        if self.prenorm:
            if metric == 'cosine':
                return self._cosine_distances_prenorm(X, Y)
            else:
                raise Exception(
                    'Vectors are normalized and will work only with cosine.')

        return smp.pairwise_distances(X, Y, metric=metric,
                                      n_jobs=n_jobs, **kwds)

    def all_distances(self, l1, metric='cosine'):
        """Return distance matrix with distances to all words."""

        l1_vecs = self.word_vectors_matrix(l1)
        l1_labels = [self.label(e) for e in l1]

        sims = self.pairwise_distances(l1_vecs, self.vectors, metric=metric)

        return pd.DataFrame(sims, l1_labels, self.words)

    def pair_distance(self, w1, w2, metric='cosine'):
        """Calculate distance between two words."""

        distance = self.pairwise_distances(
            self.get_vector(w1),
            self.get_vector(w2), metric=metric)

        return distance[0, 0]

    def pair_distances(self, pairs_list, metric='cosine', na_val=False):
        """
        Calculate pairs of distances based on a list of tuples with pairs of
        words.

        If na_val is True assign np.nan to that value.
        """

        distances = {}

        for w1, w2 in pairs_list:
            if self.defined_at(w1) and self.defined_at(w2):
                distance = self.pair_distance(w1, w2, metric)
                distances[(w1, w2)] = distance
            elif na_val:
                distances[(w1, w2)] = np.nan

        return distances

    def matrix_distances(self, l1, l2=None, metric='cosine'):
        """Return distance matrix with distances between pairs of words."""

        l1_vecs = self.word_vectors_matrix(l1)
        l1_labels = [self.label(e) for e in l1]

        if l2 is None:
            sims = self.pairwise_distances(l1_vecs, metric=metric)
            l2 = l1
        else:
            l2_vecs = self.word_vectors_matrix(l2)
            l2_labels = [self.label(e) for e in l2]
            sims = self.pairwise_distances(l1_vecs, l2_vecs, metric=metric)

        return pd.DataFrame(sims, l1_labels, l2_labels)

    def _cosine_distances_prenorm(self, X, Y):
        """
        Return cosine distances based on a prenormalized vectors.

        It allows for much faster computation of cosine distances.
        """
        if not self.prenorm:
            raise Exception(
                'Vectors must be prenormalized!')
        if Y is None:
            Y = X
        X, Y = smp.check_pairwise_arrays(X, Y)
        sims = X.dot(Y.T)

        if scipy.sparse.issparse(sims):
            sims = sims.todense()

        return 1 - sims

    def offset(self, positive=[], negative=[], metric='cosine',
               n=10, filter_used=True):
        """
        Calculate a vector and return its neighbours.

        Implements the vector offset method as described in

        Tomas Mikolov, Scott Wen-tau Yih, and Geoffrey Zweig. 2013. Linguistic
        regularities in continuous space word representations. In NAACL HLT.
        """
        if not len(positive + negative) > 0:
            return None

        for w in positive + negative:
            if not self.defined_at(w):
                return None

        positive_vec = self.word_vectors_matrix(positive)
        negative_vec = self.word_vectors_matrix(negative)

        vectors = np.vstack([
            positive_vec,
            negative_vec * -1.0])
        norm_vectors = np.linalg.norm(vectors, axis=1)

        vectors = np.dot(vectors.T, np.diag(1./norm_vectors)).T

        rel = vectors.sum(axis=0)

        sims = self.pairwise_distances(rel, self.vectors, metric=metric)

        all_indexes = sims.argsort()[0].tolist()

        counter = 0
        neighs_indexes = []
        neighs = []

        for i in all_indexes:
            word = self.words[i]
            if filter_used and ([word] in positive or [word] in negative):
                continue

            neighs_indexes.append(i)
            neighs.append(word)

            counter += 1

            if counter == n:
                break

        neighs_dist = [float(v) for v in sims[0, neighs_indexes]]
        return list(zip(neighs, neighs_dist))

    # Vector selection methods

    def word_vectors_matrix(self, elements):
        """
        Return a matrix containing vectors of the elements.

        Takes a collection of string/lists as an argument.
        Returns a matrix with vectors computed as simple word vectors if sting
        or a sum of vectors if a list.
        """
        if not elements:
            return self.vectors[[], :]

        if all((isinstance(e, str) for e in elements)):
            elem_rows = [self.word2id[e] for e in elements]
            result = self.vectors[elem_rows, :]
        else:
            vectors = []
            for elem in elements:
                vector = self.get_vector(elem)
                vectors.append(vector)

            if scipy.sparse.issparse(vectors[0]):
                result = scipy.sparse.vstack(vectors)
            else:
                result = np.vstack(vectors)

        return result

    def combined_vector(self, words):
        """Return the sum of all the vectors for a set of words."""

        rows = self._row_nums(words)

        if not scipy.sparse.issparse(self.vectors):
            vectors = self.vectors[rows, :]
            vector = np.sum(vectors, 0)
        else:
            vector = np.array(self.vectors[rows, :].sum(0))[0]

        if self.prenorm:
            return vector/np.linalg.norm(vector)
        else:
            return vector

    def word_vector(self, word, dense=False):
        """
        Return a word vector.

        If dense is True, will transform the vector to dense format if
        necessary.
        """
        word_row = self.words.index(word)
        vector = self.vectors[word_row, :]
        if dense and scipy.sparse.issparse(vector):
            return np.array(vector.todense())[0]
        return vector

    def get_vector(self, element, dense=False):
        """
        Get either a vector corresponding either to a word (if element
        is a string) or a sum of vectors (if element is a list).
        """
        if isinstance(element, str):
            vector = self.word_vector(element, dense=dense)
        elif isinstance(element, list):
            vector = self.combined_vector(element)
        else:
            raise Exception("An element must be a string or a list")
        return vector.reshape(1, -1)

    def vector_entropy(self, word):
        vector = self.word_vector(word, dense=True)
        return scipy.stats.entropy(vector, base=2)

    def label(self, e):
        """Create a label for an element."""
        if isinstance(e, str):
            return e
        elif isinstance(e, list):
            return ' '.join(e)
        else:
            raise Exception("Illegal element!")

    # Helper methods

    def _row_nums(self, words):
        """Return number of rows for words."""
        return [self.word2id[w] for w in words]

    def _words(self, indexes):
        """Return number of rows for words."""
        return self._all_elements(self.words, indexes)

    @staticmethod
    def _all_indexes(l, elems):
        """Return indexes of elements in a list."""
        return [l.index(e) for e in elems]

    @staticmethod
    def _all_elements(l, elems):
        """Return elements with indexes in a list."""
        return [l[i] for i in elems]

    def __repr__(self):
        nrows, ncols = self.shape
        return '<Semantic space (%s): %s words, %s dimensions>' % (
            self.title,
            nrows, ncols)
