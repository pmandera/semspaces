"""
Save and load semantic spaces.
"""
import csv
import os
import codecs
import re
import gzip

from contextlib import closing

import tempfile
import zipfile
import shutil
import fs.zipfs
import numpy as np
import scipy.io
import scipy.sparse

try:
    import pandas as pd
except ImportError:
    pd = None
    print('Warning: pandas not available. Importing to pandas will not work.')


class AbstractSemanticSpaceMarket(object):
    """Read and write from/to semantic space format."""
    def __init__(self, fname, mode='r'):
        self.fname = fname
        self.mode = mode

    def get_file(self, fname, mode):
        """
        Get a file.

        Should be implemented depending on used filesystem and return
        a file object if the requested file exists. None otherwise.
        """
        raise NotImplementedError(
            "rows_file(...) must be overriden by a subclass.")

    def rows_file(self, mode):
        """Return rows file."""
        return self.get_file('row-labels', mode)

    def cols_file(self, mode):
        """Return columns file."""
        return self.get_file('col-labels', mode)

    def readme_file(self, mode):
        return self.get_file('README.md', mode)

    def data_file(self, mode):
        return self.get_file('data.mtx', mode)

    # Reading methods

    def read_rows(self):
        """Read rows."""
        rows_file = self.rows_file('r')
        if rows_file:
            rows = [r.strip('\n') for r in rows_file.readlines()]
            rows_file.close()
            return rows
        else:
            return None

    def read_cols(self):
        """Read columns."""
        cols_file = self.cols_file('r')
        if cols_file:
            cols = [c.strip('\n') for c in cols_file.readlines()]
            cols_file.close()
            return cols
        else:
            return None

    def read_readme(self):
        """Read columns."""
        readme_file = self.readme_file('r')
        if readme_file:
            readme = readme_file.readlines()
            readme_file.close()
            if len(readme) >= 3:
                title = readme[0].strip()
                description = ''.join(readme[2:])
            elif len(readme) >= 1:
                title = readme[0].strip()
                description = ''
            else:
                print('Warning: README.md exists but seems to be malformed.')
                return None
            return (title, description)
        else:
            return None

    def read_data(self):
        """Read matrix."""
        matrix_file = self.data_file('r')
        if matrix_file:
            matrix = scipy.io.mmread(matrix_file)
            matrix_file.close()
            if scipy.sparse.issparse(matrix):
                matrix = matrix.tocsr()
            return matrix
        else:
            return None

    def read_all(self):
        """Convenience function for reading all data."""
        rows = self.read_rows()
        cols = self.read_cols()
        readme = self.read_readme()
        data = self.read_data()
        if readme is not None:
            return (data, rows, cols, readme[0], readme[1])
        else:
            return (data, rows, cols, None, None)

    def read_to_pandas(self):
        """Read to pandas dataframe"""
        data, rows, cols, r1, r2 = self.read_all()
        df = pd.DataFrame(data, index=rows)
        if cols is not None:
            df.columns = cols
        return df

    # Writing methods

    def write_rows(self, rows):
        """Write rows."""
        rows_file = self.rows_file('wb')
        for row in rows:
            rows_file.write(row + '\n')
        rows_file.close()

    def write_cols(self, cols):
        """Write columns."""
        cols_file = self.cols_file('wb')
        for col in cols:
            cols_file.write(col + '\n')
        cols_file.close()

    def write_readme(self, title, description=''):
        """Write README.md."""
        readme_file = self.readme_file('wb')
        readme_file.write('%s\n\n%s' % (title, description))
        readme_file.close()

    def write_data(self, matrix, comment='', precision=None):
        """Write matrix."""
        matrix_file = self.data_file('wb')
        scipy.io.mmwrite(matrix_file, matrix, comment=comment,
                         precision=precision)
        matrix_file.close()

    def write_all(self, matrix, rows, cols=None, readme_title='',
                  readme_desc=''):
        """Convenience function for writing all elements at once."""
        self.write_data(matrix)
        self.write_rows(rows)
        if cols is not None:
            self.write_cols(cols)
        self.write_readme(readme_title, readme_desc)

    def write_from_pandas(self, df, readme_title='', readme_desc=''):
        """Write a semantic space from pandas data frame"""
        rows = list(df.index)
        cols = [str(c) for c in list(df.columns)]
        self.write_all(df.as_matrix(), rows, cols, readme_title, readme_desc)

    def close(self):
        pass

    def __repr__(self):
        class_name = self.__class__.__name__
        return "%s('%s', '%s')" % (class_name, self.fname, self.mode)

    def __del__(self):
        self.close()


class ZipSemanticSpaceMarket(AbstractSemanticSpaceMarket):
    """Read and write from/to semantic space format Zip file."""
    def __init__(self, fname, mode='r'):
        super(ZipSemanticSpaceMarket, self).__init__(fname, mode)
        self.root_file = self.create_fs(fname, mode)

    @staticmethod
    def create_fs(uri, mode):
        return fs.zipfs.ZipFS(uri, mode, allow_zip_64=True)

    def get_file(self, fname, mode):
        if mode != 'r' or self.root_file.isfile(fname):
            f = self.root_file.open(fname, mode, encoding='utf-8')
            return f
        else:
            return None

    def close(self):
        self.root_file.close()


class DirSemanticSpaceMarket(AbstractSemanticSpaceMarket):
    """Read and write from/to semantic space format Zip file."""
    def get_file(self, fname, mode):
        path = os.path.join(self.fname, fname)
        if mode != 'r' or os.path.isfile(path):
            if fname[-4:] == '.mtx':
                # scipy.io.mmread works better with plain open
                f = open(path, mode)
            else:
                f = codecs.open(path, mode, encoding='utf-8')
            return f
        else:
            return None

class ExternalZipSemanticSpaceMarket(DirSemanticSpaceMarket):
    """
    Read and write from/to semantic space format Zip file by
    first decompressing the data to temporary folder.

    This implementation has less memory footprint than the
    ZipSemanticSpaceMarket.
    """
    def __init__(self, fname, mode='r'):
        self.archive_name = fname
        self.temp = tempfile.mkdtemp()
        self.saved = False

        if mode == 'r':
            with zipfile.ZipFile(fname, 'r', allowZip64=True) as z:
                z.extractall(self.temp)

        super(ExternalZipSemanticSpaceMarket, self).__init__(self.temp, mode)

    def close(self):
        if self.mode == 'w' and not self.saved:
            with closing(zipfile.ZipFile(self.archive_name, 'w',
                                 zipfile.ZIP_DEFLATED, allowZip64=True)) as z:
                for root, dirs, files in os.walk(self.temp):
                    for f in files:
                        z.write(os.path.join(root, f), f)
            self.saved = True

        if os.path.isdir(self.temp):
            shutil.rmtree(self.temp)


class SemanticSpaceMarket(ExternalZipSemanticSpaceMarket):
    """Default semantic space input output class"""


class CSVWriter(object):
    """Saves word vectors to a CSV format.

    :param fname: filename
    :param space: semantic space
    :param delim: delimiter (defaults to ' ' as in word2vec output format
    :param compress: whether to gzip the file
    :param shape: whether to write the shape of the space on the first line
    """
    @staticmethod
    def write(fname, space, delim=' ',
              compress=True, header=True,
              shape=False):

        if compress:
            fout = gzip.open(fname, 'wt')
        else:
            fout = open(fname, 'wt')

        if header is True:
            if space.title is not None:
                fout.write('# TITLE: %s\n' % space.title)
            if space.readme is not None:
                readme_lines = space.readme.split('\n')
                for line in readme_lines:
                    fout.write('# %s\n' % line.strip())
        if shape:
            nrows, ncols = space.vectors.shape
            fout.write('%s %s\n' % (nrows, ncols))

        space.to_pandas().to_csv(fout, encoding='utf-8',
                                 sep=' ', header=False)
        fout.close()


# Interoperability with  word2vec google tool

class CSVReader(object):
    """
    Reads word vectors from a file in a CSV format.

    Assumes the file to have no header and list words in the first column.

    Works also with files created by the Google word2vec tool, which
    indicate the number of rows and columns in the first line.

    It will also read additional information about the space from the comment
    lines (starting with '#') at the beginning of the file.

    The title is read from a line starting with:

        # TITLE: (.*)

    The remaining lines are read as readme.
    """
    @staticmethod
    def read_header(fin):
        """Return the title and comments lines, advance fin to the end of
        the header."""
        title = None
        readme = []

        title_regex = re.compile('# TITLE: (.*)')

        # remember position to be able to come back
        position = fin.tell()

        line = fin.readline()

        while line:
            if line[0] == '#':
                position = fin.tell()

                title_match = title_regex.match(line)
                if title_match:
                    title = title_match.group(1)
                else:
                    readme_line = re.sub('#[ ]?', '', line.strip())
                    readme.append(readme_line)
            else:
                fin.seek(position)
                break

            line = fin.readline()

        return (title, '\n'.join(readme))

    @staticmethod
    def read_ncol(fin, delim):
        """Determine number of columns"""
        position = fin.tell()
        row = fin.readline().split(delim)
        if len(row) == 2:
            ncol = int(row[1])
        else:
            ncol = len(row) - 1
            fin.seek(position)
        return ncol

    @staticmethod
    def read_vectors(fin, ncol, dtype='float64', delim=' '):
        """Return a list with tuples (word, word_vector)."""
        reader = csv.reader(fin, delimiter=delim, quoting=csv.QUOTE_NONE)
        word_vectors = [
            (
                row[0],
                np.array(row[1: ncol + 1], dtype=dtype)
            )
            for row in reader
        ]
        return word_vectors

    @staticmethod
    def read_vectors_pd(fin, ncol, delim=' '):
        """Return Pandas DF with words as index"""
        df = pd.read_csv(fin, sep=delim, header=None, index_col=0,
                         keep_default_na=False, quoting=csv.QUOTE_NONE,
                         usecols=range(ncol + 1), encoding='utf-8')
        return df

    @staticmethod
    def read_file(fname, dtype='float64', delim=' '):
        """Return a tuple with (words, [list of vector values])."""
        if fname.endswith('.gz'):
            fin = gzip.open(fname, 'rt')
        else:
            fin = open(fname, 'rt')

        title, readme = CSVReader.read_header(fin)
        ncol = CSVReader.read_ncol(fin, delim)

        if pd is not None:
            df = CSVReader.read_vectors_pd(fin, ncol, delim)
            vectors = np.array(df.astype(dtype))
            words = list(df.index)
        else:
            word_vectors = CSVReader.read_vectors(fin, ncol, dtype, delim)
            words, vectors = list(zip(*word_vectors))
            vectors = np.array(vectors)

        return (words, vectors, title, readme)

    @classmethod
    def read(cls, fname, dtype='float64', delim=' '):
        """
        Read semantic space data from a CSV file.

        :param fname: filename
        :param dtype: numpy data type of a resulting matrix
        :param delim: delimiter
        :returns: tupel (words, np.array vector matrix, title, readme)
        """
        return cls.read_file(fname, dtype=dtype, delim=' ')
