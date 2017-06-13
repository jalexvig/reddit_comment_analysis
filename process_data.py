import numpy as np

import pandas as pd
from scipy.sparse import csr_matrix, csc_matrix, coo_matrix
from sklearn.feature_extraction.text import TfidfTransformer

from parse_data import PartialResult, get_data


class SparseDF(object):

    def __init__(self, data, index, columns):

        self.index = np.asarray(index)
        self.columns = np.asarray(columns)

        if isinstance(data, csr_matrix):
            self._csr_data = data
        elif isinstance(data, csc_matrix):
            self._csc_data = data
        elif isinstance(data, coo_matrix):
            self._coo_data = data
        else:
            raise TypeError('Unrecognized sparse matrix data dtype %s' % type(data))

    @property
    def csr_data(self):

        self.set_data('csr')

        return self._csr_data

    @property
    def csc_data(self):

        self.set_data('csc')

        return self._csc_data

    def set_data(self, sparse_type):

        data_name = '_%s_data' % sparse_type

        if hasattr(self, data_name):
            return

        for existing_data_name in ['_%s_data' % t for t in ['coo', 'csr', 'csc']]:
            try:
                existing_data = getattr(self, existing_data_name)
            except AttributeError:
                continue
            else:
                val = getattr(existing_data, 'to%s' % sparse_type)()
                setattr(self, data_name, val)

    @property
    def coo_data(self):

        self.set_data('coo')

        return self._coo_data

    def get(self, selector, axis=0):

        if axis == 0:
            i = np.where(self.index == selector)[0]
            data = np.squeeze(self.csr_data[i].toarray())
            return pd.Series(data, index=self.columns)
        elif axis == 1:
            i = np.where(self.columns == selector)[0]
            data = np.squeeze(self.csc_data[:, i].toarray())
            return pd.Series(data, index=self.index)

    def get_tups_above_threshold(self, threshold):

        m = self.coo_data.data >= threshold

        rows = self.index[self.coo_data.row[m]]
        cols = self.columns[self.coo_data.col[m]]
        data = self.coo_data.data[m]

        return zip(rows, cols, data)


def get_tfidf(counts):

    res = TfidfTransformer().fit_transform(counts)

    return res


# @profile_lines
def cull_data(data, vocab, subreddits, regexs=None, min_num_vocab=0, subreddits_count=None, min_num_posts=0):

    if min_num_posts:
        srs = subreddits_count.index[subreddits_count >= min_num_posts]
        mask_subreddits = pd.Series(subreddits).isin(srs).values
        data = data[mask_subreddits]
        subreddits = subreddits[mask_subreddits]

    vocab = np.asarray(vocab)
    mask_vocab = np.ones(len(vocab), dtype=bool)

    if regexs:
        mask_vocab &= ~pd.Series(vocab).str.match('|'.join(regexs)).values

    if min_num_vocab:
        mask_vocab &= np.bincount(data.indices) >= min_num_vocab

    data = data[:, mask_vocab]
    vocab = vocab[mask_vocab]

    return data, subreddits, vocab


def main():

    res, sr_counts = get_data()

    excl = ['[ .!\,?:;\]\[\'"”“’()&$|\-`%]+$', '[n]?\'([mts]|re|ve)?$', 'delete$']
    counts, subreddits, vocab = cull_data(res.tokens, res.vocab, res.subreddits, excl, min_num_vocab=10,
                                          subreddits_count=sr_counts, min_num_posts=10)

    tfidf = get_tfidf(counts)

    sdf = SparseDF(tfidf, subreddits, vocab)

    return sdf


if __name__ == '__main__':

    main()
