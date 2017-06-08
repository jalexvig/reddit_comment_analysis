import pickle
import numpy as np

import pandas as pd
from my_utils import pickle_dump, pickle_load
from sklearn.feature_extraction.text import TfidfTransformer

from parse_data import PartialResult, get_subreddit_data, get_all_results


def get_tfidf(df):

    transformer = TfidfTransformer()

    res = transformer.fit_transform(df)

    return res


def main():

    # res = get_all_results()
    # pickle_dump(res, 'result_tmp.pkl')
    #
    # res = pickle_load('result_tmp.pkl')
    # data, index = get_subreddit_data(res.subreddits, res.tokens, res.vocab)
    # vocab = res.vocab
    # pickle_dump((data, index, vocab), 'data.pkl')

    data, index, vocab = pickle_load('data.pkl')

    excl = ['[ .!\,?:;\]\[\'"()&$|\-`%]+$', '[n]?\'([mts]|re|ve)?$']
    mask = ~pd.Series(vocab).str.match('|'.join(excl)).values

    data = data[:, mask]

    # TODO(jalex): Figure out why this takes a long time
    vocab = np.asarray(vocab)[mask]

    tfidf = get_tfidf(data)

    df = pd.DataFrame(tfidf.toarray(), index=index, columns=vocab)

    print(df.loc['nsfw'].nlargest(8))


if __name__ == '__main__':

    main()
