import functools
import json
import multiprocessing as mp
import os
import pickle
import queue
import re
import shutil
import time
from itertools import islice

import numpy as np
import pandas as pd
from my_decorators import profile_funcs, profile_lines
from nltk import word_tokenize, pos_tag
from nltk.stem import WordNetLemmatizer
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import CountVectorizer


class LemmaTokenizer(object):

    def __init__(self):
        self.lemmatizer = WordNetLemmatizer()

    def __call__(self, doc):
        res = []
        for token, pos in pos_tag(word_tokenize(doc)):
            pos = pos.lower()
            # if token in ['#', '%', '&', "''", "'ll", "'re", ""]:
            #     print(token, pos)
            # if pos in ['pos', 'md']:
            #     print(token, pos)
            if pos == 'pos':
                continue

            if pos[0] in ['a', 'n', 'v']:
                res.append(self.lemmatizer.lemmatize(token, pos[0]))
            else:
                res.append(self.lemmatizer.lemmatize(token))

        return res


def preprocess(s):

    s = s.lower()
    # s = re.sub('[ ,!.?\[\]\-_|*&#%^/:;]+', ' ', s)
    s = re.sub('[ *^]+', ' ', s)

    return s


def load_save(func=None, fpath=None):

    assert fpath is not None

    if func is None:
        return functools.partial(load_save, fpath=fpath)

    @functools.wraps(func)
    def inner(*args, **kwargs):

        if os.path.isfile(fpath):
            with open(fpath, 'rb') as f:
                res = pickle.load(f)
        else:
            res = func(*args, **kwargs)
            with open(fpath, 'wb') as f:
                pickle.dump(res, f)

        return res

    return inner


def worker(fn, jobs_q, results_q):

    for args in iter(jobs_q.get, None):

        fn(*args)
        results_q.put(None)


def apply_async_ll(fn, iterable, nprocs=mp.cpu_count()):

    jobs_q = mp.Queue(maxsize=nprocs)
    results_q = mp.Queue()

    procs = [mp.Process(target=worker, args=(fn, jobs_q, results_q)) for _ in range(nprocs)]
    for proc in procs:
        proc.start()

    it = iter(iterable)

    consumed = True
    while True:

        try:
            if consumed:
                x = next(it)
                consumed = False
            jobs_q.put(x, block=False)
        except queue.Full:
            results_q.get(block=True)
        except StopIteration:
            break
        else:
            consumed = True

    for _ in range(nprocs):
        jobs_q.put(None, block=True)

    for proc in procs:
        proc.join()


def get_batches(batch_size, fpath='data/RC_2015-01', start_batch=0):

    with open(fpath) as f:
        i = 0
        while True:
            lines = list(islice(f, batch_size))
            if not lines:
                return
            if i < start_batch:
                i += 1
                continue
            data = [json.loads(l) for l in lines]
            yield i, data
            i += 1


class PartialResult(object):

    def __init__(self, tokens, vocab, subreddits, ids):

        self.tokens = tokens
        self.vocab = vocab
        self.subreddits = subreddits
        self.ids = ids

    @staticmethod
    def combine(partial_results):

        vocab = sorted(set(x for pr in partial_results for x in pr.vocab))

        for pr in partial_results:
            pr.reindex_vocab(vocab)

        tokens = partial_results[0].tokens.copy()
        subreddits = list(partial_results[0].subreddits)
        ids = list(partial_results[0].ids)

        for pr in partial_results[1:]:
            tokens.data = np.concatenate((tokens.data, pr.tokens.data))
            tokens.indices = np.concatenate((tokens.indices, pr.tokens.indices))
            tokens.indptr = np.concatenate((tokens.indptr, pr.tokens.indptr[1:] + tokens.nnz))
            tokens._shape = (tokens.shape[0] + pr.tokens.shape[0], tokens.shape[1])
            subreddits += pr.subreddits
            ids += pr.ids

        res = PartialResult(tokens, vocab, subreddits, ids)

        return res

    def reindex_vocab(self, vocab_all):

        assert(sorted(self.vocab) == self.vocab)

        tokens = self.tokens.tocsc()

        indptr = []

        j = 0
        for token in vocab_all:
            indptr.append(tokens.indptr[j])

            if j >= len(self.vocab):
                continue

            if token == self.vocab[j]:
                j += 1

        indptr.append(tokens.indptr[j])

        tokens._shape = (tokens.shape[0], len(vocab_all))

        tokens.indptr = np.array(indptr, dtype=np.int32)

        self.vocab = vocab_all
        self.tokens = tokens.tocsr()


def proc_batch(i, batch, dpath_out):

    t0 = time.time()

    tokenizer = LemmaTokenizer()

    vect = CountVectorizer(tokenizer=tokenizer, preprocessor=preprocess, stop_words='english')

    subreddits, comments, ids = zip(*map(lambda d: (d['subreddit'], d['body'], d['id']), batch))
    res = vect.fit_transform(comments)
    vocab = vect.get_feature_names()

    fpath = os.path.join(dpath_out, '%i.pkl' % i)
    with open(fpath, 'wb') as f:
        pickle.dump(PartialResult(res, vocab, subreddits, ids), f)

    print(i, time.time() - t0)

    return res


def main(dpath_out='data/vectorized_data', resume=True, batch_size=100000, multiproc=True):

    if not os.path.isdir(dpath_out):
        os.mkdir(dpath_out)

    if resume:
        try:
            start_batch_index = max(int(os.path.splitext(fname)[0]) for fname in os.listdir(dpath_out)) + 1
        except ValueError:
            start_batch_index = 0
    else:
        start_batch_index = 0
        shutil.rmtree(dpath_out)
        os.mkdir(dpath_out)

    it = get_batches(start_batch=start_batch_index, batch_size=batch_size)

    t_start = time.time()

    if multiproc:
        fn = functools.partial(proc_batch, dpath_out=dpath_out)
        apply_async_ll(fn, it)
    else:
        for i, data in it:
            proc_batch(i, data, dpath_out)

    print(time.time() - t_start)


def get_all_results(dpath='data/vectorized_data/'):

    l = []

    for i, fname in enumerate(os.listdir(dpath)):
        if i > 5:
            break
        fpath = os.path.join(dpath, fname)
        with open(fpath, 'rb') as f:
            l.append(pickle.load(f))

    res = PartialResult.combine(l)

    return res


# @profile_funcs
# @profile_lines
def get_subreddit_data(subreddits, tokens, vocab, echo_freq=1000):

    subreddits = np.array(subreddits)

    data = []

    indices_sorted = np.argsort(subreddits)
    subreddits_sorted = np.sort(subreddits)
    indices_sorted_change = np.where(subreddits_sorted[:-1] != subreddits_sorted[1:])[0] + 1
    indices_sorted_change = np.insert(indices_sorted_change, [0, -1], [0, len(subreddits)])

    index = np.unique(subreddits_sorted)
    # index = index[:100]

    for i, (j, k) in enumerate(zip(indices_sorted_change, indices_sorted_change[1:])):

        if i % echo_freq == 0:
            print(i, index[i])

        sr_indices = indices_sorted[j: k]
        a = tokens[sr_indices]
        b = a.sum(axis=0)
        c = np.asarray(b)
        sr_tokens = np.squeeze(c)
        data.append(sr_tokens)

    data = np.stack(data)

    data = csr_matrix(data)

    # df = pd.DataFrame(data, index=index, columns=vocab)

    return data, index


if __name__ == '__main__':

    main(resume=True)
