from argparse import ArgumentParser
from functools import partial
import pickle as pickle

import numpy as np


def make_id2method(vocab):
    return dict((i, method) for  method , (i, _) in vocab.items())

def merge_main_context(W, merge_fun=lambda m, c: np.mean([m, c], axis=0),
                       normalize=True):
    """
    Merge the main-word and context-word vectors for a weight matrix
    using the provided merge function (which accepts a main-word and
    context-word vector and returns a merged version).

    By default, `merge_fun` returns the mean of the two vectors.
    """

    vocab_size = int(len(W) / 2)
    for i, row in enumerate(W[:vocab_size]):
        merged = merge_fun(row, W[i + vocab_size])
        if normalize:
            merged /= np.linalg.norm(merged)
        W[i, :] = merged

    return W[:vocab_size]


def most_similar(W, vocab, id2method, method, n=15):
    """
    Find the `n` words most similar to the given `word`. The provided
    `W` must have unit vector rows, and must have merged main- and
    context-word vectors (i.e., `len(W) == len(word2id)`).

    Returns a list of word strings.
    """

    assert len(W) == len(vocab)

    method_id = vocab[method][0]

    dists = np.dot(W, W[method_id])
    #print(W[method_id]),strat stop step
    top_ids = np.argsort(dists)[::-1][:n + 1]

    return [id2method[id] for id in top_ids if id != method_id][:n]


def findcommon(vocab,valvocab,):

    n=0
    methodid = []
    for k in valvocab.keys():
        for j in vocab.keys():
            if(k==j and (n<=300)):
                methodid.append(k)
                n+=1
                print(n)
    return methodid


def recompare(W, valW, vocab,valvocab, methodid,n=5,m=1):

    assert len(W) == len(vocab)
    num=0
    for k in range(len(methodid)):

        method_id = methodid[k]
        id = vocab[method_id][0]
        dists = np.dot(W, W[id])
        top_ids = np.argsort(dists)[::-1][:n + 1]
        id = valvocab[method_id][0]
        valdists = np.dot(valW, valW[id])
        valtop_ids = np.argsort(valdists)[::-1][:m + 1]

        for i in range(0,len(top_ids)):
            for j in range(0, len(valtop_ids)):
                if top_ids[i] == valtop_ids[j]:
                    if valtop_ids[j] != id:#id是数 method——id是文字
                        num+=1
    print(num/len(methodid))

    return num/len(methodid)


def MRRcompare(W, valW, vocab,valvocab, methodid, n=10,m=10):


    assert len(W) == len(vocab)
    sum=0
    for k in range(len(methodid)):

        method_id = methodid[k]
        id = vocab[method_id][0]
        dists = np.dot(W, W[id])
        top_ids = np.argsort(dists)[::-1][:n + 1]
        id = valvocab[method_id][0]
        valdists = np.dot(valW, valW[id])
        valtop_ids = np.argsort(valdists)[::-1][:m + 1]

        for i in range(0,len(top_ids)):
            for j in range(0, len(valtop_ids)):
                if top_ids[i] == valtop_ids[j]:
                    if valtop_ids[j] != id:
                        sum = sum+1/j
    print(sum/len(methodid))

    return sum/len(methodid)


def Successrate(W,vocab,corpus,tvocab,n=10):


    assert len(W) == len(vocab)
    sum=0

    for k in tvocab.keys():
        k = k +'\n'
        id = vocab[k][0]
        dists = np.dot(W, W[id])
        top_ids = np.argsort(dists)[::-1][:n + 1]

        methodlist = FindGround(corpus,k)
        methodlist = list(set(methodlist))
        for j in top_ids:
            for i in methodlist :
                method_id = vocab[i][0]
                if method_id == j :
                    sum+=1

    print((sum-len(tvocab))/len(tvocab))

    return (sum/len(tvocab)*100)

def FindGround(corpus,method):
    for k in corpus.keys() :
        for i in corpus[k]:
            if i == method :
                return corpus[k]
     #list




def parse_args():
    parser = ArgumentParser(
        description=('Evaluate a GloVe vector-space model on a word '
                     'analogy test set'))

    parser.add_argument('vectors_path', type=partial(open, mode='rb'),
                        help=('Path to serialized vectors file as '
                              'produced by this GloVe implementation'))

    parser.add_argument('analogies_paths', type=partial(open, mode='r'),
                        nargs='+',
                        help=('Paths to analogy text files, where each '
                              'line consists of four words separated by '
                              'spaces `a b c d`, expressing the analogy '
                              'a:b :: c:d'))

    return parser.parse_args()