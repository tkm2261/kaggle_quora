from nltk import word_tokenize
from nltk.stem import WordNetLemmatizer
from gensim.models.doc2vec import LabeledSentence
from gensim import models
import pandas
import numpy
import pickle
from multiprocessing import Pool

from spacy.en import English
parser = English()
wnl = WordNetLemmatizer()

from logging import getLogger
logger = getLogger(__name__)
from collections import Counter

import scipy.spatial.distance as di
# If a word appears only once, we ignore it completely (likely a typo)
# Epsilon defines a smoothing constant, which makes the effect of extremely rare words smaller

from nltk.corpus import stopwords
stops = set(stopwords.words("english"))

import aspell
asp = aspell.Speller('lang', 'en')

import sense2vec
model = sense2vec.load()
SIZE = 128


def _calc(row):
    vec = []
    for i, word in enumerate(row):
        #word = word.lower()
        try:
            _, query_vector = model[word]
            vec.append(query_vector)
        except KeyError:
            continue

    if len(vec) > 0:
        vec = numpy.array(vec)
        a = numpy.min(vec, axis=0)
        b = numpy.max(vec, axis=0)
        vec = numpy.where(numpy.fabs(b) > numpy.fabs(a), b, a)
        return vec  #
        # return numpy.mean(vec, axis=0)

    else:
        return numpy.zeros(SIZE)

from numba import jit


@jit
def dist_cst(v1, v2, func):
    n = v1.shape[0]
    ret = numpy.zeros(n)
    for i in range(n):
        ret[i] = func(v1[i], v2[i])
    return ret


def load_data(row):

    q1 = parser(str(row[0]))
    q2 = parser(str(row[1]))

    list_word1 = ["{}|{}".format(w.lemma_, w.pos_) for w in q1]
    list_word2 = ["{}|{}".format(w.lemma_, w.pos_) for w in q2]

    vec1 = _calc(list_word1)
    vec2 = _calc(list_word2)

    list_dist = [di.euclidean, di.cityblock, di.cosine, di.correlation, di.chebyshev, di.canberra, di.braycurtis]
    list_ret = []
    for func in list_dist:
        list_ret.append(func(vec1, vec2))
    ret = list_ret
    return numpy.r_[vec1, vec2, ret]


if __name__ == '__main__':
    from logging import StreamHandler, DEBUG, Formatter, FileHandler

    log_fmt = Formatter('%(asctime)s %(name)s %(lineno)d [%(levelname)s][%(funcName)s] %(message)s ')
    handler = FileHandler('svo.py.log', 'w')
    handler.setLevel(DEBUG)
    handler.setFormatter(log_fmt)
    logger.setLevel(DEBUG)
    logger.addHandler(handler)

    handler = StreamHandler()
    handler.setLevel('INFO')
    handler.setFormatter(log_fmt)
    logger.setLevel('INFO')
    logger.addHandler(handler)
    p = Pool()
    """
    df = pandas.read_csv('../data/train_clean2.csv',
                         usecols=['question1', 'question2']).values

    ret = numpy.array(list(p.map(load_data, df)))
    print(ret[:100])
    with open('train_sense.pkl', 'wb') as f:
        pickle.dump(ret, f, -1)
    logger.info('tran end')
    """
    df = pandas.read_csv('../data/test_clean2.csv',
                         usecols=['question1', 'question2']).values

    ret = numpy.array(list(p.map(load_data, df)))
    with open('test_sense.pkl', 'wb') as f:
        pickle.dump(ret, f, -1)

    p.close()
    p.join()
