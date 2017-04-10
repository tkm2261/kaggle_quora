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
from subject_object_extraction import findSVOs
from logging import getLogger
logger = getLogger(__name__)
from collections import Counter

# If a word appears only once, we ignore it completely (likely a typo)
# Epsilon defines a smoothing constant, which makes the effect of extremely rare words smaller


def get_weight(count, D, eps=10, min_count=2):
    if count < min_count:
        return 0
    else:
        return numpy.log(D / (count + eps))


def aaa(row):
    return [word.lower_ for word in parser(str(row).lower())]


def make_idf():
    print('enter')

    df = pandas.read_csv('../data/train_increase.csv',
                         usecols=['question1', 'question2']).values

    df2 = pandas.read_csv('../data/test_clean2.csv',
                          usecols=['question1', 'question2']).values
    """
    p = Pool()
    _ret = p.map(aaa, df[:, 0])
    _ret = p.map(aaa, df[:, 1])
    _ret = p.map(aaa, df2[:, 0])
    _ret = p.map(aaa, df2[:, 1])
    p.close()
    p.join()
    ret = []
    for r in _ret:
        ret += r

    counts = Counter(ret)
    with open('svo_counter.pkl', 'wb') as f:
        pickle.dump(counts, f, -1)
    """
    with open('svo_counter.pkl', 'rb') as f:
        counts = pickle.load(f)

    weights = {word: get_weight(count, (df.shape[0] + df2.shape[0]) * 2) for word, count in counts.items()}
    print(len(weights))
    print('exit')
    return weights

weights = make_idf()


def feat(set1, set2):
    num_ent = 0
    val_ent = 0.
    sets = set1 & set2
    aaa = len(set1 | set2)
    if aaa > 0:
        rate_ent = len(sets) / aaa
    else:
        rate_ent = 0

    for word in sets:
        num_ent += 1  #
        val_ent += weights.get(word, 0)
    return num_ent, val_ent, rate_ent

import Levenshtein

LIST_ADV = ['what', 'why', 'which', 'who', 'when', 'how']


def load_data(row):
    q1 = str(row[0]).lower()
    q2 = str(row[1]).lower()
    ret1 = [0 for _ in range(len(LIST_ADV))]
    ret2 = [0 for _ in range(len(LIST_ADV))]
    ret3 = [0 for _ in range(len(LIST_ADV))]
    for i, adv in enumerate(LIST_ADV):
        if adv in q1:
            ret1[i] += 1
        if adv in q2:
            ret2[i] += 1
        if adv in q1 and adv in q2:
            ret3[i] += 1
    list_ret = ret1 + ret2 + ret3
    return list_ret


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
    df = pandas.read_csv('../data/train_increase.csv',
                         usecols=['question1', 'question2']).values

    ret = numpy.array(list(p.map(load_data, df)))
    print(ret[:100])
    with open('train_5w1h.pkl', 'wb') as f:
        pickle.dump(ret, f, -1)
    logger.info('tran end')
    """
    df = pandas.read_csv('../data/test_clean2.csv',
                         usecols=['question1', 'question2']).values

    ret = numpy.array(list(p.map(load_data, df)))
    with open('test_5w1h.pkl', 'wb') as f:
        pickle.dump(ret, f, -1)

    p.close()
    p.join()
