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

from nltk.corpus import stopwords
stops = set(stopwords.words("english"))


def get_weight(count, D, eps=1, min_count=2):
    if count < min_count:
        return 0
    else:
        return numpy.log(D / (count + eps))

import aspell
asp = aspell.Speller('lang', 'en')


def get(w):
    try:
        return asp.suggest(w)[0]
    except IndexError:
        return w


def aaa(row):
    # return str(row).lower().split()
    aaa = [word.lower_ for word in parser(str(row).lower()) if word not in stops]
    #aaa = [word if word in asp else get(word) for word in aaa]
    return aaa


def make_idf():
    print('enter')

    df = pandas.read_csv('../data/train_clean.csv',
                         usecols=['question1', 'question2']).values

    df2 = pandas.read_csv('../data/test_clean.csv',
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
        val_ent += weights.get(word, 10.)
    return num_ent, val_ent, rate_ent


LIST_TAGS = ['-LRB-', '-PRB-', ',', ':', '.', "''", '""', '#', '``', '$', 'ADD', 'AFX', 'BES', 'CC', 'CD', 'DT', 'EX', 'FW', 'GW', 'HVS', 'HYPH', 'IN', 'JJ', 'JJR', 'JJS', 'LS', 'MD', 'NFP',
             'NIL', 'NN', 'NNP', 'NNPS', 'NNS', 'PDT', 'POS', 'PRP', 'PRP$', 'RB', 'RBR', 'RBS', 'RP', 'SP', 'SYM', 'TO', 'UH', 'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ', 'WDT', 'WP', 'WP$', 'WRB', 'XX']


def load_data(row):

    q1 = parser(str(row[0]))
    q2 = parser(str(row[1]))

    list_ret = []
    for tag in LIST_TAGS:
        set_sub1 = set([ele.lower_ for ele in q1 if ele.tag_ == tag])
        set_sub2 = set([ele.lower_ for ele in q2 if ele.tag_ == tag])
        num_sub, val_sub, rate_sub = feat(set_sub1, set_sub2)
        list_ret += [num_sub, val_sub, rate_sub]

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
    df = pandas.read_csv('../data/train_clean.csv',
                         usecols=['question1', 'question2']).values

    ret = numpy.array(list(p.map(load_data, df)))
    print(ret[:100])

    with open('train_tag.pkl', 'wb') as f:
        pickle.dump(ret, f, -1)
    logger.info('tran end')

    """
    df = pandas.read_csv('../data/test_clean.csv',
                         usecols=['question1', 'question2']).values

    ret = numpy.array(list(p.map(load_data, df)))
    with open('test_tag.pkl', 'wb') as f:
        pickle.dump(ret, f, -1)

    p.close()
    p.join()
    """
