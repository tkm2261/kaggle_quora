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

import aspell
asp = aspell.Speller('lang', 'en')


def get(w):
    try:
        return asp.suggest(w)[0]
    except IndexError:
        return w


def get_weight(count, D, eps=1, min_count=2):
    if count < min_count:
        return 0
    else:
        return numpy.log(D / (count + eps))


def aaa(row):
    # return str(row).lower().split()
    aaa = [word.lower_ for word in parser(str(row).lower()) if word not in stops]
    # aaa = [word if word in asp else get(word) for word in aaa]
    return aaa


def make_idf():
    print('enter')

    df = pandas.read_csv('../data/train_clean2.csv',
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
        # word = str(word)
        # if word not in asp:
        #    word = get(word)

        num_ent += 1  #
        val_ent += weights.get(word, 10.)
    return num_ent, val_ent, rate_ent

import Levenshtein
from jellyfish._jellyfish import damerau_levenshtein_distance, jaro_distance, match_rating_comparison
import re
PAT = re.compile('[A-Z0-9]+')
#PAT2 = re.compile('([A-Z]) +([A-Z])')


def load_data(row):
    q1 = parser(str(row[0]))
    q2 = parser(str(row[1]))

    set_ent1 = set([ele.orth_ for ele in q1])
    set_ent2 = set([ele.orth_ for ele in q2])

    set_up1 = set([ele.lower() for ele in set_ent1 if PAT.match(ele) is not None])
    set_up2 = set([ele.lower() for ele in set_ent2 if PAT.match(ele) is not None])

    set_low1 = set([ele.lower_ for ele in q1])
    set_low2 = set([ele.lower_ for ele in q2])

    set_int = set_up1 & set_up2
    set_merge = set_up1 | set_up2
    if len(set_merge) > 0:
        rate0 = len(set_int) / len(set_merge)
    else:
        rate0 = 0
    num1 = 0
    val1 = 0
    for word in set_up1:
        if word in set_low2:
            num1 += 1
            val1 += weights.get(word, 10.)
    if len(set_merge) > 0:
        rate1 = num1 / len(set_merge)
    else:
        rate1 = 0
    num2 = 0
    val2 = 0
    for word in set_up2:
        if word in set_low1:
            num2 += 1
            val2 += weights.get(word, 10.)
    if len(set_merge) > 0:
        rate2 = num2 / len(set_merge)
    else:
        rate2 = 0
    list_ret = [rate0, num1, val1, rate1, num2, val2, rate2]
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
    with open('train_ent2.pkl', 'wb') as f:
        pickle.dump(ret, f, -1)
    logger.info('tran end')

    df = pandas.read_csv('../data/test_clean.csv',
                         usecols=['question1', 'question2']).values

    ret = numpy.array(list(p.map(load_data, df)))
    with open('test_ent2.pkl', 'wb') as f:
        pickle.dump(ret, f, -1)

    p.close()
    p.join()
