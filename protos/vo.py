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
        val_ent += 1
    return num_ent, val_ent, rate_ent


def feat2(set1, set2):
    num_full_match = 0
    num_part_match = 0
    num_all = len(set1) + len(set2)
    if num_all == 0:
        return (0, 0, 0, 0)
    for w1 in set1:
        for w2 in set2:
            if w1 == w2:
                num_full_match += 1
            elif w1 in w2 or w2 in w1:
                num_part_match += 1
    rate_full_match = num_full_match * 2 / num_all
    rate_part_match = num_part_match * 2 / num_all

    return num_full_match, num_part_match, rate_full_match, rate_part_match


def load_data(row):

    q1 = parser(str(row[0]))
    q2 = parser(str(row[1]))

    set_ent1 = set([ele.lemma_ for ele in q1.ents])
    set_ent2 = set([ele.lemma_ for ele in q2.ents])
    num_full_match, num_part_match, rate_full_match, rate_part_match = feat2(set_ent1, set_ent2)

    set_svo1 = set([(ele[0].lower(), ele[1].lower(), ele[2].lower()) for ele in findSVOs(q1)])
    set_svo2 = set([(ele[0].lower(), ele[1].lower(), ele[2].lower()) for ele in findSVOs(q2)])

    set_svo1 = set([(wnl.lemmatize(ele[0]), wnl.lemmatize(ele[1]), wnl.lemmatize(ele[2]))
                    for ele in set_svo1])
    set_svo2 = set([(wnl.lemmatize(ele[0]), wnl.lemmatize(ele[1]), wnl.lemmatize(ele[2]))
                    for ele in set_svo2])

    set_vo1 = set((ele[1], ele[2]) for ele in set_svo1)
    set_vo2 = set((ele[1], ele[2]) for ele in set_svo2)
    num_vo, val_vo, rate_vo = feat(set_vo1, set_vo2)

    return [num_full_match, num_part_match, rate_full_match, rate_part_match, num_vo, val_vo, rate_vo]


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
    with open('train_vo.pkl', 'wb') as f:
        pickle.dump(ret, f, -1)
    logger.info('tran end')
    """
    df = pandas.read_csv('../data/test_clean2.csv',
                         usecols=['question1', 'question2']).values

    ret = numpy.array(list(p.map(load_data, df)))
    with open('test_vo.pkl', 'wb') as f:
        pickle.dump(ret, f, -1)

    p.close()
    p.join()
