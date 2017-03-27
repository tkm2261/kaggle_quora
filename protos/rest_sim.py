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


def load_data(row):
    q1 = parser(str(row[0]))
    q2 = parser(str(row[1]))

    set_last1 = set([wnl.lemmatize(ele.lower_) for ele in q1 if ele.pos_ != 'PUNCT'])
    set_last2 = set([wnl.lemmatize(ele.lower_) for ele in q2 if ele.pos_ != 'PUNCT'])
    both1 = set_last1 - (set_last1 & set_last2)
    both2 = set_last2 - (set_last1 & set_last2)

    vec1 = None
    for word in both1:
        if vec1 is None:
            vec1 = parser.vocab[word].repvec
        else:
            vec1 += parser.vocab[word].repvec
    vec2 = None
    for word in both2:
        if vec2 is None:
            vec2 = parser.vocab[word].repvec
        else:
            vec2 += parser.vocab[word].repvec

    if vec1 is None or vec2 is None:
        score = 0
    else:
        score = numpy.sqrt(((vec1 - vec2) ** 2).sum())
    return [len(both1), len(both2), score]


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
    df = pandas.read_csv('../data/train.csv',
                         usecols=['question1', 'question2']).values

    ret = numpy.array(list(p.map(load_data, df)))
    print(ret[:100])
    with open('train_rest_sim.pkl', 'wb') as f:
        pickle.dump(ret, f, -1)
    logger.info('tran end')
    df = pandas.read_csv('../data/test.csv',
                         usecols=['question1', 'question2']).values

    ret = numpy.array(list(p.map(load_data, df)))
    with open('test_rest_sim.pkl', 'wb') as f:
        pickle.dump(ret, f, -1)

    p.close()
    p.join()
