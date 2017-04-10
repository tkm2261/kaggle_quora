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
from scipy.spatial.distance import euclidean, cosine


def load_wordvec():

    f = open('../fasttext/model_clean2_l.vec', 'r')
    map_result = {}
    f.readline()
    for line in f:
        line = line.strip().split(' ')
        word = line[0]
        vec = numpy.array(list(map(float, line[1:])))
        map_result[word.lower()] = vec
    return map_result

map_wordvec = load_wordvec()
mean_vec = numpy.mean([val for val in map_wordvec.values() if val.shape[0] == 100], axis=0)
logger.info('size: {}'.format(mean_vec.shape))


def load_data(row):
    q1 = parser(str(row[0]))
    q2 = parser(str(row[1]))

    set_last1 = set([ele.lower_ for ele in q1])  # if ele.pos_ != 'PUNCT'])
    set_last2 = set([ele.lower_ for ele in q2])  # if ele.pos_ != 'PUNCT'])
    both1 = set_last1  # - (set_last1 & set_last2)
    both2 = set_last2  # - (set_last1 & set_last2)

    vec1 = None
    for word in both1:
        if vec1 is None:
            vec1 = map_wordvec.get(word, mean_vec).copy()  # parser.vocab[word].repvec
        else:
            vec1 += map_wordvec.get(word, mean_vec)  # parser.vocab[word].repvec
    vec2 = None
    for word in both2:
        if vec2 is None:
            vec2 = map_wordvec.get(word, mean_vec).copy()  # parser.vocab[word].repvec
        else:
            vec2 += map_wordvec.get(word, mean_vec)  # parser.vocab[word].repvec

    if vec1 is None or vec2 is None:
        score = 0
        score2 = 0
    else:
        score = cosine(vec1, vec2)  # numpy.sqrt(((vec1 - vec2) ** 2).sum())
        score2 = euclidean(vec1, vec2)
    return [len(both1), len(both2), score, score2]


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
    df = pandas.read_csv('../data/train_clean2_rev.csv',
                         usecols=['question1', 'question2']).values

    ret = numpy.array(list(p.map(load_data, df)))
    print(ret[:100])
    with open('train_rest_sim3.pkl', 'wb') as f:
        pickle.dump(ret, f, -1)
    logger.info('tran end')

    df = pandas.read_csv('../data/test_clean2_rev.csv',
                         usecols=['question1', 'question2']).values

    exit()
    """
    ret = numpy.array(list(p.map(load_data, df)))
    with open('test_rest_sim2.pkl', 'wb') as f:
        pickle.dump(ret, f, -1)

    p.close()
    p.join()
    """
