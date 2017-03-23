from nltk import word_tokenize
from nltk.stem import WordNetLemmatizer
from gensim.models.doc2vec import LabeledSentence
from gensim import models
from gensim import corpora
from gensim.matutils import corpus2csc
from collections import Counter
import pandas as pd
import numpy as np
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from multiprocessing import Pool
from sklearn.model_selection import GridSearchCV, ParameterGrid, StratifiedKFold, cross_val_predict
import xgboost as xgb
from lightgbm.sklearn import LGBMClassifier
from sklearn.metrics import log_loss, roc_auc_score
import gc
from logging import getLogger
logger = getLogger(__name__)

from features_tmp import FEATURE
from scipy.spatial import distance as di
CHUNK_SIZE = 100000


def dist(v1, v2):
    return np.sqrt(((v1 - v2) ** 2).sum(axis=1))


def dist_man(v1, v2):
    return np.fabs(v1 - v2).sum(axis=1)

from numba import jit


@jit
def dist_cst(v1, v2, func):
    n = v1.shape[0]
    ret = np.zeros(n)
    for i in range(n):
        ret[i] = func(v1[i], v2[i])
    return ret


def calc(x):
    num = (x.shape[1] - 1) / 2
    logger.info('%s' % num)
    q1 = x[:, :num]
    q2 = x[:, num:-1]
    cos = x[:, -1]
    d = dist(q1, q2)
    d2 = dist_man(q1, q2)

    list_dist = [di.correlation, di.chebyshev, di.canberra, di.braycurtis]
    list_ret = []
    for func in list_dist:
        logger.info('{}'.format(func))
        list_ret.append(dist_cst(q1, q2, func))
    ret = np.c_[list_ret].T
    return np.c_[q1, q2, cos, d, d2, ret]


def train_data():
    logger.info('start')

    logger.info('2')
    x = pd.read_csv('lda100/lda_train.csv').values.astype(np.float32)
    x = calc(x)
    with open('lda100/lda_train.pkl', 'wb') as f:
        pickle.dump(x, f, -1)

    logger.info('3')
    x = pd.read_csv('lsi50/lsi_train.csv').values.astype(np.float32)
    x = calc(x)
    with open('lsi50/lsi_train.pkl', 'wb') as f:
        pickle.dump(x, f, -1)

    x = pd.read_csv('w2v100/w2v_train.csv').values.astype(np.float32)
    x = calc(x)
    with open('w2v100/w2v_train.pkl', 'wb') as f:
        pickle.dump(x, f, -1)

    logger.info('6')
    with open('../fasttext/fast_train.pkl', 'rb') as f:
        x = pickle.load(f).astype(np.float32)
    x = calc(x)
    with open('../fasttext/fast_train2.pkl', 'wb') as f:
        pickle.dump(x, f, -1)

    with open('../fasttext/fast_train_decay.pkl', 'rb') as f:
        x = pickle.load(f).astype(np.float32)
    x = calc(x)
    with open('../fasttext/fast_train_decay2.pkl', 'wb') as f:
        pickle.dump(x, f, -1)

    logger.info('7')

    with open('../glove/glove_train.pkl', 'rb') as f:
        x = pickle.load(f).astype(np.float32)
    x = calc(x)
    with open('../glove/glove_train2.pkl', 'wb') as f:
        pickle.dump(x, f, -1)

    with open('../glove/glove_train_100_max.pkl', 'rb') as f:
        x = pickle.load(f).astype(np.float32)
    x = calc(x)
    with open('../glove/glove_train_100_max2.pkl', 'wb') as f:
        pickle.dump(x, f, -1)

    logger.info('8')

    with open('../lexvec/lexvec_train_100.pkl', 'rb') as f:
        x = pickle.load(f).astype(np.float32)
    x = calc(x)
    with open('../lexvec/lexvec_train_100_2.pkl', 'wb') as f:
        pickle.dump(x, f, -1)

    with open('../lexvec/lexvec_train_100_max.pkl', 'rb') as f:
        x = pickle.load(f).astype(np.float32)
    x = calc(x)
    with open('../lexvec/lexvec_train_100_max2.pkl', 'wb') as f:
        pickle.dump(x, f, -1)


def test_data():
    logger.info('start')

    logger.info('2')
    x = pd.read_csv('lda100/lda_test.csv').values.astype(np.float32)
    x = calc(x)
    with open('lda100/lda_test.pkl', 'wb') as f:
        pickle.dump(x, f, -1)

    logger.info('3')
    x = pd.read_csv('lsi50/lsi_test.csv').values.astype(np.float32)
    x = calc(x)
    with open('lsi50/lsi_test.pkl', 'wb') as f:
        pickle.dump(x, f, -1)

    x = pd.read_csv('w2v100/w2v_test.csv').values.astype(np.float32)
    x = calc(x)
    with open('w2v100/w2v_test.pkl', 'wb') as f:
        pickle.dump(x, f, -1)

    logger.info('6')
    with open('../fasttext/fast_test.pkl', 'rb') as f:
        x = pickle.load(f).astype(np.float32)
    x = calc(x)
    with open('../fasttext/fast_test2.pkl', 'wb') as f:
        pickle.dump(x, f, -1)

    with open('../fasttext/fast_test_decay.pkl', 'rb') as f:
        x = pickle.load(f).astype(np.float32)
    x = calc(x)
    with open('../fasttext/fast_test_decay2.pkl', 'wb') as f:
        pickle.dump(x, f, -1)

    logger.info('7')

    with open('../glove/glove_test.pkl', 'rb') as f:
        x = pickle.load(f).astype(np.float32)
    x = calc(x)
    with open('../glove/glove_test2.pkl', 'wb') as f:
        pickle.dump(x, f, -1)

    with open('../glove/glove_test_100_max.pkl', 'rb') as f:
        x = pickle.load(f).astype(np.float32)
    x = calc(x)
    with open('../glove/glove_test_100_max2.pkl', 'wb') as f:
        pickle.dump(x, f, -1)

    logger.info('8')

    with open('../lexvec/lexvec_test_100.pkl', 'rb') as f:
        x = pickle.load(f).astype(np.float32)
    x = calc(x)
    with open('../lexvec/lexvec_test_100_2.pkl', 'wb') as f:
        pickle.dump(x, f, -1)

    with open('../lexvec/lexvec_test_100_max.pkl', 'rb') as f:
        x = pickle.load(f).astype(np.float32)
    x = calc(x)
    with open('../lexvec/lexvec_test_100_max2.pkl', 'wb') as f:
        pickle.dump(x, f, -1)

if __name__ == '__main__':
    from logging import StreamHandler, DEBUG, Formatter, FileHandler

    log_fmt = Formatter('%(asctime)s %(name)s %(lineno)d [%(levelname)s][%(funcName)s] %(message)s ')

    handler = StreamHandler()
    handler.setLevel('INFO')
    handler.setFormatter(log_fmt)
    logger.setLevel('INFO')
    logger.addHandler(handler)

    logger.info('load start')
    ################
    x_train = train_data()
    x_test = test_data()
