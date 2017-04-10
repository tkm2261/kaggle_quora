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
    logger.info('6')
    logger.info('2')
    with open('fast_train_clean2_inc.pkl', 'rb') as f:
        x = pickle.load(f).astype(np.float32)
    x = calc(x)
    with open('fast_train_clean2_inc2.pkl', 'wb') as f:
        pickle.dump(x, f, -1)

    with open('fast_train_clean2_low_max.pkl', 'rb') as f:
        x = pickle.load(f).astype(np.float32)
    x = calc(x)
    with open('fast_train_clean2_low_max2.pkl', 'wb') as f:
        pickle.dump(x, f, -1)

    with open('glove_train_clean2.pkl', 'rb') as f:
        x = pickle.load(f).astype(np.float32)
    x = calc(x)
    with open('glove_train_clean2_2.pkl', 'wb') as f:
        pickle.dump(x, f, -1)

    with open('glove_train_clean2_max.pkl', 'rb') as f:
        x = pickle.load(f).astype(np.float32)
    x = calc(x)
    with open('glove_train_clean2_max2.pkl', 'wb') as f:
        pickle.dump(x, f, -1)


def test_data():
    logger.info('start')
    with open('fast_test_clean2_low.pkl', 'rb') as f:
        x = pickle.load(f).astype(np.float32)
    x = calc(x)
    with open('fast_test_clean2_low2.pkl', 'wb') as f:
        pickle.dump(x, f, -1)

    with open('fast_test_clean2_low_max.pkl', 'rb') as f:
        x = pickle.load(f).astype(np.float32)
    x = calc(x)
    with open('fast_test_clean2_low_max2.pkl', 'wb') as f:
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
    #x_test = test_data()
