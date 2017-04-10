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

CHUNK_SIZE = 100000


def train_data():
    logger.info('start')
    x = pd.read_csv('kernel_train3.csv').values.astype(np.float32)
    x_train = pd.read_csv('count_tfidf_train.csv').values.astype(np.float32)
    x_train = np.c_[x_train, x]

    x_train[np.isnan(x_train)] = -100
    x_train[np.isinf(x_train)] = -100
    return x_train  # [:, FEATURE + [-1]]


def calc_weight(y_train):
    pos_rate = 0.165
    neg_rate = 1 - pos_rate
    pos_num = y_train.sum()
    neg_num = y_train.shape[0] - y_train.sum()

    logger.info('pos_rate: %s, target pos_rate: %s, pos_num: %s' % (pos_num / y_train.shape[0], pos_rate, pos_num))

    w = (neg_num * pos_rate) / (pos_num * (1 - pos_rate))
    sample_weight = np.where(y_train == 1, w, 1)
    calc_pos_rate = (w * pos_num) / (w * pos_num + neg_num)
    logger.info('calc pos_rate: %s' % calc_pos_rate)
    return sample_weight

if __name__ == '__main__':
    from logging import StreamHandler, DEBUG, Formatter, FileHandler

    log_fmt = Formatter('%(asctime)s %(name)s %(lineno)d [%(levelname)s][%(funcName)s] %(message)s ')
    handler = FileHandler('tfidf_k.py.log', 'w')
    handler.setLevel(DEBUG)
    handler.setFormatter(log_fmt)
    logger.setLevel(DEBUG)
    logger.addHandler(handler)

    handler = StreamHandler()
    handler.setLevel('INFO')
    handler.setFormatter(log_fmt)
    logger.setLevel('INFO')
    logger.addHandler(handler)

    logger.info('load start')
    df_train = pd.read_csv('../data/train.csv', usecols=['is_duplicate'])
    df_test = pd.read_csv('../data/test.csv', usecols=['test_id'])

    ################
    #x_train_rev = train_data_rev()
    x_train = train_data()

    #####################

    y_train = df_train['is_duplicate'].values
    with open('tfidf_all_pred2.pkl', 'rb') as f:
        cross_pred = pickle.load(f).astype(np.float32)
    y_train_mod = y_train.copy()
    #y_train_mod[cross_pred > 0.99] = 1

    sample_weight = calc_weight(y_train)
    sample_weight_mod = calc_weight(y_train_mod)

    del df_train
    gc.collect()

    logger.info('x_shape: {}'.format(x_train.shape))

    # w = (pos_num * neg_rate) / (neg_num * (1 - neg_rate))
    # sample_weight = np.where(y_train == 0, w, 1)
    #
    # calc_pos_rate = (pos_num) / (pos_num + w * neg_num)
    # logger.info('calc pos_rate: %s' % calc_pos_rate)

    logger.info('sampling start')

    from sklearn.cross_validation import train_test_split
    #{'min_child_weight': 5, 'subsample': 0.9, 'seed': 2261, 'reg_lambda': 1, 'num_leaves': 1000, 'min_child_samples': 100, 'max_depth': 10, 'boosting_type': 'gbdt', 'reg_alpha': 1, 'n_estimators': 619, 'min_split_gain': 0, 'colsample_bytree': 0.7, 'learning_rate': 0.1, 'max_bin': 500}

    # x_train, x_valid, y_train, y_valid = train_test_split(x_train, y_train, test_size=0.2, random_state=4242)
    all_params = {'max_depth': [10],
                  'learning_rate': [0.1],  # [0.06, 0.1, 0.2],
                  'n_estimators': [10000],
                  'min_child_weight': [5],
                  'colsample_bytree': [0.7],
                  'boosting_type': ['gbdt'],
                  'num_leaves': [1000],
                  'subsample': [0.9],
                  'min_child_samples': [100],
                  'reg_alpha': [1],
                  'reg_lambda': [1],
                  'max_bin': [500],
                  'min_split_gain': [0],
                  #'is_unbalance': [True, False],
                  #'subsample_freq': [1, 3],
                  #'drop_rate': [0.1],
                  #'skip_drop': [0.5],
                  'seed': [2261]
                  }
    min_score = (100, 100, 100)
    min_params = None
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=871)
    use_score = 0
    cnt = 0

    df_fast = pd.read_csv('../data/train_fast.csv', header=None, delimiter="\t")

    for params in ParameterGrid(all_params):
        list_score = []
        list_score2 = []
        list_best_iter = []
        all_pred = np.zeros(y_train.shape[0])
        for train, test in cv.split(x_train, y_train):
            df_fast.ix[train].to_csv('../data/train_fast_trn.csv', header=None, index=False)
            df_fast.ix[test].to_csv('../data/train_fast_val.csv', header=None, index=False)
            print(train.shape, test.shape)
            break
