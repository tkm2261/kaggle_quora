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


if __name__ == '__main__':
    from logging import StreamHandler, DEBUG, Formatter, FileHandler

    log_fmt = Formatter('%(asctime)s %(name)s %(lineno)d [%(levelname)s][%(funcName)s] %(message)s ')

    handler = StreamHandler()
    handler.setLevel('INFO')
    handler.setFormatter(log_fmt)
    logger.setLevel('INFO')
    logger.addHandler(handler)

    logger.info('load start')
    df_train = pd.read_csv('../data/train.csv')
    x_train = df_train[['is_duplicate', 'question1', 'question2']].values
    y_train = df_train['is_duplicate'].values
    del df_train
    gc.collect()

    logger.info('x_shape: {}'.format(x_train.shape))
    pos_rate = 0.165
    pos_num = y_train.sum()
    neg_num = y_train.shape[0] - y_train.sum()
    logger.info('pos_rate: %s, target pos_rate: %s, pos_num: %s' % (pos_num / y_train.shape[0], pos_rate, pos_num))

    w = (neg_num * pos_rate) / (pos_num * (1 - pos_rate))
    sample_weight = np.where(y_train == 1, w, 1)
    calc_pos_rate = (w * pos_num) / (w * pos_num + neg_num)
    logger.info('calc pos_rate: %s' % calc_pos_rate)

    logger.info('sampling start')

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=871)

    for train, test in cv.split(x_train, y_train):
        trn_x = x_train[train]
        val_x = x_train[test]
        trn_y = y_train[train]
        val_y = y_train[test]
        trn_w = sample_weight[train]
        val_w = sample_weight[test]

        with open('tfidf_val.pkl', 'rb') as f:
            pred, val_y, val_w = pickle.load(f)
        import pdb
        pdb.set_trace()
        pd.DataFrame(np.c_[pred.reshape(-1, 1), val_y.reshape(-1, 1), val_x]).to_csv('check_pred.csv')
