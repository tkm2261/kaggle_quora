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

GRAPH = ['cnum',
         #'pred',
         #'new',
         'vmax',
         'vmin',
         'vavg',
         'l_num', 'r_num', 'm_num',
         'l_cnum_max', 'r_cnum_max', 'l_cnum_min', 'r_cnum_min', 'l_cnum_avg', 'r_cnum_avg',
         'l_eign_cent', 'r_eign_cent',
         'n_med'
         ]


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
    with open('tfidf_all_pred_final_0512.pkl', 'rb') as f:
        pred = pickle.load(f)

    df_train['pred'] = pred
    x = pd.read_csv('clique_data_0517.csv')
    for col in GRAPH:
        df_train[col] = x[col]

    df_train.to_csv('check_data.csv')
