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
from tqdm import tqdm
from features_tmp import FEATURE

CHUNK_SIZE = 100000

GRAPH = ['cnum', 'pred', 'new', 'vmax',
         'vmin', 'vavg']  # , 'l_num', 'r_num', 'm_num']


def train_data():
    logger.info('start')

    with open('tfidf_all_pred2_2.pkl', 'rb') as f:
        x = pickle.load(f).astype(np.float32)
    x_train = x  # np.c_[x_train, x]
    logger.info('{}'.format(x_train.shape))

    #x_train = x_train[:, FEATURE]
    x = pd.read_csv('clique_data.csv')[GRAPH].fillna(-100).values
    x_train = np.c_[x_train, x]
    logger.info('{}'.format(x_train.shape))

    #x = pd.read_csv('clique_data2.csv')[GRAPH].values
    #x_train = np.c_[x_train, x]
    # logger.info('{}'.format(x_train.shape))
    """
    with open('train_magic.pkl', 'rb') as f:
        x = pickle.load(f).astype(np.float32)
    x_train = np.c_[x_train, x]
    logger.info('{}'.format(x_train.shape))
    """

    """
    with open('tfidf_all_pred_final_0502.pkl', 'rb') as f:
        x = pickle.load(f).astype(np.float32)
    x_train = np.c_[x_train, x]
    logger.info('{}'.format(x_train.shape))
    """
    x_train[np.isnan(x_train)] = -100
    x_train[np.isinf(x_train)] = -100
    return x_train

import dask.array as da


def test_data():
    logger.info('start')
    with open('test_preds2_2.pkl', 'rb') as f:
        preds = pickle.load(f).astype(np.float32)

    x = preds.reshape((-1, 1))
    x = da.from_array(x, chunks=CHUNK_SIZE)
    x_test = x  # da.concatenate([x_test, x], axis=1)

    x = pd.read_csv('clique_data_test.csv')[GRAPH].fillna(-100).values
    x = da.from_array(x, chunks=CHUNK_SIZE)
    x_test = da.concatenate([x_test, x], axis=1)

    #x = pd.read_csv('clique_data_test2.csv')[GRAPH].values
    #x = da.from_array(x, chunks=CHUNK_SIZE)
    #x_test = da.concatenate([x_test, x], axis=1)

    """
    with open('test_magic.pkl', 'rb') as f:
        x = pickle.load(f).astype(np.float32)

    x = da.from_array(x, chunks=CHUNK_SIZE)
    x_test = da.concatenate([x_test, x], axis=1)
    """
    """
    with open('model_ft.pkl', 'rb') as f:
        clf = pickle.load(f)

    preds = []
    for i in range(int(x_test.shape[0] / CHUNK_SIZE) + 1):
        logger.debug('chunk %s' % i)
        d = x_test[i * CHUNK_SIZE: (i + 1) * CHUNK_SIZE].compute()
        p_test = clf.predict_proba(d, num_iteration=6459)
        preds.append(p_test)
        del d
        gc.collect()
    preds = np.concatenate(preds)[:, 1]
    x = preds.reshape((-1, 1))
    x = da.from_array(x, chunks=CHUNK_SIZE)
    x_test = da.concatenate([x_test, x], axis=1)
    """
    #x_test = x_test[:, FEATURE]

    return x_test


def calc_weight(y_train, pos_rate=0.165):

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
    # x_train_rev = train_data_rev()
    x_train = train_data()
    logger.info('x_shape: {}'.format(x_train.shape))
    #####################

    y_train = df_train['is_duplicate'].values
    with open('tfidf_all_pred2_2.pkl', 'rb') as f:
        cross_pred = pickle.load(f).astype(np.float32)
    y_train_mod = y_train.copy()
    # y_train_mod[cross_pred > 0.99] = 1

    sample_weight = calc_weight(y_train)
    sample_weight_mod = calc_weight(y_train_mod)

    del df_train
    gc.collect()

    # calc_pos_rate = (pos_num) / (pos_num + w * neg_num)
    # logger.info('calc pos_rate: %s' % calc_pos_rate)

    logger.info('sampling start')

    from sklearn.cross_validation import train_test_split

    # x_train, x_valid, y_train, y_valid = train_test_split(x_train, y_train, test_size=0.2, random_state=4242)

    all_params = {'max_depth': [5],
                  'n_estimators': [10000],
                  'min_child_weight': [0],
                  'subsample': [0.9],
                  'colsample_bytree': [0.8],
                  'colsample_bylevel': [0.8],
                  'booster': ['dart'],
                  'eta': [0.06],
                  #'normalize_type': ['forest'],
                  #'sample_type': ['weighted'],
                  #'rate_drop': [0.1],
                  #'skip_drop': [0.5],
                  'silent': [True],
                  'eval_metric': ['logloss'],
                  'objective': ['binary:logistic']
                  }

    min_score = (100, 100, 100)
    min_params = None
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=871)
    use_score = 0
    for params in ParameterGrid(all_params):
        list_score = []
        list_score2 = []
        list_best_iter = []

        for train, test in cv.split(x_train, y_train):
            trn_x = x_train[train]
            val_x = x_train[test]
            trn_y = y_train[train]
            val_y = y_train[test]
            trn_w = sample_weight[train]
            val_w = sample_weight[test]

            d_train = xgb.DMatrix(trn_x, label=trn_y, weight=trn_w)
            d_valid = xgb.DMatrix(val_x, label=val_y, weight=val_w)

            watchlist = [(d_train, 'train'), (d_valid, 'valid')]

            clf = xgb.train(params,
                            d_train,
                            params['n_estimators'],
                            watchlist,
                            early_stopping_rounds=100)
            # verbose_eval=0)

            pred = clf.predict(d_valid)
            # with open('tfidf_val.pkl', 'wb') as f:
            #    pickle.dump((pred, val_y, val_w), f, -1)

            _score = log_loss(val_y, clf.predict(d_valid), sample_weight=val_w)
            _score2 = - roc_auc_score(val_y, clf.predict(d_valid), sample_weight=val_w)
            # logger.debug('   _score: %s' % _score)
            list_score.append(_score)
            list_score2.append(_score2)
            if clf.best_iteration != -1:
                list_best_iter.append(clf.best_iteration)
            else:
                list_best_iter.append(params['n_estimators'])
            break
        logger.info('trees: {}'.format(list_best_iter))
        params['n_estimators'] = np.mean(list_best_iter, dtype=int)
        score = (np.mean(list_score), np.min(list_score), np.max(list_score))
        score2 = (np.mean(list_score2), np.min(list_score2), np.max(list_score2))

        logger.info('param: %s' % (params))
        logger.info('loss: {} (avg min max {})'.format(score[use_score], score))
        logger.info('score: {} (avg min max {})'.format(score2[use_score], score2))
        if min_score[use_score] > score[use_score]:
            min_score = score
            min_score2 = score2
            min_params = params
        logger.info('best score: {} {}'.format(min_score[use_score], min_score))
        logger.info('best score2: {} {}'.format(min_score2[use_score], min_score2))
        logger.info('best_param: {}'.format(min_params))

    gc.collect()

    d_train = xgb.DMatrix(x_train, label=y_train, weight=sample_weight)
    # for params in ParameterGrid(all_params):
    #    min_params = params
    clf = xgb.train(min_params,
                    d_train,
                    min_params['n_estimators'])
    # verbose_eval=0)

    with open('model_xgb.pkl', 'wb') as f:
        pickle.dump(clf, f, -1)
    del x_train
    gc.collect()

    #imp = pd.DataFrame(clf.feature_importances_, columns=['imp'])
    #imp_use = imp[imp['imp'] > 0].sort_values('imp', ascending=False)
    #logger.info('imp use {}'.format(imp_use.shape))
    # with open('features_train.py', 'w') as f:
    #    f.write('FEATURE = [' + ','.join(map(str, imp_use.index.values)) + ']\n')

    x_test = test_data()
    #d_test = xgb.DMatrix(x_test)
    logger.info('train end')
    with open('model_xgb.pkl', 'rb') as f:
        clf = pickle.load(f)
    print(clf)

    #p_test = clf.predict(d_test)

    preds = []
    for i in range(int(df_test.shape[0] / CHUNK_SIZE) + 1):
        d = x_test[i * CHUNK_SIZE: (i + 1) * CHUNK_SIZE].compute()
        d = xgb.DMatrix(d)

        p_test = clf.predict(d)
        preds.append(p_test)
    p_test = np.concatenate(preds)  # [:, 1]

    sub = pd.DataFrame()
    sub['test_id'] = df_test['test_id']
    sub['is_duplicate'] = p_test
    sub.to_csv('submit_xgb.csv', index=False)
    logger.info('learn start')
