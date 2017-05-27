
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
from features_stack import FEATURE

CHUNK_SIZE = 100000
GRAPH = ['cnum',
         'pred',
         #'new',
         'vmax',
         'vmin',
         'vavg',
         'appnum',
         'emax',
         'emin',
         'l_score', 'r_score', 'm_score',
         'l_num', 'r_num', 'm_num',
         'l_min', 'l_max', 'r_min', 'r_max',
         'l_cnum_max', 'r_cnum_max', 'l_cnum_min', 'r_cnum_min', 'l_cnum_avg', 'r_cnum_avg',
         'l_eign_cent', 'r_eign_cent',
         'n_med', 'med_min', 'med_max', 'med_avg',
         'med_l_min', 'med_l_max', 'med_l_avg',
         'med_r_min', 'med_r_max', 'med_r_avg'

         ]
GRAPH0520 = ['cnum',
             'pred',
             #'new',
             'vmax',
             'vmin',
             'vavg',
             'appnum',
             'emax',
             'emin',
             'l_score', 'r_score', 'm_score',
             'l_num', 'r_num', 'm_num',
             'l_min', 'l_max', 'r_min', 'r_max',
             'l_cnum_max', 'r_cnum_max', 'l_cnum_min', 'r_cnum_min', 'l_cnum_avg', 'r_cnum_avg',
             'l_eign_cent', 'r_eign_cent',
             'n_med', 'med_min', 'med_max', 'med_avg',
             'med_l_min', 'med_l_max', 'med_l_avg',
             'med_r_min', 'med_r_max', 'med_r_avg',
             'l_c_max', 'l_c_min', 'l_c_avg',
             'r_c_max', 'r_c_min', 'r_c_avg'

             ]
GRAPHA = ['cnum',
          'pred',
          ]

GRAPH0 = ['cnum',
          'pred',
          #'new',
          'vmax',
          'vmin',
          'vavg',
          'appnum',
          'emax',
          'emin',
          'l_score', 'r_score', 'm_score',
          'l_num', 'r_num', 'm_num',
          'l_min', 'l_max', 'r_min', 'r_max',
          ]

GRAPH2 = [
    'pred',
    #'new',
    'vmax',
    'vmin',
    'vavg',
    'appnum',
    'emax',
    'emin',
    'l_score', 'r_score', 'm_score',
    'l_min', 'l_max', 'r_min', 'r_max']


def train_data():
    logger.info('start')

    with open('tfidf_all_pred2_0512.pkl', 'rb') as f:
        x = pickle.load(f).astype(np.float32)
    x_train = x  # np.c_[x_train, x]
    logger.info('{}'.format(x_train.shape))

    with open('tfidf_all_pred2_0515.pkl', 'rb') as f:
        x = pickle.load(f).astype(np.float32)
    x_train = np.c_[x_train, x]
    logger.info('{}'.format(x_train.shape))

    with open('tfidf_all_pred2_0516.pkl', 'rb') as f:
        x = pickle.load(f).astype(np.float32)
    x_train = np.c_[x_train, x]
    logger.info('{}'.format(x_train.shape))

    with open('tfidf_all_pred2_0517.pkl', 'rb') as f:
        x = pickle.load(f).astype(np.float32)
    x_train = np.c_[x_train, x]
    logger.info('{}'.format(x_train.shape))

    with open('tfidf_all_pred2_0518.pkl', 'rb') as f:
        x = pickle.load(f).astype(np.float32)
    x_train = np.c_[x_train, x]
    logger.info('{}'.format(x_train.shape))

    with open('tfidf_all_pred2_0520.pkl', 'rb') as f:
        x = pickle.load(f).astype(np.float32)
    x_train = np.c_[x_train, x]
    logger.info('{}'.format(x_train.shape))

    with open('tfidf_all_pred2_0525.pkl', 'rb') as f:
        x = pickle.load(f).astype(np.float32)
    x_train = np.c_[x_train, x]
    logger.info('{}'.format(x_train.shape))

    x = pd.read_csv('clique_data_0517.csv')[GRAPH].values
    x_train = np.c_[x_train, x]
    logger.info('{}'.format(x_train.shape))

    x = pd.read_csv('clique_data_0518.csv')[GRAPH].values
    x_train = np.c_[x_train, x]
    logger.info('{}'.format(x_train.shape))

    x = pd.read_csv('clique_data_0520.csv')[GRAPH0520].values
    x_train = np.c_[x_train, x]
    logger.info('{}'.format(x_train.shape))

    x = pd.read_csv('clique_data_0525.csv')[GRAPH0520].values
    x_train = np.c_[x_train, x]
    logger.info('{}'.format(x_train.shape))

    x = pd.read_csv('clique_data_0512.csv')[GRAPH].values
    x_train = np.c_[x_train, x]
    logger.info('{}'.format(x_train.shape))

    x = pd.read_csv('loop_data_0520.csv')[GRAPH].values
    x_train = np.c_[x_train, x]
    logger.info('{}'.format(x_train.shape))

    x = pd.read_csv('loop2_data_0520.csv')[GRAPH].values
    x_train = np.c_[x_train, x]
    logger.info('{}'.format(x_train.shape))

    x = pd.read_csv('loop3_data_0520.csv')[GRAPH].values
    x_train = np.c_[x_train, x]
    logger.info('{}'.format(x_train.shape))

    x = pd.read_csv('cluster_data_0517.csv')[GRAPH].values
    x_train = np.c_[x_train, x]

    x = pd.read_csv('clique_data_0509.csv')[GRAPH0].values
    x_train = np.c_[x_train, x]
    logger.info('{}'.format(x_train.shape))

    x = pd.read_csv('clique_data_2_0509.csv')[GRAPH2].values
    x_train = np.c_[x_train, x]
    logger.info('{}'.format(x_train.shape))

    x = pd.read_csv('clique_data_0506.csv')[GRAPH0].values
    x_train = np.c_[x_train, x]
    logger.info('{}'.format(x_train.shape))

    with open('train_magic.pkl', 'rb') as f:
        x = pickle.load(f).astype(np.float32)
    x_train = np.c_[x_train, x]
    logger.info('{}'.format(x_train.shape))

    x = pd.read_csv('train_magic2.csv').values.astype(np.float32)
    x_train = np.c_[x_train, x]
    logger.info('{}'.format(x_train.shape))

    x_train[np.isnan(x_train)] = -100
    x_train[np.isinf(x_train)] = -100
    #x_train = x_train[:, FEATURE]
    return x_train

import dask.array as da


def test_data():
    logger.info('start')

    with open('test_preds2_0512.pkl', 'rb') as f:
        preds = pickle.load(f).astype(np.float32)
    x = preds.reshape((-1, 1))
    x = da.from_array(x, chunks=CHUNK_SIZE)
    x_test = x  # da.concatenate([x_test, x], axis=1)
    logger.info('{}'.format(x_test.shape))

    with open('test_preds2_0515.pkl', 'rb') as f:
        preds = pickle.load(f).astype(np.float32)
    x = preds.reshape((-1, 1))
    x = da.from_array(x, chunks=CHUNK_SIZE)
    x_test = da.concatenate([x_test, x], axis=1)
    logger.info('{}'.format(x_test.shape))

    with open('test_preds2_0516.pkl', 'rb') as f:
        preds = pickle.load(f).astype(np.float32)
    x = preds.reshape((-1, 1))
    x = da.from_array(x, chunks=CHUNK_SIZE)
    x_test = da.concatenate([x_test, x], axis=1)
    logger.info('{}'.format(x_test.shape))

    with open('test_preds2_0517.pkl', 'rb') as f:
        preds = pickle.load(f).astype(np.float32)
    x = preds.reshape((-1, 1))
    x = da.from_array(x, chunks=CHUNK_SIZE)
    x_test = da.concatenate([x_test, x], axis=1)
    logger.info('{}'.format(x_test.shape))

    with open('test_preds2_0518.pkl', 'rb') as f:
        preds = pickle.load(f).astype(np.float32)
    x = preds.reshape((-1, 1))
    x = da.from_array(x, chunks=CHUNK_SIZE)
    x_test = da.concatenate([x_test, x], axis=1)
    logger.info('{}'.format(x_test.shape))

    with open('test_preds2_0520.pkl', 'rb') as f:
        preds = pickle.load(f).astype(np.float32)
    x = preds.reshape((-1, 1))
    x = da.from_array(x, chunks=CHUNK_SIZE)
    x_test = da.concatenate([x_test, x], axis=1)
    logger.info('{}'.format(x_test.shape))

    with open('test_preds2_0525.pkl', 'rb') as f:
        preds = pickle.load(f).astype(np.float32)
    x = preds.reshape((-1, 1))
    x = da.from_array(x, chunks=CHUNK_SIZE)
    x_test = da.concatenate([x_test, x], axis=1)
    logger.info('{}'.format(x_test.shape))

    x = pd.read_csv('clique_data_test_0517.csv')[GRAPH].values
    x = da.from_array(x, chunks=CHUNK_SIZE)
    x_test = da.concatenate([x_test, x], axis=1)
    logger.info('{}'.format(x_test.shape))

    x = pd.read_csv('clique_data_test_0518.csv')[GRAPH].values
    x = da.from_array(x, chunks=CHUNK_SIZE)
    x_test = da.concatenate([x_test, x], axis=1)
    logger.info('{}'.format(x_test.shape))

    x = pd.read_csv('clique_data_test_0520.csv')[GRAPH0520].values
    x = da.from_array(x, chunks=CHUNK_SIZE)
    x_test = da.concatenate([x_test, x], axis=1)
    logger.info('{}'.format(x_test.shape))

    x = pd.read_csv('clique_data_test_0525.csv')[GRAPH0520].values
    x = da.from_array(x, chunks=CHUNK_SIZE)
    x_test = da.concatenate([x_test, x], axis=1)
    logger.info('{}'.format(x_test.shape))

    x = pd.read_csv('clique_data_test_0512.csv')[GRAPH].values
    x = da.from_array(x, chunks=CHUNK_SIZE)
    x_test = da.concatenate([x_test, x], axis=1)
    logger.info('{}'.format(x_test.shape))

    x = pd.read_csv('loop_data_test_0520.csv')[GRAPH].values
    x = da.from_array(x, chunks=CHUNK_SIZE)
    x_test = da.concatenate([x_test, x], axis=1)
    logger.info('{}'.format(x_test.shape))

    x = pd.read_csv('loop2_data_test_0520.csv')[GRAPH].values
    x = da.from_array(x, chunks=CHUNK_SIZE)
    x_test = da.concatenate([x_test, x], axis=1)
    logger.info('{}'.format(x_test.shape))

    x = pd.read_csv('loop3_data_test_0520.csv')[GRAPH].values
    x = da.from_array(x, chunks=CHUNK_SIZE)
    x_test = da.concatenate([x_test, x], axis=1)
    logger.info('{}'.format(x_test.shape))

    x = pd.read_csv('cluster_data_test_0517.csv')[GRAPH].values
    x = da.from_array(x, chunks=CHUNK_SIZE)
    x_test = da.concatenate([x_test, x], axis=1)
    logger.info('{}'.format(x_test.shape))

    x = pd.read_csv('clique_data_test_0509.csv')[GRAPH0].values
    x = da.from_array(x, chunks=CHUNK_SIZE)
    x_test = da.concatenate([x_test, x], axis=1)
    logger.info('{}'.format(x_test.shape))

    x = pd.read_csv('clique_data_test_2_0509.csv')[GRAPH2].values
    x = da.from_array(x, chunks=CHUNK_SIZE)
    x_test = da.concatenate([x_test, x], axis=1)
    logger.info('{}'.format(x_test.shape))

    x = pd.read_csv('clique_data_test_0506.csv')[GRAPH0].values
    x = da.from_array(x, chunks=CHUNK_SIZE)
    x_test = da.concatenate([x_test, x], axis=1)
    logger.info('{}'.format(x_test.shape))

    with open('test_magic.pkl', 'rb') as f:
        x = pickle.load(f).astype(np.float32)
        x = da.from_array(x, chunks=CHUNK_SIZE)
    x_test = da.concatenate([x_test, x], axis=1)
    logger.info('{}'.format(x_test.shape))

    x = pd.read_csv('test_magic2.csv').values
    x = x.reshape((-1, 1))
    x = da.from_array(x, chunks=CHUNK_SIZE)
    x_test = da.concatenate([x_test, x], axis=1)
    logger.info('{}'.format(x_test.shape))

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


def _calc_weight(y_train, pos_rate=0.165):

    neg_rate = 1 - pos_rate
    pos_num = y_train.sum()
    neg_num = y_train.shape[0] - y_train.sum()

    logger.info('pos_rate: %s, target pos_rate: %s, pos_num: %s' % (pos_num / y_train.shape[0], pos_rate, pos_num))

    w = (pos_num * neg_rate) / (neg_num * (1 - neg_rate))
    sample_weight = np.where(y_train == 0, w, 1)
    calc_pos_rate = pos_num / (pos_num + w * neg_num)
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

    all_params = {'max_depth': [6],
                  'learning_rate': [0.001],  # [0.06, 0.1, 0.2],
                  'n_estimators': [100000],
                  'min_child_weight': [20],
                  'colsample_bytree': [0.8],
                  #'boosting_type': ['dart'],  # ['gbdt'],
                  #'xgboost_dart_mode': [False],
                  'num_leaves': [1280],
                  'subsample': [0.9],
                  'min_child_samples': [20],
                  'reg_alpha': [1],
                  'reg_lambda': [0],
                  'max_bin': [5000],
                  'min_split_gain': [0.1],
                  'silent': [True],
                  'seed': [6436]
                  }

    min_score = (100, 100, 100)
    min_params = None
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=871)
    use_score = 0

    for params in tqdm(list(ParameterGrid(all_params))):

        cnt = 0
        list_score = []
        list_score2 = []
        list_best_iter = []
        all_pred = np.zeros(y_train.shape[0])
        for train, test in cv.split(x_train, y_train):
            trn_x = x_train[train]
            val_x = x_train[test]

            # trn_xr = x_train_rev[train]
            # val_xr = x_train_rev[test]

            trn_y = y_train[train]
            val_y = y_train[test]
            trn_w = sample_weight[train]
            val_w = sample_weight[test]

            # trn_x = np.r_[trn_x, trn_xr]
            # trn_y = np.r_[trn_y, trn_y]
            # trn_w = np.r_[trn_w, trn_w]

            trn_ym = y_train_mod[train]
            val_ym = y_train_mod[test]

            trn_wm = sample_weight_mod[train]
            val_wm = sample_weight_mod[test]

            trn_p = cross_pred[train]
            val_p = cross_pred[test]

            # trn_y[trn_p > 0.9] = 1
            # reg = LogisticRegression(C=0.1, solver='sag', n_jobs=-1)
            # pred_x = cross_val_predict(reg, trn_x, trn_y, cv=5, n_jobs=-1)
            # trn_x = np.c_[trn_x, pred_x]

            clf = LGBMClassifier(**params)
            clf.fit(trn_x, trn_y,
                    sample_weight=trn_w,
                    eval_sample_weight=[val_w],
                    eval_set=[(val_x, val_y)],
                    verbose=True,
                    # eval_metric='logloss',
                    early_stopping_rounds=300
                    )
            pred = clf.predict_proba(val_x)[:, 1]
            all_pred[test] = pred

            _score = log_loss(val_y, pred, sample_weight=val_w)
            _score2 = - roc_auc_score(val_y, pred, sample_weight=val_w)
            # logger.debug('   _score: %s' % _score)
            list_score.append(_score)
            list_score2.append(_score2)
            if clf.best_iteration != -1:
                list_best_iter.append(clf.best_iteration)
            else:
                list_best_iter.append(params['n_estimators'])
            # break
        with open('tfidf_all_pred_final_0518.pkl', 'wb') as f:
            pickle.dump(all_pred, f, -1)

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

    # for params in ParameterGrid(all_params):
    #    min_params = params

    # x_train = np.r_[x_train, x_train_rev]
    # y_train = np.r_[y_train, y_train]
    # sample_weight = np.r_[sample_weight, sample_weight]

    clf = LGBMClassifier(**min_params)
    clf.fit(x_train, y_train, sample_weight=sample_weight)
    with open('model_ft.pkl', 'wb') as f:
        pickle.dump(clf, f, -1)
    del x_train
    gc.collect()

    with open('model_ft.pkl', 'rb') as f:
        clf = pickle.load(f)
    imp = pd.DataFrame(clf.feature_importances_, columns=['imp'])
    n_features = imp.shape[0]
    imp_use = imp[imp['imp'] > 0].sort_values('imp', ascending=False)
    logger.info('imp use {}'.format(imp_use.shape))
    with open('features_train.py', 'w') as f:
        f.write('FEATURE = [' + ','.join(map(str, imp_use.index.values)) + ']\n')

    x_test = test_data()

    if x_test.shape[1] != n_features:
        raise Exception('Not match feature num: %s %s' % (x_test.shape[1], n_features))
    logger.info('train end')
    preds = []
    for i in range(int(df_test.shape[0] / CHUNK_SIZE) + 1):
        d = x_test[i * CHUNK_SIZE: (i + 1) * CHUNK_SIZE].compute()

        d[np.isnan(d)] = -100
        d[np.isinf(d)] = -100

        p_test = clf.predict_proba(d)
        preds.append(p_test)
    p_test = np.concatenate(preds)[:, 1]
    with open('test_preds2_final_0518.pkl', 'wb') as f:
        pickle.dump(p_test, f, -1)

    sub = pd.DataFrame()
    sub['test_id'] = df_test['test_id']
    sub['is_duplicate'] = p_test
    sub.to_csv('submit_fi.csv', index=False)
    logger.info('learn start')
