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
from ifidf_k import train_data, test_data

if __name__ == '__main__':
    from logging import StreamHandler, DEBUG, Formatter, FileHandler

    log_fmt = Formatter('%(asctime)s %(name)s %(lineno)d [%(levelname)s][%(funcName)s] %(message)s ')
    handler = FileHandler('tfidf_xgb.py.log', 'w')
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
    df_train = pd.read_csv('../data/train.csv')
    df_test = pd.read_csv('../data/test.csv')

    ################
    x_train = train_data()

    #####################

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

    all_params = {'max_depth': [5],
                  'n_estimators': [10000],
                  'learning_rate': [0.06],
                  'scale_pos_weight': [1],
                  'min_child_weight': [0],
                  'subsample': [0.9],
                  'colsample_bytree': [0.8],
                  'colsample_bylevel': [0.8],
                  #'booster': ['dart'],
                  'eta': [0.06],
                  #'normalize_type': ['forest'],
                  #'sample_type': ['weighted'],
                  #'rate_drop': [0.1],
                  #'skip_drop': [0.5],
                  'silent': [False],
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
                            early_stopping_rounds=100,
                            verbose_eval=1)

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
                    min_params['n_estimators'],
                    verbose_eval=1)

    with open('model_xgb.pkl', 'wb') as f:
        pickle.dump(clf, f, -1)
    del x_train
    gc.collect()

    with open('model_xgb.pkl', 'rb') as f:
        clf = pickle.load(f)
    #imp = pd.DataFrame(clf.feature_importances_, columns=['imp'])
    #imp_use = imp[imp['imp'] > 0].sort_values('imp', ascending=False)
    #logger.info('imp use {}'.format(imp_use.shape))
    # with open('features_train.py', 'w') as f:
    #    f.write('FEATURE = [' + ','.join(map(str, imp_use.index.values)) + ']\n')

    x_test = test_data()
    d_test = xgb.DMatrix(x_test, weight=sample_weight)
    logger.info('train end')
    p_test = clf.predict(d_test)
    sub = pd.DataFrame()
    sub['test_id'] = df_test['test_id']
    sub['is_duplicate'] = p_test
    sub.to_csv('submit.csv', index=False)
    logger.info('learn start')
