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
from scipy.sparse import hstack
from logging import getLogger
logger = getLogger(__name__)


def feat(idf, count, tfidf):
    from features_idf import FEATURE as idf_feat
    from features_tfidf import FEATURE as tfidf_feat
    from features_cnt import FEATURE as cnt_feat

    idf = idf  # [:, idf_feat]
    tfidf = tfidf  # [:, tfidf_feat]
    count = count  # [:, cnt_feat]
    return hstack([idf, count, tfidf], format='csr')

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
    df_train = pd.read_csv('../data/train.csv')
    df_test = pd.read_csv('../data/test.csv')

    with open('train_sparse.pkl', 'rb') as f:
        idf, count, tfidf = pickle.load(f)
        x_train = feat(idf, count, tfidf)
    with open('test_sparse.pkl', 'rb') as f:
        idf, count, tfidf = pickle.load(f)
        x_test = feat(idf, count, tfidf)

    from features_tic2 import FEATURE
    feat = np.array(FEATURE[1:1001]) - 1
    with open('train_tic_val_1000.pkl', 'wb') as f:
        pickle.dump(x_train[:, feat], f, -1)
    with open('test_tic_val_1000.pkl', 'wb') as f:
        pickle.dump(x_test[:, feat], f, -1)
    exit()

    """
    x_train = x_train.values
    x_test = x_test.values
    """
    y_train = df_train['is_duplicate'].values
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

    from sklearn.cross_validation import train_test_split
    #x_train, x_valid, y_train, y_valid = train_test_split(x_train, y_train, test_size=0.2, random_state=4242)
    all_params = {'max_depth': [10],
                  'learning_rate': [0.1],  # [0.06, 0.1, 0.2],
                  'n_estimators': [2000],
                  'min_child_weight': [3],
                  'colsample_bytree': [0.5],
                  'boosting_type': ['gbdt'],
                  'subsample': [0.9],
                  'min_child_samples': [10],
                  #'num_leaves': [300],
                  'reg_alpha': [0.1],
                  'reg_lambda': [0.1],
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
    for params in ParameterGrid(all_params):
        list_score = []
        list_score2 = []
        list_best_iter = []

        for test, train in cv.split(x_train, y_train):
            trn_x = x_train[train]
            val_x = x_train[test]
            trn_y = y_train[train]
            val_y = y_train[test]
            trn_w = sample_weight[train]
            val_w = sample_weight[test]
            with open('tfidf_val.pkl', 'rb') as f:
                pred, _, _ = pickle.load(f)

            trn_x = hstack([pred.reshape((-1, 1)), trn_x], format='csr')
            #reg = LogisticRegression(C=0.1, solver='sag', n_jobs=-1)
            #pred_x = cross_val_predict(reg, trn_x, trn_y, cv=5, n_jobs=-1)
            #trn_x = np.c_[trn_x, pred_x]

            clf = LGBMClassifier(**params)
            clf.fit(trn_x, trn_y,
                    sample_weight=trn_w,
                    eval_sample_weight=[trn_w],
                    eval_set=[(trn_x, trn_y)],
                    verbose=True,
                    # eval_metric='logloss',
                    early_stopping_rounds=300
                    )
            imp = pd.DataFrame(clf.feature_importances_, columns=['imp'])
            imp_use = imp[imp['imp'] > 0].sort_values('imp', ascending=False)
            logger.info('imp use {}'.format(imp_use.shape))
            with open('features_tic2.py', 'w') as f:
                f.write('FEATURE = [' + ','.join(map(str, imp_use.index.values)) + ']\n')

            pred = clf.predict_proba(val_x)[:, 1]
            with open('sparse_val.pkl', 'wb') as f:
                pickle.dump((pred, val_y, val_w), f, -1)
            _score = log_loss(val_y, pred, sample_weight=val_w)
            _score2 = - roc_auc_score(val_y, pred, sample_weight=val_w)
            # logger.debug('   _score: %s' % _score)
            list_score.append(_score)
            list_score2.append(_score2)
            if clf.best_iteration != -1:
                list_best_iter.append(clf.best_iteration)
            else:
                list_best_iter.append(params['n_estimators'])
            imp = pd.DataFrame(clf.feature_importances_, columns=['imp'])
            imp_use = imp[imp['imp'] > 0].sort_values('imp', ascending=False)
            logger.info('imp use {}'.format(imp_use.shape))
            with open('features_tic.py', 'w') as f:
                f.write('FEATURE = [' + ','.join(map(str, imp_use.index.values)) + ']\n')
            with open('train_tic_300.pkl', 'wb') as f:
                pickle.dump(x_train[:, imp_use.index.values[:300]], f, -1)
            with open('test_tic_300.pkl', 'wb') as f:
                pickle.dump(x_test[:, imp_use.index.values[:300]], f, -1)
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

    clf = LGBMClassifier(**min_params)
    clf.fit(x_train, y_train, sample_weight=sample_weight)

    p_test = clf.predict_proba(x_test)[:, 1]
    sub = pd.DataFrame()
    sub['test_id'] = df_test['test_id']
    sub['is_duplicate'] = p_test
    sub.to_csv('submit.csv', index=False)
    logger.info('learn start')
    import xgboost as xgb
    params = {}
    params['objective'] = 'binary:logistic'
    params['eval_metric'] = 'logloss'
    params['eta'] = 0.02
    params['max_depth'] = 4

    """
    d_train = xgb.DMatrix(x_tra, label=y_tra, weight=trn_w)
    d_valid = xgb.DMatrix(x_val, label=y_val, weight=val_w)

    watchlist = [(d_train, 'train'), (d_valid, 'valid')]

    bst = xgb.train(params, d_train, 400, watchlist, early_stopping_rounds=50, verbose_eval=1)
    """
