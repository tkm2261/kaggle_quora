
import os
import pickle
from logging import getLogger
import pandas as pd
import numpy as np
from sklearn import cross_validation
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.model_selection import GridSearchCV, ParameterGrid, StratifiedKFold
import xgboost as xgb
from lightgbm.sklearn import LGBMClassifier


logger = getLogger(__name__)


def train_lightgbm(verbose=True):
    """Train a boosted tree with LightGBM."""
    logger.info("Training with LightGBM")
    with open('vec300/train_data.pkl', 'rb') as f:
        x, y = pickle.load(f)
    x = pd.read_csv('count_tfidf_train.csv').values
    #x = np.c_[x, x2]
    pos_rate = 0.165
    pos_num = y.sum()
    neg_num = y.shape[0] - y.sum()
    logger.info('pos_rate: %s, target pos_rate: %s, pos_num: %s' % (pos_num / y.shape[0], pos_rate, pos_num))

    w = (neg_num * pos_rate) / (pos_num * (1 - pos_rate))
    sample_weight = np.where(y == 1, w, 1)
    calc_pos_rate = (w * pos_num) / (w * pos_num + neg_num)
    logger.info('calc pos_rate: %s' % calc_pos_rate)

    # x = np.c_[x, x2]
    logger.info('data size: {}'.format(x.shape))
    # 2017-03-18 14:25:47,709 __main__ 83 [INFO][train_lightgbm] best_param:
    # {'seed': 2261, 'n_estimators': 1000, 'subsample': 0.7, 'boosting_type':
    # 'gbdt', 'max_depth': 10, 'learning_rate': 0.06, 'min_child_weight': 0,
    # 'colsample_bytree': 0.5}
    all_params = {'max_depth': [5, 10, 15, 20],
                  'learning_rate': [0.1],  # [0.06, 0.1, 0.2],
                  'n_estimators': [3000],
                  'min_child_weight': [0],
                  'subsample': [0.7],
                  'colsample_bytree': [0.5],
                  'boosting_type': ['gbdt'],
                  #'num_leaves': [2, 3],
                  #'reg_alpha': [0.1, 0, 1],
                  #'reg_lambda': [0.1, 0, 1],
                  #'is_unbalance': [True, False],
                  #'subsample_freq': [1, 3],
                  'seed': [2261]
                  }
    min_score = (100, 100, 100)
    min_params = None
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=871)
    use_score = 0
    for params in ParameterGrid(all_params):
        list_score = []
        list_score2 = []
        list_best_iter = []
        for train, test in cv.split(x, y):
            trn_x = x[train]
            val_x = x[test]
            trn_y = y[train]
            val_y = y[test]
            trn_w = sample_weight[train]
            val_w = sample_weight[test]

            clf = LGBMClassifier(**params)
            clf.fit(trn_x, trn_y,
                    sample_weight=trn_w,
                    eval_sample_weight=[val_w],
                    eval_set=[(val_x, val_y)],
                    verbose=verbose,
                    # eval_metric='log_loss',
                    early_stopping_rounds=300
                    )

            _score = log_loss(val_y, clf.predict_proba(val_x)[:, 1], sample_weight=val_w)
            _score2 = - roc_auc_score(val_y, clf.predict_proba(val_x)[:, 1], sample_weight=val_w)
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

    """
    imp = pd.DataFrame(clf.feature_importances_, columns=['imp'])
    with open('features.py', 'a') as f:
        f.write('FEATURE = [' + ','.join(map(str, imp[imp['imp'] > 0].index.values)) + ']\n')
    """
    clf = LGBMClassifier(**min_params)
    clf.fit(x, y)

    return clf

if __name__ == "__main__":
    from logging import StreamHandler, DEBUG, Formatter, FileHandler

    log_fmt = Formatter('%(asctime)s %(name)s %(lineno)d [%(levelname)s][%(funcName)s] %(message)s ')
    handler = FileHandler('train.py.log', 'w')
    handler.setLevel(DEBUG)
    handler.setFormatter(log_fmt)
    logger.setLevel(DEBUG)
    logger.addHandler(handler)

    handler = StreamHandler()
    handler.setLevel(DEBUG)
    handler.setFormatter(log_fmt)
    logger.setLevel(DEBUG)
    logger.addHandler(handler)

    clf = train_lightgbm()
    # clf = train_xgboost()
    with open('model.pkl', 'wb') as f:
        pickle.dump(clf, f, -1)
