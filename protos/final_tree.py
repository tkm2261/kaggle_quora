from sklearn.model_selection import cross_val_predict
from sklearn.tree.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from lightgbm.sklearn import LGBMClassifier
import pandas
import pickle
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import GridSearchCV, ParameterGrid, StratifiedKFold, cross_val_predict
from tfidf_k import calc_weight
from sklearn.metrics import log_loss, roc_auc_score

from logging import StreamHandler, DEBUG, Formatter, FileHandler

log_fmt = Formatter('%(asctime)s %(name)s %(lineno)d [%(levelname)s][%(funcName)s] %(message)s ')

from logging import getLogger
logger = getLogger(__name__)

handler = StreamHandler()
handler.setLevel('INFO')
handler.setFormatter(log_fmt)
logger.setLevel('INFO')
logger.addHandler(handler)


aaa = pandas.read_csv('clique_data.csv')
sample_weight = calc_weight(aaa['label'].values)
# , 'emax', 'emin']  # ,  # 'l_score', 'r_score', 'm_score']  #
use_cols = ['cnum', 'pred', 'new', 'vmax', 'vmin', 'vavg']  # , 'emax', 'emin']

#'l_num', 'r_num', 'm_num']

x_train = aaa[use_cols].values
y_train = aaa['label'].values


all_params = {'max_depth': [14],
              'learning_rate': [0.02],  # [0.06, 0.1, 0.2],
              'n_estimators': [10000],
              'min_child_weight': [1],
              'colsample_bytree': [0.7],
              'boosting_type': ['gbdt'],
              #'num_leaves': [1300, 1500, 2000],
              'subsample': [0.99],
              'min_child_samples': [5],
              'reg_alpha': [0],
              'reg_lambda': [0],
              'max_bin': [500],
              'min_split_gain': [0.1],
              'silent': [True],
              'seed': [2261]
              }
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=871)
min_score = (100, 100, 100)
min_params = None

use_score = 0
logger.info('x size {}'.format(x_train.shape))
for params in tqdm(list(ParameterGrid(all_params))):

    cnt = 0
    list_score = []
    list_score2 = []
    list_best_iter = []
    all_pred = np.zeros(y_train.shape[0])

    for train, test in cv.split(x_train, y_train):
        trn_x = x_train[train]
        val_x = x_train[test]

        trn_y = y_train[train]
        val_y = y_train[test]

        trn_w = sample_weight[train]
        val_w = sample_weight[test]

        clf = LGBMClassifier(**params)
        clf.fit(trn_x, trn_y,
                sample_weight=trn_w,
                eval_sample_weight=[val_w],
                eval_set=[(val_x, val_y)],
                verbose=False,
                # eval_metric='logloss',
                early_stopping_rounds=100
                )
        pred = clf.predict_proba(val_x)[:, 1]
        _score = log_loss(val_y, pred, sample_weight=val_w)
        _score2 = - roc_auc_score(val_y, pred, sample_weight=val_w)
        list_score.append(_score)
        list_score2.append(_score2)
        if clf.best_iteration != -1:
            list_best_iter.append(clf.best_iteration)
        else:
            list_best_iter.append(params['n_estimators'])
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

final_tree = LGBMClassifier(**min_params)
final_tree.fit(x_train, y_train, sample_weight=sample_weight)
with open('final_tree.pkl', 'wb') as f:
    pickle.dump(final_tree, f, -1)
