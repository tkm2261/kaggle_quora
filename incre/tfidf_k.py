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

#from features_tmp import FEATURE

CHUNK_SIZE = 100000


def train_data():
    logger.info('start')
    x_train = pd.read_csv('kernel_train3.csv').values.astype(np.float32)
    x = pd.read_csv('count_tfidf_train_inc.csv').values.astype(np.float32)
    x_train = np.c_[x_train, x]

    x = pd.read_csv('kernel_train.csv').values.astype(np.float32)
    x_train = np.c_[x_train, x]

    x = pd.read_csv('count_tfidf_train_clean2_inc.csv').values.astype(np.float32)
    x_train = np.c_[x_train, x]

    logger.info('1')
    x = pd.read_csv('count_tfidf_norm_train_clean2_inc.csv').values.astype(np.float32)
    x_train = np.c_[x_train, x]
    x = pd.read_csv('count_tfidf_norm_train_inc.csv').values.astype(np.float32)
    x_train = np.c_[x_train, x]

    with open('train_svo.pkl', 'rb') as f:
        x = pickle.load(f)
    x_train = np.c_[x_train, x]

    with open('train_svo3.pkl', 'rb') as f:
        x = pickle.load(f)
    x_train = np.c_[x_train, x]

    with open('train_5w1h.pkl', 'rb') as f:
        x = pickle.load(f)
    x_train = np.c_[x_train, x]
    with open('train_rest_sim3.pkl', 'rb') as f:
        x = pickle.load(f)
    x_train = np.c_[x_train, x]

    with open('train_tag.pkl', 'rb') as f:
        x = pickle.load(f)
    x_train = np.c_[x_train, x]
    with open('train_ent.pkl', 'rb') as f:
        x = pickle.load(f)
    x_train = np.c_[x_train, x]

    logger.info('6')

    with open('fast_train_clean2_inc2.pkl', 'rb') as f:
        x = pickle.load(f).astype(np.float32)
    x_train = np.c_[x_train, x]
    with open('fast_train_clean2_low_max2.pkl', 'rb') as f:
        x = pickle.load(f).astype(np.float32)
    x_train = np.c_[x_train, x]
    logger.info('7')

    with open('glove_train_clean2_2.pkl', 'rb') as f:
        x = pickle.load(f).astype(np.float32)
    x_train = np.c_[x_train, x]

    with open('glove_train_clean2_max2.pkl', 'rb') as f:
        x = pickle.load(f).astype(np.float32)
    x_train = np.c_[x_train, x]
    logger.info('8')
    """
    with open('../sense2vec/train_sense.pkl', 'rb') as f:
        x = pickle.load(f).astype(np.float32)
    x_train = np.c_[x_train, x]
    """
    # with open('tfidf_all_pred2.pkl', 'rb') as f:
    #    x = pickle.load(f).astype(np.float32)
    #x_train = np.c_[x_train, x]

    x_train[np.isnan(x_train)] = -100
    x_train[np.isinf(x_train)] = -100
    return x_train  # [:, FEATURE + [-1]]


import dask.array as da


def test_data():
    logger.info('start')
    x_test = pd.read_csv('kernel_test3.csv').values.astype(np.float32)
    x = pd.read_csv('count_tfidf_test_inc.csv').values.astype(np.float32)
    x_test = da.from_array(x_test, chunks=CHUNK_SIZE)

    x = da.from_array(x, chunks=CHUNK_SIZE)
    x_test = da.concatenate([x_test, x], axis=1)

    x = pd.read_csv('kernel_test.csv').values.astype(np.float32)
    x = da.from_array(x, chunks=CHUNK_SIZE)
    x_test = da.concatenate([x_test, x], axis=1)

    x = pd.read_csv('count_tfidf_test_clean2_inc.csv').values.astype(np.float32)
    x = da.from_array(x, chunks=CHUNK_SIZE)
    x_test = da.concatenate([x_test, x], axis=1)

    logger.info('1')
    x = pd.read_csv('count_tfidf_norm_test_clean2_inc.csv').values.astype(np.float32)
    x = da.from_array(x, chunks=CHUNK_SIZE)
    x_test = da.concatenate([x_test, x], axis=1)

    x = pd.read_csv('count_tfidf_norm_test_inc.csv').values.astype(np.float32)
    x = da.from_array(x, chunks=CHUNK_SIZE)
    x_test = da.concatenate([x_test, x], axis=1)

    with open('test_svo.pkl', 'rb') as f:
        x = pickle.load(f)
        x = da.from_array(x, chunks=CHUNK_SIZE)
    x_test = da.concatenate([x_test, x], axis=1)

    with open('test_svo3.pkl', 'rb') as f:
        x = pickle.load(f)
        x = da.from_array(x, chunks=CHUNK_SIZE)
    x_test = da.concatenate([x_test, x], axis=1)

    with open('test_5w1h.pkl', 'rb') as f:
        x = pickle.load(f)
        x = da.from_array(x, chunks=CHUNK_SIZE)
    x_test = da.concatenate([x_test, x], axis=1)

    with open('test_rest_sim3.pkl', 'rb') as f:
        x = pickle.load(f)
        x = da.from_array(x, chunks=CHUNK_SIZE)
    x_test = da.concatenate([x_test, x], axis=1)

    with open('test_tag.pkl', 'rb') as f:
        x = pickle.load(f)
        x = da.from_array(x, chunks=CHUNK_SIZE)
    x_test = da.concatenate([x_test, x], axis=1)

    with open('test_ent.pkl', 'rb') as f:
        x = pickle.load(f)
        x = da.from_array(x, chunks=CHUNK_SIZE)
    x_test = da.concatenate([x_test, x], axis=1)

    logger.info('6')
    with open('../fasttext/fast_test_clean2_low2.pkl', 'rb') as f:
        x = pickle.load(f).astype(np.float32)
        x = da.from_array(x, chunks=CHUNK_SIZE)
    x_test = da.concatenate([x_test, x], axis=1)

    with open('../fasttext/fast_test_clean2_low_max2.pkl', 'rb') as f:
        x = pickle.load(f).astype(np.float32)
        x = da.from_array(x, chunks=CHUNK_SIZE)
    x_test = da.concatenate([x_test, x], axis=1)

    logger.info('7')

    with open('../glove/glove_test_clean2.pkl', 'rb') as f:
        x = pickle.load(f).astype(np.float32)
        x = da.from_array(x, chunks=CHUNK_SIZE)
    x_test = da.concatenate([x_test, x], axis=1)

    with open('../glove/glove_test_clean_max2.pkl', 'rb') as f:
        x = pickle.load(f).astype(np.float32)
        x = da.from_array(x, chunks=CHUNK_SIZE)
    x_test = da.concatenate([x_test, x], axis=1)
    logger.info('8')
    return x_test  # [:, FEATURE + [-1]]


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
    df_train = pd.read_csv('../data/train_increase.csv', usecols=['is_duplicate'])
    df_test = pd.read_csv('../data/test.csv', usecols=['test_id'])

    ################
    #x_train_rev = train_data_rev()
    x_train = train_data()
    logger.info('x_shape: {}'.format(x_train.shape))
    #####################
    """
    y_train = df_train['is_duplicate'].values

    sample_weight = calc_weight(y_train)

    del df_train
    gc.collect()



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
                  'learning_rate': [0.01],  # [0.06, 0.1, 0.2],
                  'n_estimators': [10000],
                  'min_child_weight': [3],
                  'colsample_bytree': [0.7],
                  'boosting_type': ['gbdt'],
                  'num_leaves': [1300],
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

    for params in ParameterGrid(all_params):

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
                    verbose=True,
                    # eval_metric='logloss',
                    early_stopping_rounds=100
                    )
            pred = clf.predict_proba(val_x)[:, 1]
            with open('tfidf_val_%s.pkl' % cnt, 'wb') as f:
                pickle.dump((pred, val_y, val_w), f, -1)
                cnt += 1
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
            break
        # with open('tfidf_all_pred2.pkl', 'wb') as f:
        #    pickle.dump(all_pred, f, -1)

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

    #x_train = np.r_[x_train, x_train_rev]
    #y_train = np.r_[y_train, y_train]
    #sample_weight = np.r_[sample_weight, sample_weight]

    clf = LGBMClassifier(**min_params)
    clf.fit(x_train, y_train, sample_weight=sample_weight)
    with open('model.pkl', 'wb') as f:
        pickle.dump(clf, f, -1)
    del x_train
    gc.collect()
    """
    with open('model.pkl', 'rb') as f:
        clf = pickle.load(f)
    imp = pd.DataFrame(clf.feature_importances_, columns=['imp'])
    imp_use = imp[imp['imp'] > 0].sort_values('imp', ascending=False)
    logger.info('imp use {}'.format(imp_use.shape))
    with open('features_train.py', 'w') as f:
        f.write('FEATURE = [' + ','.join(map(str, imp_use.index.values)) + ']\n')

    x_test = test_data()
    logger.info('train end')
    preds = []
    logger.info('x_test_shape: {}'.format(x_test.shape))
    for i in range(int(df_test.shape[0] / CHUNK_SIZE) + 1):
        d = x_test[i * CHUNK_SIZE: (i + 1) * CHUNK_SIZE].compute()

        d[np.isnan(d)] = -100
        d[np.isinf(d)] = -100

        p_test = clf.predict_proba(d)
        preds.append(p_test)
    p_test = np.concatenate(preds)[:, 1]

    sub = pd.DataFrame()
    sub['test_id'] = df_test['test_id']
    sub['is_duplicate'] = p_test
    sub.to_csv('submit.csv', index=False)
    logger.info('learn start')
