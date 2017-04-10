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

    with open('train_sparse_sym.pkl', 'rb') as f:
        idf, count, tfidf = pickle.load(f)
        x_train = feat(idf, count, tfidf)
    with open('test_sparse_sym.pkl', 'rb') as f:
        idf, count, tfidf = pickle.load(f)
        x_test = feat(idf, count, tfidf)

    with open('tfidf_all_pred2.pkl', 'rb') as f:
        pred = pickle.load(f).astype(np.float32)
    x_train = hstack([pred.reshape((-1, 1)), x_train], format='csr')

    """
    from features_tic2 import FEATURE
    feat = np.array(FEATURE[1:1001]) - 1
    with open('train_tic_val_1000.pkl', 'wb') as f:
        pickle.dump(x_train[:, feat], f, -1)
    with open('test_tic_val_1000.pkl', 'wb') as f:
        pickle.dump(x_test[:, feat], f, -1)
    exit()
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
                  'n_estimators': [500],
                  'min_child_weight': [3],
                  'colsample_bytree': [0.5],
                  'boosting_type': ['gbdt'],
                  'subsample': [0.9],
                  'min_child_samples': [10],
                  #'num_leaves': [300],
                  'reg_alpha': [1],
                  'reg_lambda': [1],
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

        clf = LGBMClassifier(**params)
        clf.fit(x_train, y_train,
                sample_weight=sample_weight,
                verbose=True,
                # eval_metric='logloss',
                )
        imp = pd.DataFrame(clf.feature_importances_, columns=['imp'])
        imp_use = imp[imp['imp'] > 0].sort_values('imp', ascending=False)
        logger.info('imp use {}'.format(imp_use.shape))
        with open('features_tic_0403.py', 'w') as f:
            f.write('FEATURE = [' + ','.join(map(str, imp_use.index.values)) + ']\n')

        with open('train_tic_300.pkl', 'wb') as f:
            pickle.dump(x_train[:, imp_use.index.values[1:301]], f, -1)
        with open('test_tic_300.pkl', 'wb') as f:
            pickle.dump(x_test[:, imp_use.index.values[1:301]], f, -1)
        break
