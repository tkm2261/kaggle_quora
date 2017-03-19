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
from multiprocessing import Pool
from sklearn.model_selection import GridSearchCV, ParameterGrid, StratifiedKFold
import xgboost as xgb
from lightgbm.sklearn import LGBMClassifier
from sklearn.metrics import log_loss, roc_auc_score

from logging import getLogger
logger = getLogger(__name__)

wnl = WordNetLemmatizer()


def split_into_words(text):
    return [wnl.lemmatize(t) for t in word_tokenize(text)]


def doc_to_sentence(doc):
    words = split_into_words(doc)
    return words


def corpus_to_sentences(df):
    p = Pool()
    ret = p.map(_load, enumerate(df.values))
    p.close()
    p.join()
    return ret


def _load(args):
    i, row = args
    if i % 10000 == 0:
        logger.info('sent %s/%s' % (i, 800000))
    return doc_to_sentence(row)


class LemmaTokenizer(object):

    def __init__(self):
        self.wnl = WordNetLemmatizer()

    def __call__(self, doc):
        return [self.wnl.lemmatize(t) for t in word_tokenize(doc)]


def get_weight(count, eps=10000, min_count=2):
    if count < min_count:
        return 0
    else:
        return 1 / (count + eps)

from nltk.corpus import stopwords
stops = set(stopwords.words("english"))


def word_match_share(row):
    q1words = {}
    q2words = {}
    for word in str(row['question1']).lower().split():
        if word not in stops:
            q1words[word] = 1
    for word in str(row['question2']).lower().split():
        if word not in stops:
            q2words[word] = 1
    if len(q1words) == 0 or len(q2words) == 0:
        # The computer-generated chaff includes a few questions that are nothing but stopwords
        return 0
    shared_words_in_q1 = [w for w in q1words.keys() if w in q2words]
    shared_words_in_q2 = [w for w in q2words.keys() if w in q1words]
    R = (len(shared_words_in_q1) + len(shared_words_in_q2)) / (len(q1words) + len(q2words))
    return R


def tfidf_word_match_share(row):
    q1words = {}
    q2words = {}
    for word in str(row['question1']).lower().split():
        if word not in stops:
            q1words[word] = 1
    for word in str(row['question2']).lower().split():
        if word not in stops:
            q2words[word] = 1
    if len(q1words) == 0 or len(q2words) == 0:
        # The computer-generated chaff includes a few questions that are nothing but stopwords
        return 0
    shared_weights = [weights.get(w, 0) for w in q1words.keys() if w in q2words] + \
        [weights.get(w, 0) for w in q2words.keys() if w in q1words]
    total_weights = [weights.get(w, 0) for w in q1words] + [weights.get(w, 0) for w in q2words]

    R = np.sum(shared_weights) / np.sum(total_weights)
    return R

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
    """
    train_qs = pd.Series(df_train['question1'].tolist() + df_train['question2'].tolist()).astype(str)
    test_qs = pd.Series(df_test['question1'].tolist() + df_test['question2'].tolist()).astype(str)

    logger.info('count start')
    words = (" ".join(train_qs)).lower().split()
    counts = Counter(words)
    logger.info('weight start')
    weights = {word: get_weight(count) for word, count in counts.items()}
    logger.info('count_share start')
    train_word_match = df_train.apply(word_match_share, axis=1, raw=True)
    logger.info('tfidf_share start')
    tfidf_train_word_match = df_train.apply(tfidf_word_match_share, axis=1, raw=True)

    x_train = pd.DataFrame()
    x_test = pd.DataFrame()
    x_train['word_match'] = train_word_match
    x_train['tfidf_word_match'] = tfidf_train_word_match
    x_test['word_match'] = df_test.apply(word_match_share, axis=1, raw=True)
    x_test['tfidf_word_match'] = df_test.apply(tfidf_word_match_share, axis=1, raw=True)

    x_train.to_csv('kernel_train.csv', index=False)
    x_test.to_csv('kernel_test.csv', index=False)
    """

    x_train = pd.read_csv('count_tfidf_train.csv').values
    x_test = pd.read_csv('count_tfidf_test.csv').values
    """
    x_train = x_train.values
    x_test = x_test.values
    """
    y_train = df_train['is_duplicate'].values

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
    all_params = {'max_depth': [20],
                  'learning_rate': [0.1],  # [0.06, 0.1, 0.2],
                  'n_estimators': [5000],
                  'min_child_weight': [1],
                  'subsample': [1],
                  'colsample_bytree': [0.7],
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
            clf = LGBMClassifier(**params)
            clf.fit(trn_x, trn_y,
                    sample_weight=trn_w,
                    eval_sample_weight=[val_w],
                    eval_set=[(val_x, val_y)],
                    verbose=True,
                    # eval_metric='logloss',
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
