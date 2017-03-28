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
from multiprocessing import Pool
from features_tmp import FEATURE

CHUNK_SIZE = 100000
from nltk.corpus import stopwords
stops = set(stopwords.words("english"))

df_train = pd.read_csv('../data/train.csv')
df_test = pd.read_csv('../data/test.csv')
train_qs = pd.Series(df_train['question1'].tolist() + df_train['question2'].tolist()).astype(str)
D = df_train.shape[0]
def get_weight(count, eps=10, min_count=2):
    if count < min_count:
        return 0
    else:
        return np.log( D / (count + eps))

from nltk.stem import WordNetLemmatizer

from spacy.en import English
parser = English()
wnl = WordNetLemmatizer()

def split_word(row):
    return str(row).lower().split()

logger.info('count start')
words = split_word(" ".join(train_qs))
counts = Counter(words)

weights = {word: get_weight(count) for word, count in counts.items()}

def tfidf_word_match_share(row):
        q1words = {}
        q2words = {}
        for word in split_word(row[0]): 
            if word not in stops:
                q1words[word] = 1
        for word in split_word(row[1]):
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

def make_kernel(df_train, df_test):

    def word_match_share(row):
        q1words = {}
        q2words = {}
        for word in split_word(row['question1']):
            if word not in stops:
                q1words[word] = 1
        for word in split_word(row['question2']):
            if word not in stops:
                q2words[word] = 1
        if len(q1words) == 0 or len(q2words) == 0:
            # The computer-generated chaff includes a few questions that are nothing but stopwords
            return 0
        shared_words_in_q1 = [w for w in q1words.keys() if w in q2words]
        shared_words_in_q2 = [w for w in q2words.keys() if w in q1words]
        R = (len(shared_words_in_q1) + len(shared_words_in_q2)) / (len(q1words) + len(q2words))
        return R


    logger.info('count_share start')
    train_word_match = df_train.apply(word_match_share, axis=1, raw=True)
    logger.info('tfidf_share start')
    p = Pool()	
    tfidf_train_word_match = p.map(tfidf_word_match_share, df_train[['question1', 'question2']].values)
    p.close()
    p.join()

    x_train = pd.DataFrame()
    x_test = pd.DataFrame()
    x_train['word_match'] = train_word_match
    x_train['tfidf_word_match'] = tfidf_train_word_match
    x_test['word_match'] = df_test.apply(word_match_share, axis=1, raw=True)
    x_test['tfidf_word_match'] = df_test.apply(tfidf_word_match_share, axis=1, raw=True)

    x_train.to_csv('kernel_train.csv', index=False)
    x_test.to_csv('kernel_test.csv', index=False)
    


if __name__ == '__main__':
    from logging import StreamHandler, DEBUG, Formatter, FileHandler

    log_fmt = Formatter('%(asctime)s %(name)s %(lineno)d [%(levelname)s][%(funcName)s] %(message)s ')
    handler = FileHandler('make_kernel.py.log', 'w')
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

    make_kernel(df_train, df_test)
