from nltk import word_tokenize
from nltk.stem import WordNetLemmatizer
from gensim.models.doc2vec import LabeledSentence
from gensim import models
from gensim import corpora
from gensim.matutils import corpus2csc
import pandas
import numpy
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

from multiprocessing import Pool

from logging import getLogger
logger = getLogger(__name__)

wnl = WordNetLemmatizer()
from nltk.corpus import stopwords
from nltk.corpus import wordnet as wn
stops = set(stopwords.words("english")) | set(['?', ',', '.', ';', ':', '"', "'"])

import aspell
asp = aspell.Speller('lang', 'en')


def get(w):
    try:
        return wn.synsets(w)[0].root_hypernyms()
    except IndexError:
        return []


def split_into_words(text):
    return [s.name()
            for t in word_tokenize(text.lower())
            if t not in stops
            for s in get(t)]


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


def load_data():

    df = pandas.read_csv('../data/train_clean2.csv')

    df1 = df[['qid1', 'question1']]
    df1.columns = ['qid', 'question']
    df2 = df[['qid2', 'question2']]
    df2.columns = ['qid', 'question']

    df_que = pandas.concat([df1, df2], ignore_index=True)
    df_que = df_que.drop_duplicates().fillna('').sort_values('qid')
    logger.info('df_que {}'.format(df_que.shape))
    train_num = df_que.shape[0]

    df = pandas.read_csv('../data/test_clean2.csv')
    df1 = df[['question1']]
    df1.columns = ['question']
    df2 = df[['question2']]
    df2.columns = ['question']
    df_que2 = pandas.concat([df1, df2], ignore_index=True)
    df_que2 = df_que2.drop_duplicates().fillna('')
    logger.info('df_que2 {}'.format(df_que2.shape))
    df_que2['qid'] = numpy.arange(df_que2.shape[0]) + df_que.shape[0]

    df_que = pandas.concat([df_que, df_que2], ignore_index=True)

    sentences = corpus_to_sentences(df_que['question'])
    logger.info('dict')
    dictionary = corpora.Dictionary(sentences)
    dictionary.save('./gensim.dict')
    dictionary.filter_extremes(no_below=2, no_above=1., keep_n=2000000)
    p = Pool()
    id_corpus = p.map(dictionary.doc2bow, sentences)
    p.close()
    p.join()
    with open('count_corpus_sym.pkl', 'wb') as f:
        pickle.dump(id_corpus, f, -1)

    count_mat = corpus2csc(id_corpus).T
    logger.info('count_mat {}'.format(count_mat.shape))
    with open('count_mat_sym.pkl', 'wb') as f:
        pickle.dump(count_mat, f, -1)

    tfidf_instance = models.TfidfModel(id_corpus, normalize=False)
    tfidf_corpus = tfidf_instance[id_corpus]
    tfidf_mat = corpus2csc(tfidf_corpus).T
    logger.info('tfidf_mat {}'.format(tfidf_mat.shape))
    with open('tfidf_mat_sym.pkl', 'wb') as f:
        pickle.dump(tfidf_mat, f, -1)

    logger.info('df_que {}'.format(df_que.shape))

    logger.info('end load')
    return 0


def train():
    with open('count_mat_sym.pkl', 'rb') as f:
        count_mat = pickle.load(f)
    logger.info('count_mat {}'.format(count_mat.shape))
    with open('tfidf_mat_sym.pkl', 'rb') as f:
        tfidf_mat = pickle.load(f)
    logger.info('tfidf_mat {}'.format(tfidf_mat.shape))
    map_train, map_test, train_num = make_idmap()

    df = pandas.read_csv('../data/train_clean2.csv')[['question1', 'question2']].fillna('').values
    df_train = _train(count_mat[:train_num], tfidf_mat[:train_num], df, map_train)
    df_train.to_csv('count_tfidf_train_clean2_sym.csv', index=False)

    df = pandas.read_csv('../data/test_clean2.csv')[['question1', 'question2']].fillna('').values
    df_test = _train(count_mat[train_num:], tfidf_mat[train_num:], df, map_test)
    df_test.to_csv('count_tfidf_test_clean2_sym.csv', index=False)


def _train(count_mat, tfidf_mat, df, map_train):
    logger.info('start')
    idxs1 = []
    idxs2 = []
    for q1, q2 in df:
        idxs1.append(map_train[q1])
        idxs2.append(map_train[q2])
    tmp = count_mat.copy()
    tmp.data = 1. / tmp.data
    idf_mat = tfidf_mat.multiply(tmp)

    count_vec1 = count_mat[idxs1]
    count_vec2 = count_mat[idxs2]

    count_orig_vec1 = count_vec1.copy()
    count_orig_vec2 = count_vec2.copy()

    wn1 = count_vec1.sum(axis=1)
    wn2 = count_vec2.sum(axis=1)

    count_vec1.data = numpy.ones(count_vec1.nnz, dtype=count_vec1.dtype)
    count_vec2.data = numpy.ones(count_vec2.nnz, dtype=count_vec2.dtype)

    wn1 = count_vec1.sum(axis=1)
    wn2 = count_vec2.sum(axis=1)
    tmp = count_vec1.multiply(count_vec2)
    same_num = tmp.sum(axis=1)
    same_rate = same_num / (wn1 + wn2)

    won1 = count_orig_vec1.sum(axis=1)
    won2 = count_orig_vec2.sum(axis=1)
    same_orig1 = count_orig_vec1.multiply(count_vec2).sum(axis=1)
    same_orig2 = count_orig_vec2.multiply(count_vec1).sum(axis=1)

    same_orig_rate = (same_orig1 + same_orig2) / (won1 + won2)

    tfidf_vec1 = tfidf_mat[idxs1]
    tfidf_vec2 = tfidf_mat[idxs2]

    wt1 = tfidf_vec1.sum(axis=1)
    wt2 = tfidf_vec2.sum(axis=1)

    wt1_rate = wt1 / won1
    wt2_rate = wt2 / won2

    tmp = tfidf_vec1.multiply(tfidf_vec2)
    tmp = numpy.sqrt(tmp)
    max_tfidf = tmp.max(axis=1).todense()
    sum_tfidf = tmp.sum(axis=1)
    tfidf_rate = sum_tfidf / (wt1 + wt2)

    same_w1 = tfidf_vec1.multiply(count_vec2).sum(axis=1)
    same_w2 = tfidf_vec2.multiply(count_vec1).sum(axis=1)

    tfidf_rate_dup = (same_w1 + same_w2) / (wt1 + wt2)

    idf_vec1 = idf_mat[idxs1]
    idf_vec2 = idf_mat[idxs2]

    wi1 = idf_vec1.sum(axis=1)
    wi2 = idf_vec2.sum(axis=1)

    wi1_rate = wi1 / won1
    wi2_rate = wi2 / won2

    tmp = idf_vec1.multiply(idf_vec2)
    tmp = numpy.sqrt(tmp)
    max_idf = tmp.max(axis=1).todense()
    sum_idf = tmp.sum(axis=1)
    idf_rate = sum_idf / (wi1 + wi2)

    same_wi1 = idf_vec1.multiply(count_vec2).sum(axis=1)
    same_wi2 = idf_vec2.multiply(count_vec1).sum(axis=1)

    idf_rate_dup = (same_wi1 + same_wi2) / (wi1 + wi2)

    data = numpy.concatenate([wn1, wn2, same_num, same_rate,
                              won1, won2, same_orig1, same_orig2, same_orig_rate,
                              wt1, wt2, wt1_rate, wt2_rate, max_tfidf, sum_tfidf, tfidf_rate, same_w1, same_w2, tfidf_rate_dup,
                              wi1, wi2, wi1_rate, wi2_rate, max_idf, sum_idf, idf_rate, same_wi1, same_wi2, idf_rate_dup
                              ], axis=1)
    return pandas.DataFrame(data, columns=['wn1', 'wn2', 'same_num', 'same_rate',
                                           'won1', 'won2', 'same_orig1', 'same_orig2', 'same_orig_rate',
                                           'wt1', 'wt2', 'wt1', 'wt2', 'max_tfidf',  'sum_tfidf', 'rate_tfidf', 'same_w1', 'same_w2', 'tfidf_rate_dup',
                                           'wi1', 'wi2', 'wi1', 'wi2', 'max_idf',  'sum_idf', 'rate_idf', 'same_wi1', 'same_wi2', 'idf_rate_dup',
                                           ])


def make_idmap():
    logger.info('start')

    df = pandas.read_csv('../data/train_clean2.csv')

    df1 = df[['qid1', 'question1']]
    df1.columns = ['qid', 'question']
    df2 = df[['qid2', 'question2']]
    df2.columns = ['qid', 'question']

    df_que = pandas.concat([df1, df2], ignore_index=True)
    df_que = df_que.drop_duplicates().fillna('').sort_values('qid')
    train_num = df_que.shape[0]

    map_train = dict(zip(df_que['question'], range(df_que.shape[0])))

    logger.info('df_que {}'.format(df_que.shape))
    df = pandas.read_csv('../data/test_clean2.csv')
    df1 = df[['question1']]
    df1.columns = ['question']
    df2 = df[['question2']]
    df2.columns = ['question']
    df_que2 = pandas.concat([df1, df2], ignore_index=True)
    df_que2 = df_que2.drop_duplicates().fillna('')
    logger.info('df_que2 {}'.format(df_que2.shape))
    df_que2['qid'] = numpy.arange(df_que2.shape[0])

    map_test = dict(zip(df_que2['question'], range(df_que2.shape[0])))

    return map_train, map_test, train_num

if __name__ == '__main__':
    from logging import StreamHandler, DEBUG, Formatter, FileHandler

    log_fmt = Formatter('%(asctime)s %(name)s %(lineno)d [%(levelname)s][%(funcName)s] %(message)s ')
    handler = FileHandler('doc2vec.py.log', 'w')
    handler.setLevel(DEBUG)
    handler.setFormatter(log_fmt)
    logger.setLevel(DEBUG)
    logger.addHandler(handler)

    handler = StreamHandler()
    handler.setLevel('INFO')
    handler.setFormatter(log_fmt)
    logger.setLevel('INFO')
    logger.addHandler(handler)

    load_data()
    train()
