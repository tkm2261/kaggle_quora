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
stops = set(stopwords.words("english")) | set(['?', ',', '.', ';', ':', '"', "'"])


def split_into_words(text):
    return [wnl.lemmatize(t) for t in word_tokenize(text) if t not in stops]


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


def train():
    """
    with open('count_corpus.pkl', 'rb') as f:
        id_corpus = pickle.load(f)

    lda = models.ldamulticore.LdaMulticore(corpus=id_corpus, num_topics=20)

    lda.save('lda.model')



    result = numpy.array(list(lda[id_corpus]))
    """
    map_train, map_test, train_num = make_idmap()
    with open('lda_sp.pkl', 'rb') as f:
        result = numpy.asarray(pickle.load(f).T.todense())

    df = pandas.read_csv('../data/train.csv')[['question1', 'question2']].fillna('').values
    df_train = pandas.DataFrame(_train(result[:train_num], df, map_train))
    df_train.to_csv('lda_train.csv', index=False)

    df = pandas.read_csv('../data/test.csv')[['question1', 'question2']].fillna('').values
    df_test = pandas.DataFrame(_train(result[train_num:], df, map_test))
    df_test.to_csv('lda_test.csv', index=False)


def cos_sim(v1, v2):
    return (v1 * v2).sum(axis=1) / (numpy.sqrt((v1 ** 2).sum(axis=1)) * numpy.sqrt((v2 ** 2).sum(axis=1)))


def _train(count_mat, df, map_train):
    logger.info('start')
    idxs1 = []
    idxs2 = []
    for q1, q2 in df:
        idxs1.append(map_train[q1])
        idxs2.append(map_train[q2])

    count_vec1 = count_mat[idxs1]
    count_vec2 = count_mat[idxs2]
    mat3 = cos_sim(count_vec1, count_vec2)

    return numpy.c_[count_vec1, count_vec2, mat3]


def make_idmap():
    logger.info('start')

    df = pandas.read_csv('../data/train.csv')

    df1 = df[['qid1', 'question1']]
    df1.columns = ['qid', 'question']
    df2 = df[['qid2', 'question2']]
    df2.columns = ['qid', 'question']

    df_que = pandas.concat([df1, df2], ignore_index=True)
    df_que = df_que.drop_duplicates().fillna('').sort_values('qid')
    train_num = df_que.shape[0]

    map_train = dict(zip(df_que['question'], range(df_que.shape[0])))

    logger.info('df_que {}'.format(df_que.shape))
    df = pandas.read_csv('../data/test.csv')
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

    # load_data()
    train()
