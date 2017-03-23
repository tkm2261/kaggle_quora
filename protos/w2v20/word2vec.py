from nltk import word_tokenize
from nltk.stem import WordNetLemmatizer
from gensim.models.doc2vec import LabeledSentence
from gensim import models
import pandas
import numpy
import pickle
from multiprocessing import Pool

from logging import getLogger
logger = getLogger(__name__)

wnl = WordNetLemmatizer()

SIZE = 100


def train():

    logger.info('start')
    with open('corpus.pkl', 'rb') as f:
        sentences = pickle.load(f)
        sentences = [row.words for row in sentences]
    logger.info('load end')

    model = models.Word2Vec(sentences,
                            size=SIZE,
                            window=5,
                            alpha=.025,
                            min_count=2,
                            workers=15)
    """
    #model = models.Doc2Vec(sentences, size=100, window=8, min_count=3, workers=16)
    """
    logger.info('epoch')
    model.save('word2vec.model')
    #model = models.Doc2Vec.load('doc2vec.model')

    return model


def calc(sentences):

    p = Pool()
    ret = p.map(_calc, sentences)
    p.close()
    p.join()

    return ret

model = models.Doc2Vec.load('word2vec.model')


def _calc(row):
    vec = []
    for word in row:
        try:
            vec.append(model.wv[word])
        except KeyError:
            continue

    if len(vec) > 0:
        return numpy.mean(vec, axis=0)
    else:
        return numpy.zeros(SIZE)


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


def make_data():
    with open('corpus.pkl', 'rb') as f:
        sentences = pickle.load(f)
        sentences = [row.words for row in sentences]
    mat = numpy.array(calc(sentences))

    df = pandas.read_csv('../data/train.csv')

    df1 = df[['qid1', 'question1']]
    df1.columns = ['qid', 'question']
    df2 = df[['qid2', 'question2']]
    df2.columns = ['qid', 'question']

    df_que = pandas.concat([df1, df2], ignore_index=True)
    df_que = df_que.drop_duplicates().fillna('').sort_values('qid')
    df_que['qid'] = ['TRAIN_%s' % i for i in df_que['qid']]
    logger.info('df_que {}'.format(df_que.shape))

    df_vec = pandas.DataFrame(mat[:df_que.shape[0]])
    df_vec['qid'] = df_que['qid'].values
    df_vec['question'] = df_que['question'].values

    #df_vec.to_csv('doc_vec.csv', index=False)
    with open('w2v_vec.pkl', 'wb') as f:
        pickle.dump(df_vec, f, -1)

    df = pandas.read_csv('../data/test.csv')
    df1 = df[['question1']]
    df1.columns = ['question']
    df2 = df[['question2']]
    df2.columns = ['question']
    df_que2 = pandas.concat([df1, df2], ignore_index=True)
    df_que2 = df_que2.drop_duplicates().fillna('')
    df_que2['qid'] = ['TEST_%s' % i for i in numpy.arange(df_que2.shape[0])]

    df_vec = pandas.DataFrame(mat[df_que.shape[0]:])
    df_vec['qid'] = df_que2['qid'].values
    df_vec['question'] = df_que2['question'].values

    #df_vec.to_csv('doc_vec.csv', index=False)
    with open('w2v_vec_test.pkl', 'wb') as f:
        pickle.dump(df_vec, f, -1)


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
    exit()

    make_data()
    map_train, map_test, train_num = make_idmap()

    with open('w2v_vec.pkl', 'rb') as f:
        x = pickle.load(f)[list(range(SIZE))].values

    df = pandas.read_csv('../data/train.csv')[['question1', 'question2']].fillna('').values
    df_train = pandas.DataFrame(_train(x, df, map_train))
    df_train.to_csv('w2v_train.csv', index=False)

    with open('w2v_vec_test.pkl', 'rb') as f:
        x = pickle.load(f)[list(range(SIZE))].values

    df = pandas.read_csv('../data/test.csv')[['question1', 'question2']].fillna('').values
    df_test = pandas.DataFrame(_train(x, df, map_test))
    df_test.to_csv('w2v_test.csv', index=False)
