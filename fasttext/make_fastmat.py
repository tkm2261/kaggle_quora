import numpy
import pandas
import pickle
from multiprocessing import Pool
SIZE = 100
from logging import getLogger
logger = getLogger(__name__)


def calc(sentences):

    p = Pool()
    ret = p.map(_calc, sentences)
    p.close()
    p.join()

    return ret


def load_wordvec():

    f = open('fasttext_vec.csv', 'r')
    map_result = {}
    for line in f:
        line = line.strip().split(' ')
        word = line[0]
        vec = numpy.array(list(map(float, line[1:])))
        map_result[word] = vec
    return map_result

map_wordvec = load_wordvec()


def _calc(row):
    vec = []
    for i, word in enumerate(row):
        try:
            vec.append(map_wordvec[word])
        except KeyError:
            continue

    if len(vec) > 0:
        vec = numpy.array(vec)
        a = numpy.min(vec, axis=0)
        b = numpy.max(vec, axis=0)
        vec = numpy.where(numpy.fabs(b) > numpy.fabs(a), b, a)
        return vec  #
        """
        return numpy.mean(vec, axis=0)
        """
    else:
        return numpy.zeros(SIZE)


def make_data():
    with open('../protos/corpus.pkl', 'rb') as f:
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

    df_vec = pandas.DataFrame(mat[:df_que.shape[0]])
    df_vec['qid'] = df_que['qid'].values
    df_vec['question'] = df_que['question'].values

    #df_vec.to_csv('doc_vec.csv', index=False)
    with open('fast_vec.pkl', 'wb') as f:
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
    with open('fast_vec_test.pkl', 'wb') as f:
        pickle.dump(df_vec, f, -1)


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


#
if __name__ == '__main__':
    make_data()
    map_train, map_test, train_num = make_idmap()

    with open('fast_vec.pkl', 'rb') as f:
        x = pickle.load(f)[list(range(SIZE))].values

    df = pandas.read_csv('../data/train.csv')[['question1', 'question2']].fillna('').values
    df_train = pandas.DataFrame(_train(x, df, map_train))
    #df_train.to_csv('fast_train.csv', index=False)

    with open('fast_train_decay.pkl', 'wb') as f:
        pickle.dump(df_train.values, f, -1)

    with open('fast_vec_test.pkl', 'rb') as f:
        x = pickle.load(f)[list(range(SIZE))].values

    df = pandas.read_csv('../data/test.csv')[['question1', 'question2']].fillna('').values
    df_test = pandas.DataFrame(_train(x, df, map_test))
    #df_test.to_csv('fast_test.csv', index=False)

    with open('fast_test_decay.pkl', 'wb') as f:
        pickle.dump(df_test.values, f, -1)
