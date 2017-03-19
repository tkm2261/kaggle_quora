import pandas
import numpy
import pickle
from multiprocessing import Pool
from gensim import models
from doc2vec import split_into_words
from make_data import cos_sim
from logging import getLogger
logger = getLogger(__name__)

model = models.Doc2Vec.load('doc2vec.model')


def _split_into_words(args):
    i, text = args
    if i % 10000 == 0:
        logger.info('row: %s' % i)
    return model.infer_vector(split_into_words(text))

if __name__ == '__main__':
    from logging import StreamHandler, DEBUG, Formatter, FileHandler

    log_fmt = Formatter('%(asctime)s %(name)s %(lineno)d [%(levelname)s][%(funcName)s] %(message)s ')
    handler = FileHandler('make_data.log', 'w')
    handler.setLevel(DEBUG)
    handler.setFormatter(log_fmt)
    logger.setLevel(DEBUG)
    logger.addHandler(handler)

    handler = StreamHandler()
    handler.setLevel('INFO')
    handler.setFormatter(log_fmt)
    logger.setLevel('INFO')
    logger.addHandler(handler)

    df = pandas.read_csv('../data/test.csv').fillna('')
    p = Pool()
    text_vet1 = p.map(_split_into_words, enumerate(df['question1'].values))
    text_vet2 = p.map(_split_into_words, enumerate(df['question2'].values))
    p.close()
    p.join()

    mat1 = numpy.array(text_vet1)  # model.infer_vector(text_vet1)
    mat2 = numpy.array(text_vet2)  # model.infer_vector(text_vet2)
    logger.info('mat {} {}'.format(mat1.shape, mat2.shape))
    logger.info('cos start')
    mat3 = cos_sim(mat1, mat2)
    logger.info('cos end')
    data = numpy.c_[mat1, mat2, mat3]
    with open('test_data.pkl', 'wb') as f:
        pickle.dump(data, f, -1)
