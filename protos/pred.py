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
    handler = FileHandler('pred.py.log', 'w')
    handler.setLevel(DEBUG)
    handler.setFormatter(log_fmt)
    logger.setLevel(DEBUG)
    logger.addHandler(handler)

    handler = StreamHandler()
    handler.setLevel('INFO')
    handler.setFormatter(log_fmt)
    logger.setLevel('INFO')
    logger.addHandler(handler)

    # with open('vec300/test_data.pkl', 'rb') as f:
    #    test_data = pickle.load(f)
    logger.info('vec load end')
    x2 = pandas.read_csv('count_tfidf_test.csv').values
    test_data = x2  # numpy.c_[test_data, x2]
    logger.info('tfidf load end')
    with open('model.pkl', 'rb') as f:
        clf = pickle.load(f)
    logger.info('model load end')
    logger.info('test_data {}'.format(test_data.shape))
    pred = clf.predict_proba(test_data)[:, 1]

    df_pred = pandas.read_csv('../data/sample_submission.csv')
    df_pred['is_duplicate'] = pred
    df_pred.to_csv('submit.csv', index=False)
