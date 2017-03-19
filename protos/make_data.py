import pandas
import numpy
import pickle
from multiprocessing import Pool

from logging import getLogger
logger = getLogger(__name__)


def cos_sim(v1, v2):
    return (v1 * v2).sum(axis=1) / (numpy.sqrt((v1 ** 2).sum(axis=1)) * numpy.sqrt((v2 ** 2).sum(axis=1)))

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

    #df_mst = pandas.read_csv('doc_vec.csv').set_index('qid')
    with open('doc_vec.pkl', 'rb') as f:
        df_mst = pickle.load(f).set_index('qid')

    df = pandas.read_csv('../data/train.csv')[['is_duplicate', 'qid1', 'qid2']]
    logger.info('load end {}'.format(df.shape))
    cols = list(range(300))
    mat1 = df_mst.ix[df['qid1'].values][cols].values
    mat2 = df_mst.ix[df['qid2'].values][cols].values
    logger.info('mat {} {}'.format(mat1.shape, mat2.shape))
    logger.info('cos start')
    mat3 = cos_sim(mat1, mat2)
    logger.info('cos end')
    data = numpy.c_[mat1, mat2, mat3]
    logger.info('mat {}'.format(data.shape))
    label = df['is_duplicate'].values
    with open('train_data.pkl', 'wb') as f:
        pickle.dump((data, label), f, -1)
