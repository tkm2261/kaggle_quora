import numpy as np
import re
import itertools
from collections import Counter
import numpy as np
import time
import gc
from tensorflow.contrib import learn
from gensim.models.word2vec import Word2Vec
import gzip
from random import random
from preprocess import MyVocabularyProcessor
import sys

from sklearn.model_selection import GridSearchCV, ParameterGrid, StratifiedKFold, cross_val_predict
from logging import getLogger

logger = getLogger(__name__)


class InputHelper(object):

    def getTsvData(self, data):
        x1 = []
        x2 = []
        y = []
        # positive samples from file
        for line in data:
            l = line
            if random() > 0.5:
                x1.append(str(l[0]).lower())
                x2.append(str(l[1]).lower())
            else:
                x1.append(str(l[1]).lower())
                x2.append(str(l[0]).lower())
            y.append(l[2])  # np.array([0,1]))
        return np.asarray(x1), np.asarray(x2), np.asarray(y)

    def getTsvTestData(self, data):

        x1 = []
        x2 = []
        y = []
        # positive samples from file
        for line in data:
            l = line.strip().split("\t")
            if len(l) < 3:
                continue
            x1.append(str(l[0]).lower())
            x2.append(str(l[1]).lower())
            y.append(int(l[2]))  # np.array([0,1]))
        return np.asarray(x1), np.asarray(x2), np.asarray(y)

    def batch_iter(self, data, batch_size, num_epochs, shuffle=True):
        """
        Generates a batch iterator for a dataset.
        """
        data = np.asarray(data)
        logger.info(data)
        logger.info(data.shape)
        data_size = len(data)
        num_batches_per_epoch = int(len(data) / batch_size) + 1
        for epoch in range(num_epochs):
            # Shuffle the data at each epoch
            if shuffle:
                shuffle_indices = np.random.permutation(np.arange(data_size))
                shuffled_data = data[shuffle_indices]
            else:
                shuffled_data = data
            for batch_num in range(num_batches_per_epoch):
                start_index = batch_num * batch_size
                end_index = min((batch_num + 1) * batch_size, data_size)
                yield shuffled_data[start_index:end_index]

    def getDataSets(self, data, max_document_length, percent_dev, batch_size):

        x1_text, x2_text, y = self.getTsvData(data)

        # Build vocabulary
        logger.info("Building vocabulary")
        vocab_processor = MyVocabularyProcessor(max_document_length, min_frequency=0)
        vocab_processor.fit_transform(np.concatenate((x2_text, x1_text), axis=0))
        logger.info("Length of loaded vocabulary ={}".format(len(vocab_processor.vocabulary_)))

        train_set = []
        dev_set = []
        sum_no_of_batches = 0
        x1 = np.asarray(list(vocab_processor.transform(x1_text)))
        x2 = np.asarray(list(vocab_processor.transform(x2_text)))

        # Randomly shuffle data

        pos_rate = 0.165
        pos_num = y.sum()
        neg_num = y.shape[0] - y.sum()
        logger.info('pos_rate: %s, target pos_rate: %s, pos_num: %s' % (pos_num / y.shape[0], pos_rate, pos_num))

        w = (neg_num * pos_rate) / (pos_num * (1 - pos_rate))
        sample_weight = np.where(y == 1, w, 1)
        calc_pos_rate = (w * pos_num) / (w * pos_num + neg_num)
        logger.info('calc pos_rate: %s' % calc_pos_rate)

        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=871)
        for train_idx, test_idx in cv.split(x1, y):
            break
        x1_train, x1_dev = x1[train_idx], x1[test_idx]
        x2_train, x2_dev = x2[train_idx], x2[test_idx]
        y_train, y_dev = y[train_idx], y[test_idx]
        sample_weight = (sample_weight[train_idx], sample_weight[test_idx])

        sum_no_of_batches = sum_no_of_batches + (len(y_train) // batch_size)
        train_set = (x1_train, x2_train, y_train)
        dev_set = (x1_dev, x2_dev, y_dev)
        gc.collect()
        return train_set, dev_set, vocab_processor, sum_no_of_batches, sample_weight

    def getTestDataSet(self, data, vocab_path, max_document_length):

        x1_temp, x2_temp, y = self.getTsvTestData(data)

        # Build vocabulary
        vocab_processor = MyVocabularyProcessor(max_document_length, min_frequency=0)
        vocab_processor = vocab_processor.restore(vocab_path)
        logger.info(len(vocab_processor.vocabulary_))

        x1 = np.asarray(list(vocab_processor.transform(x1_temp)))
        x2 = np.asarray(list(vocab_processor.transform(x2_temp)))
        # Randomly shuffle data
        del vocab_processor
        gc.collect()
        return x1, x2, y
