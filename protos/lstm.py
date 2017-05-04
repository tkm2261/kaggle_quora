
import os
import pandas
import numpy
import pickle
from multiprocessing import Pool
from gensim.models import KeyedVectors
from nltk.corpus import stopwords
from spacy.en import English
from logging import getLogger

from sklearn.model_selection import StratifiedKFold

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Dense, Input, LSTM, Embedding, Dropout, Activation
from keras.layers.merge import concatenate
from keras.models import Model
from keras.layers.normalization import BatchNormalization
from keras.callbacks import EarlyStopping, ModelCheckpoint


logger = getLogger(__name__)
parser = English()
BASE_DIR = '../data/'
EMBEDDING_FILE = BASE_DIR + 'GoogleNews-vectors-negative300.bin'


MAX_NB_WORDS = 200000
MAX_SEQUENCE_LENGTH = 30
EMBEDDING_DIM = 300

stops = set(stopwords.words("english")) | set(['?', ',', '.', ';', ':', '"', "'"])
print('start')
word2vec = KeyedVectors.load_word2vec_format(EMBEDDING_FILE,
                                             binary=True)
print('end')


def split_into_words(text):
    return [t.lemma_
            for t in parser(text)
            if t not in stops]


def doc_to_sentence(doc):
    words = split_into_words(doc)
    return ' '.join(words)


def corpus_to_sentences(df):
    p = Pool()
    ret = p.map(_load, enumerate(df.values))
    p.close()
    p.join()
    return ret


def _load(args):
    i, row = args
    if i % 100000 == 0:
        logger.info('sent %s' % (i))
    return doc_to_sentence(row)


def calc_weight(y_train, pos_rate=0.165):

    pos_num = y_train.sum()
    neg_num = y_train.shape[0] - y_train.sum()

    logger.info('pos_rate: %s, target pos_rate: %s, pos_num: %s' % (pos_num / y_train.shape[0], pos_rate, pos_num))

    w = (neg_num * pos_rate) / (pos_num * (1 - pos_rate))
    sample_weight = numpy.where(y_train == 1, w, 1)
    calc_pos_rate = (w * pos_num) / (w * pos_num + neg_num)
    logger.info('calc pos_rate: %s' % calc_pos_rate)
    return sample_weight


def make_embedding_matrix(word_index):
    nb_words = len(word_index) + 1
    embedding_matrix = numpy.zeros((nb_words, EMBEDDING_DIM))
    for word, i in word_index.items():
        if word in word2vec.vocab:
            embedding_matrix[i] = word2vec.word_vec(word)

    """
    p = Pool()
    embedding_matrix = numpy.r_[p.map(_get_embedding, sorted(word_index.items(), key=lambda x: x[1]))]
    p.close()
    p.join()

    if max(word_index.values()) != embedding_matrix.shape[0]:
        raise Exception('embedding_matrix size is invalid. {} {}'.format(max(word_index.values()),
                                                                         embedding_matrix.shape[0]))
    """
    return embedding_matrix


def _get_embedding(args):
    word, i = args
    if word in word2vec.vocab:
        return (i, word2vec.word_vec(word))
    else:
        return numpy.zeros(EMBEDDING_DIM)


def load_data():

    df = pandas.read_csv('../data/train_clean2.csv')
    texts_1 = corpus_to_sentences(df['question1'].astype(str))
    texts_2 = corpus_to_sentences(df['question2'].astype(str))
    labels = df['is_duplicate'].values
    logger.info('Found %s texts in train.csv' % len(texts_1))

    df = pandas.read_csv('../data/test_clean2.csv')

    test_texts_1 = corpus_to_sentences(df['question1'].astype(str))
    test_texts_2 = corpus_to_sentences(df['question2'].astype(str))
    test_ids = df['test_id'].values
    logger.info('Found %s texts in test.csv' % len(test_texts_1))

    tokenizer = Tokenizer(num_words=MAX_NB_WORDS)
    tokenizer.fit_on_texts(texts_1 + texts_2 + test_texts_1 + test_texts_2)

    sequences_1 = tokenizer.texts_to_sequences(texts_1)
    sequences_2 = tokenizer.texts_to_sequences(texts_2)
    test_sequences_1 = tokenizer.texts_to_sequences(test_texts_1)
    test_sequences_2 = tokenizer.texts_to_sequences(test_texts_2)

    word_index = tokenizer.word_index
    logger.info('Found %s unique tokens' % len(word_index))

    data_1 = pad_sequences(sequences_1, maxlen=MAX_SEQUENCE_LENGTH)
    data_2 = pad_sequences(sequences_2, maxlen=MAX_SEQUENCE_LENGTH)

    logger.info('Shape of data tensor: {}'.format(data_1.shape))
    logger.info('Shape of label tensor: {}'.format(labels.shape))

    test_data_1 = pad_sequences(test_sequences_1, maxlen=MAX_SEQUENCE_LENGTH)
    test_data_2 = pad_sequences(test_sequences_2, maxlen=MAX_SEQUENCE_LENGTH)
    logger.info('Shape of test data tensor:'.format(test_data_1.shape))

    return (labels, data_1, data_2, test_ids, test_data_1, test_data_2, word_index)


def make_lstm_model(word_index):
    num_lstm = numpy.random.randint(175, 275)
    num_dense = numpy.random.randint(100, 150)
    rate_drop_lstm = 0.15 + numpy.random.rand() * 0.25
    rate_drop_dense = 0.15 + numpy.random.rand() * 0.25

    nb_words = len(word_index) + 1
    embedding_matrix = numpy.zeros((nb_words, EMBEDDING_DIM))  # make_embedding_matrix(word_index)
    embedding_layer = Embedding(nb_words,
                                EMBEDDING_DIM,
                                weights=[embedding_matrix],
                                input_length=MAX_SEQUENCE_LENGTH,
                                trainable=True)
    lstm_layer = LSTM(num_lstm, dropout=rate_drop_lstm, recurrent_dropout=rate_drop_lstm)
    sequence_1_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
    embedded_sequences_1 = embedding_layer(sequence_1_input)
    x1 = lstm_layer(embedded_sequences_1)

    sequence_2_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
    embedded_sequences_2 = embedding_layer(sequence_2_input)
    y1 = lstm_layer(embedded_sequences_2)

    merged = concatenate([x1, y1])
    merged = Dropout(rate_drop_dense)(merged)
    merged = BatchNormalization()(merged)

    merged = Dense(num_dense, activation='relu')(merged)
    merged = Dropout(rate_drop_dense)(merged)
    merged = BatchNormalization()(merged)

    preds = Dense(1, activation='sigmoid')(merged)

    model = Model(inputs=[sequence_1_input, sequence_2_input],
                  outputs=preds)
    model.compile(loss='binary_crossentropy',
                  optimizer='nadam',
                  metrics=['acc'])
    info = 'lstm_%d_%d_%.2f_%.2f' % (num_lstm, num_dense, rate_drop_lstm, rate_drop_dense)
    logger.info(info)
    return info, model


def train():
    """
    data = load_data()
    with open('lstm_data.pkl', 'wb') as f:
        pickle.dump(data, f, -1)
    """
    with open('lstm_data.pkl', 'rb') as f:
        data = pickle.load(f)

    labels, data_1, data_2, test_ids, test_data_1, test_data_2, word_index = data
    sample_weight = calc_weight(labels)

    model_info, model = make_lstm_model(word_index)

    early_stopping = EarlyStopping(monitor='val_loss', patience=3)
    bst_model_path = model_info + '.h5'

    model_checkpoint = ModelCheckpoint(bst_model_path, save_best_only=True, save_weights_only=True)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=871)
    for train_idx, test_idx in cv.split(data_1, labels):

        data_1_train = numpy.vstack((data_1[train_idx], data_2[train_idx]))
        data_2_train = numpy.vstack((data_2[train_idx], data_1[train_idx]))
        labels_train = numpy.concatenate((labels[train_idx], labels[train_idx]))
        weight_train = numpy.concatenate((sample_weight[train_idx], sample_weight[train_idx]))

        data_1_val = numpy.vstack((data_1[test_idx], data_2[test_idx]))
        data_2_val = numpy.vstack((data_2[test_idx], data_1[test_idx]))
        labels_val = numpy.concatenate((labels[test_idx], labels[test_idx]))
        weight_val = numpy.concatenate((sample_weight[test_idx], sample_weight[test_idx]))

        break

    hist = model.fit([data_1_train, data_2_train],
                     labels_train,
                     validation_data=([data_1_val, data_2_val], labels_val, weight_val),
                     epochs=200,
                     batch_size=2048,
                     shuffle=True,
                     sample_weight=weight_train,
                     callbacks=[early_stopping, model_checkpoint])

    model.load_weights(bst_model_path)
    bst_val_score = min(hist.history['val_loss'])

    logger.info('best val score: %s' % bst_val_score)

    preds = model.predict([test_data_1, test_data_2], batch_size=8192, verbose=1)
    preds += model.predict([test_data_2, test_data_1], batch_size=8192, verbose=1)
    preds /= 2

    submission = pandas.DataFrame({'test_id': test_ids, 'is_duplicate': preds.ravel()})
    submission.to_csv('%.4f_' % (bst_val_score) + model_info + '.csv', index=False)


if __name__ == '__main__':
    from logging import StreamHandler, DEBUG, Formatter, FileHandler

    log_fmt = Formatter('%(asctime)s %(name)s %(lineno)d [%(levelname)s][%(funcName)s] %(message)s ')
    handler = FileHandler('{}.log'.format(os.path.basename(__file__)), 'w')
    handler.setLevel(DEBUG)
    handler.setFormatter(log_fmt)
    logger.setLevel(DEBUG)
    logger.addHandler(handler)

    handler = StreamHandler()
    handler.setLevel('INFO')
    handler.setFormatter(log_fmt)
    logger.setLevel('INFO')
    logger.addHandler(handler)

    train()
