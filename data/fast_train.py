import pandas
import numpy
import re


def load_data():

    df = pandas.read_csv('train_clean2.csv').fillna('')
    out = open('train_fast.csv', 'w')
    out2 = open('train_fast_rev.csv', 'w')
    for line in df.values:
        id, qid1, qid2, question1, question2, is_duplicate = line
        out.write('__label__{} , {}? {}?\n'.format(is_duplicate, question1, question2))
        out2.write('__label__{} , {}? {}?\n'.format(is_duplicate, question2, question1))
    out.close()
    out2.close()
    print('train')
load_data()
