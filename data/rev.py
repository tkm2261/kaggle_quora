import pandas
import numpy
import re


def load_data():

    df = pandas.read_csv('train_clean2.csv').fillna('')
    tmp = df['question2'] .values
    df['question2'] = df['question1'].values
    df['question1'] = tmp

    df.to_csv('train_clean2_rev.csv', index=False)
    """
    out.write('"{}","{}","{}","{}","{}","{}"\n'.format(*df.columns.values))
    for line in df.values:
        id, qid1, qid2, _question1, _question2, is_duplicate = line
        question1 = _question2
        question2 = _question1
        out.write('"{}","{}","{}","{}","{}","{}"\n'.format(id, qid1, qid2, question1, question2, is_duplicate))
    out.close()
    """
    print('train')

    df = pandas.read_csv('test_clean2.csv').fillna('')
    tmp = df['question2'] .values
    df['question2'] = df['question1'].values
    df['question1'] = tmp

    df.to_csv('test_clean2_rev.csv', index=False)
    """
    df = pandas.read_csv('test.csv').fillna('')
    out = open('test_rev.csv', 'w')
    out.write('"{}","{}","{}"\n'.format(*df.columns.values))
    for line in df.values:
        id, _question1, _question2 = line
        question1 = _question2
        question2 = _question1
        out.write('"{}","{}","{}"\n'.format(id, question1, question2))
    out.close()
    """

load_data()
