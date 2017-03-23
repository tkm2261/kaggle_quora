import pandas
import numpy


def load_data():

    df = pandas.read_csv('train.csv')

    df1 = df[['qid1', 'question1']]
    df1.columns = ['qid', 'question']
    df2 = df[['qid2', 'question2']]
    df2.columns = ['qid', 'question']

    df_que = pandas.concat([df1, df2], ignore_index=True)
    df_que = df_que.drop_duplicates().fillna('').sort_values('qid')
    df_que['qid'] = ['TRAIN_%s' % i for i in df_que['qid']]

    df = pandas.read_csv('test.csv')
    df1 = df[['question1']]
    df1.columns = ['question']
    df2 = df[['question2']]
    df2.columns = ['question']
    df_que2 = pandas.concat([df1, df2], ignore_index=True)
    df_que2 = df_que2.drop_duplicates().fillna('')
    df_que2['qid'] = ['TEST_%s' % i for i in numpy.arange(df_que2.shape[0])]

    df_que = pandas.concat([df_que, df_que2], ignore_index=True)

    df_que[['question']].to_csv('questions.txt', index=False, header=False)

load_data()
