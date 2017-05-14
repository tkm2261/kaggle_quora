from itertools import combinations
from tqdm import tqdm
import pickle
import pandas
import numpy
from scipy.stats import skew, kurtosis

df = pandas.read_csv('../data/train.csv')

# with open('tfidf_all_pred_final_0512.pkl', 'rb') as f:
#    x = pickle.load(f).astype(numpy.float32)

# with open('tfidf_all_pred3_0509.pkl', 'rb') as f:
#    x = pickle.load(f).astype(numpy.float32)
# with open('tfidf_all_pred2_0506.pkl', 'rb') as f:
#    x = pickle.load(f).astype(numpy.float32)
with open('tfidf_all_pred2_0512.pkl', 'rb') as f:
    x = pickle.load(f).astype(numpy.float32)
df['pred'] = x

avg_pos = df[df['is_duplicate'] == 1]['pred'].mean()
avg_neg = df[df['is_duplicate'] == 0]['pred'].mean()

import networkx as nx
numpy.random.seed(111)
G = nx.Graph()

edges = [tuple(x) for x in df[['question1', 'question2', 'pred']].values]
G.add_weighted_edges_from(edges)

map_dist = nx.all_pairs_node_connectivity(G)

list_dist = []
for q1, q2 in tqdm(df[['question1', 'question2']].values):
    list_dist.append(map_dist[q1][q2])

list_dist = numpy.array(list_dist)

with open('train_dist.pkl', 'wb') as f:
    pickle.dump(list_dist, f, -1)
