import pandas

df = pandas.read_csv('../data/test.csv')
submit = pandas.read_csv('submit.csv')
df['pred'] = submit['is_duplicate'].values

df2 = pandas.read_csv('../data/train.csv')
pos_rate = df2['is_duplicate'].sum() / df2.shape[0]
df2['pred'] = df2['is_duplicate'] / pos_rate * 0.165


import networkx as nx
from itertools import combinations
G = nx.Graph()

#edges = [tuple(x) for x in df[['question1', 'question2', 'pred']].values]
# G.add_weighted_edges_from(edges)
map_score = dict(((x[0], x[1]), x[2]) for x in df[['question1', 'question2', 'pred']].values)

#edges = [tuple(x) for x in df2[['question1', 'question2', 'pred']].values]
# G.add_weighted_edges_from(edges)
map_score2 = dict(((x[0], x[1]), x[2]) for x in df2[['question1', 'question2', 'pred']].values)
# map_score.update(map_score2)
edges = [(k[0], k[1], v) for k, v in map_score.items()]
G.add_weighted_edges_from(edges)

import copy
cnt = 0


for c in nx.connected_components(G):
    H = G.subgraph(c)
    cliques = sorted(list(nx.find_cliques(H)), key=len)

    print(len(cliques[0]))
