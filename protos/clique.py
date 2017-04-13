import pickle
import pandas
import numpy
import sys
MAX_THRESH = float(sys.argv[1])
MIN_THRESH = float(sys.argv[2])

df = pandas.read_csv('../data/train.csv')

with open('tfidf_all_pred3.pkl', 'rb') as f:
    x = pickle.load(f).astype(numpy.float32)
df['pred'] = x

import networkx as nx
G = nx.Graph()

edges = [tuple(x) for x in df[['question1', 'question2', 'pred']].values]
G.add_weighted_edges_from(edges)
# G.add_weighted_edges_from(add_edges)
map_score = dict(((x[0], x[1]), x[2]) for x in df[['question1', 'question2', 'pred']].values)
map_dup = dict(((x[0], x[1]), x[2]) for x in df[['question1', 'question2', 'is_duplicate']].values)

import copy
from itertools import combinations
idx = 0
map_result = copy.deepcopy(map_score)

for cli in nx.find_cliques(G):
    if len(cli) < 3:
        continue
    keys = {}
    for q1, q2 in combinations(cli, 2):
        if (q1, q2) in map_score:
            keys[q1, q2] = map_score[q1, q2]
        elif (q2, q1) in map_score:
            keys[q2, q1] = map_score[q2, q1]

    val_max = numpy.max(list(keys.values()))
    val_min = numpy.min(list(keys.values()))
    val_avg = numpy.mean(list(keys.values()))
    if val_avg > MAX_THRESH:  # avg_pos:
        val = val_max
    elif val_avg > MIN_THRESH:
        val = val_avg
    else:
        val = val_min
    keys = {k: val for k in keys}
    map_result.update(keys)

    idx += 1


list_res = []
for key, new in map_result.items():
    #new = map_result[key]
    try:
        label = map_dup[key]
    except:
        continue
    pred = map_score[key]
    new_pred = new
    list_res.append((label, pred, new_pred))
aaa = pandas.DataFrame(list_res, columns=['label', 'pred', 'new'])

from tfidf_k import calc_weight
from sklearn.metrics import log_loss, roc_auc_score
sw = calc_weight(aaa['label'].values)
print(MAX_THRESH, MIN_THRESH)
print(roc_auc_score(aaa['label'].values, aaa['pred'].values, sample_weight=sw))
print(log_loss(aaa['label'].values, aaa['pred'].values, sample_weight=sw))
print(roc_auc_score(aaa['label'].values, aaa['new'].values, sample_weight=sw))
print(log_loss(aaa['label'].values, aaa['new'].values, sample_weight=sw))
print('-----')
