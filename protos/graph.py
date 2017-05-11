from itertools import combinations
from tqdm import tqdm
import pickle
import pandas
import numpy
from scipy.stats import skew, kurtosis

df = pandas.read_csv('../data/train.csv')

with open('tfidf_all_pred2_0506.pkl', 'rb') as f:
    x = pickle.load(f).astype(numpy.float32)
df['pred'] = x

avg_pos = df[df['is_duplicate'] == 1]['pred'].mean()
avg_neg = df[df['is_duplicate'] == 0]['pred'].mean()

import networkx as nx
numpy.random.seed(111)
G = nx.Graph()

edges = [tuple(x) for x in df[['question1', 'question2', 'pred']].values]
G.add_weighted_edges_from(edges)
# G.add_weighted_edges_from(add_edges)
map_score = dict(((x[0], x[1]), x[2]) for x in df[['question1', 'question2', 'pred']].values)
map_dup = dict(((x[0], x[1]), x[2]) for x in df[['question1', 'question2', 'is_duplicate']].values)

cliques = sorted(list(nx.find_cliques(G)), key=lambda x: (len(x), max(map(str, x))))

import copy
from itertools import combinations
from collections import defaultdict
cnt = 0

map_result = copy.deepcopy(map_score)
map_cnum = {}
map_data = {}
map_app = defaultdict(int)

map_max = {}
map_min = {}

for cli in tqdm(cliques):
    # if len(cli) < 3:
    #    continue
    keys = {}
    for q1, q2 in combinations(cli, 2):
        if (q1, q2) in map_result:  # map_score:
            keys[q1, q2] = map_result[q1, q2]
        elif (q2, q1) in map_result:
            keys[q2, q1] = map_result[q2, q1]

    val_max = numpy.max(list(keys.values()))
    val_min = numpy.min(list(keys.values()))
    val_avg = numpy.mean(list(keys.values()))
    """
    val_med = numpy.median(list(keys.values()))
    val_std = numpy.std(list(keys.values()))
    val_skew = skew(list(keys.values()))
    val_kurt = kurtosis(list(keys.values()))
    """

    if val_avg > 0.4:  # avg_pos:
        val = val_avg  # val_max
        keys = {k: numpy.max([val, map_result[k]]) for k in keys}
    elif val_avg > 0.05:
        val = val_avg
        keys = {k: numpy.max([val, map_result[k]]) for k in keys}
    else:
        val = val_avg  # val_min
        keys = {k: numpy.min([val, map_result[k]]) for k in keys}
    #keys = {k: val for k in keys}
    map_result.update(keys)
    keys = {k: len(cli) for k in keys}
    map_cnum.update(keys)
    keys = {k: (val_max, val_min, val_avg) for k in keys}
    map_data.update(keys)
    for k in keys:
        map_app[k] += 1
        map_max[k] = max(val, map_max.get(k, -1))
        map_min[k] = min(val, map_min.get(k, 999))

list_res = []
use_cols = ['cnum', 'pred', 'new', 'vmax', 'vmin', 'vavg', 'appnum', 'emax', 'emin', 'l_score',
            'r_score', 'm_score', 'l_num', 'r_num', 'm_num']  # , 'vmed', 'vstd', 'vskew', 'vkurt']
# for key, new in map_result.items():
for q1, q2 in tqdm(df[['question1', 'question2']].values):
    key = (q1, q2)
    if q1 == q2:
        map_result[key] = 1.

    new = map_result[key]
    try:
        label = map_dup[key]
    except:
        continue
    pred = map_score[key]
    cnum = map_cnum.get(key, -1)
    data = list(map_data[key])
    app = map_app[key]
    new_pred = new
    emin = map_min[key]
    emax = map_max[key]

    l_num = len(G[key[0]])
    r_num = len(G[key[1]])
    m_num = (l_num + r_num) / 2

    l_score = numpy.mean([map_result[key[0], to] if (key[0], to)
                          in map_result else map_result[to, key[0]] for to in G[key[0]]])
    r_score = numpy.mean([map_result[to, key[1]] if (to, key[1])
                          in map_result else map_result[key[1], to] for to in G[key[1]]])
    m_score = (l_score + r_score) / 2

    list_res.append([label, cnum, pred, new_pred] + data + [app, emax,
                                                            emin, l_score, r_score, m_score, l_num, r_num, m_num])
aaa = pandas.DataFrame(list_res, columns=['label'] + use_cols)

from tfidf_k import calc_weight
from sklearn.metrics import log_loss, roc_auc_score
sw = calc_weight(aaa['label'].values)
print(roc_auc_score(aaa['label'].values, aaa['pred'].values, sample_weight=sw))
print(log_loss(aaa['label'].values, aaa['pred'].values, sample_weight=sw))
print(roc_auc_score(aaa['label'].values, aaa['new'].values, sample_weight=sw))
print(log_loss(aaa['label'].values, aaa['new'].values, sample_weight=sw))

aaa.to_csv('clique_data.csv', index=False)
