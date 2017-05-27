from itertools import combinations
from tqdm import tqdm
import pickle
import pandas
import numpy
from scipy.stats import skew, kurtosis
import networkx as nx

G = nx.Graph()

df = pandas.read_csv('../data/test.csv')
edges = [tuple(x) for x in df[['question1', 'question2']].values]
G.add_edges_from(edges)

df = pandas.read_csv('../data/train.csv')
edges = [tuple(x) for x in df[['question1', 'question2']].values]
G.add_edges_from(edges)

selfloop_edges = G.selfloop_edges()

print('self loop edges: %s' % (len(selfloop_edges)))
G.remove_edges_from(selfloop_edges)

cliques = sorted(list(nx.find_cliques(G)), key=lambda x: (len(x), max(map(str, x))))

# with open('tfidf_all_pred_final_0512.pkl', 'rb') as f:
#    x = pickle.load(f).astype(numpy.float32)

# with open('tfidf_all_pred3_0509.pkl', 'rb') as f:
#    x = pickle.load(f).astype(numpy.float32)
# with open('tfidf_all_pred2_0506.pkl', 'rb') as f:
#    x = pickle.load(f).astype(numpy.float32)
with open('tfidf_all_pred2_0526.pkl', 'rb') as f:
    x = pickle.load(f).astype(numpy.float32)
df['pred'] = x

avg_pos = df[df['is_duplicate'] == 1]['pred'].mean()
avg_neg = df[df['is_duplicate'] == 0]['pred'].mean()


numpy.random.seed(111)
G = nx.Graph()

edges = [tuple(x) for x in df[['question1', 'question2', 'pred']].values]
G.add_weighted_edges_from(edges)
selfloop_edges = G.selfloop_edges()

print('self loop edges: %s' % (len(selfloop_edges)))
G.remove_edges_from(selfloop_edges)

# G.add_weighted_edges_from(add_edges)
map_score = dict(((x[0], x[1]), x[2]) for x in df[['question1', 'question2', 'pred']].values)
map_dup = dict(((x[0], x[1]), x[2]) for x in df[['question1', 'question2', 'is_duplicate']].values)

#map_eign_cent = nx.eigenvector_centrality(G, weight=None)
# with open('map_eign_cent_train.pkl', 'wb') as f:
#    pickle.dump(map_eign_cent, f, -1)
with open('map_eign_cent_train.pkl', 'rb') as f:
    map_eign_cent = pickle.load(f)


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

map_clique_data = {}

for cli in tqdm(cliques):
    # if len(cli) < 3:
    #    continue
    keys = {}
    for q1, q2 in combinations(cli, 2):
        if (q1, q2) in map_score:  # map_score:
            keys[q1, q2] = map_score[q1, q2]
        elif (q2, q1) in map_score:
            keys[q2, q1] = map_score[q2, q1]

    if len(keys) == 0:
        continue

    val_max = numpy.max(list(keys.values()))
    val_min = numpy.min(list(keys.values()))
    val_avg = numpy.mean(list(keys.values()))

    for q1, q2 in keys:
        map_clique_data[q1] = [val_max, val_min, val_avg]
        map_clique_data[q2] = [val_max, val_min, val_avg]
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
    keys = {k: (val_max, val_min, val_avg  # , val_med, val_std, val_skew, val_kurt
                ) for k in keys}
    map_data.update(keys)
    for k in keys:
        map_app[k] += 1
        map_max[k] = max(val, map_max.get(k, -1))
        map_min[k] = min(val, map_min.get(k, 999))

list_res = []
use_cols = ['cnum', 'pred', 'new', 'vmax', 'vmin', 'vavg', 'appnum', 'emax', 'emin', 'l_score',
            'r_score', 'm_score', 'l_num', 'r_num', 'm_num', 'l_min', 'l_max', 'r_min', 'r_max',
            # 'vmed', 'vstd', 'vskew', 'vkurt',
            #'l_med', 'l_std', 'l_skew', 'l_kurt',
            #'r_med', 'r_std', 'r_skew', 'r_kurt'
            'l_cnum_max', 'r_cnum_max', 'l_cnum_min', 'r_cnum_min', 'l_cnum_avg', 'r_cnum_avg',
            'l_eign_cent', 'r_eign_cent',
            'n_med', 'med_min', 'med_max', 'med_avg',
            'med_l_min', 'med_l_max', 'med_l_avg',
            'med_r_min', 'med_r_max', 'med_r_avg',
            'l_c_max', 'l_c_min', 'l_c_avg',
            'r_c_max', 'r_c_min', 'r_c_avg'
            ]

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

    l_scores = [map_score[key[0], to] if (key[0], to)
                in map_score else map_score[to, key[0]] for to in G[key[0]]]
    r_scores = [map_score[to, key[1]] if (to, key[1])
                in map_score else map_score[key[1], to] for to in G[key[1]]]
    l_score = numpy.mean(l_scores)
    r_score = numpy.mean(r_scores)
    m_score = (l_score + r_score) / 2

    l_min = numpy.min(l_scores)
    l_max = numpy.max(l_scores)
    r_min = numpy.min(r_scores)
    r_max = numpy.max(r_scores)

    l_cnums = [map_cnum.get((key[0], to), 1) for to in G[key[0]]]
    r_cnums = [map_cnum.get((to, key[1]), 1) for to in G[key[1]]]

    l_cnum_max = numpy.max(l_cnums)
    r_cnum_max = numpy.max(r_cnums)
    l_cnum_min = numpy.min(l_cnums)
    r_cnum_min = numpy.min(r_cnums)
    l_cnum_avg = numpy.mean(l_cnums)
    r_cnum_avg = numpy.mean(r_cnums)

    l_eign_cent = map_eign_cent.get(key[0], 0)
    r_eign_cent = map_eign_cent.get(key[1], 0)

    nodes = set(G[key[0]]) & set(G[key[1]]) - set(key)
    n_med = len(nodes)
    med_weights = []
    med_l_weights = []
    med_r_weights = []
    for n in nodes:
        score1 = G[key[0]][n]['weight']
        score2 = + G[n][key[1]]['weight']
        score = (score1 + score2) / 2
        med_weights.append(score)
        med_l_weights.append(score1)
        med_r_weights.append(score1)
    if len(med_weights) == 0:
        med_weights = [-1]
    if len(med_l_weights) == 0:
        med_l_weights = [-1]
    if len(med_r_weights) == 0:
        med_r_weights = [-1]
    med_min = numpy.min(med_weights)
    med_max = numpy.max(med_weights)
    med_avg = numpy.mean(med_weights)

    med_l_min = numpy.min(med_l_weights)
    med_l_max = numpy.max(med_l_weights)
    med_l_avg = numpy.mean(med_l_weights)

    med_r_min = numpy.min(med_r_weights)
    med_r_max = numpy.max(med_r_weights)
    med_r_avg = numpy.mean(med_r_weights)

    l_c_max, l_c_min, l_c_avg = map_clique_data[q1]
    r_c_max, r_c_min, r_c_avg = map_clique_data[q2]

    """
    l_med = numpy.median(l_scores)
    l_std = numpy.std(l_scores)
    l_skew = skew(l_scores)
    l_kurt = kurtosis(l_scores)

    r_med = numpy.median(r_scores)
    r_std = numpy.std(r_scores)
    r_skew = skew(r_scores)
    r_kurt = kurtosis(r_scores)
    """
    list_res.append([label, cnum, pred, new_pred] + data + [app, emax,
                                                            emin, l_score, r_score, m_score, l_num, r_num, m_num,
                                                            l_min, l_max, r_min, r_max,
                                                            #l_med, l_std, l_skew, l_kurt,
                                                            #r_med, r_std, r_skew, r_kurt
                                                            l_cnum_max, r_cnum_max, l_cnum_min, r_cnum_min, l_cnum_avg, r_cnum_avg,
                                                            l_eign_cent, r_eign_cent,
                                                            n_med, med_min, med_max, med_avg,
                                                            med_l_min, med_l_max, med_l_avg,
                                                            med_r_min, med_r_max, med_r_avg,
                                                            l_c_max, l_c_min, l_c_avg,
                                                            r_c_max, r_c_min, r_c_avg
                                                            ])
aaa = pandas.DataFrame(list_res, columns=['label'] + use_cols)

from tfidf_k import calc_weight
from sklearn.metrics import log_loss, roc_auc_score
sw = calc_weight(aaa['label'].values)
print(roc_auc_score(aaa['label'].values, aaa['pred'].values, sample_weight=sw))
print(log_loss(aaa['label'].values, aaa['pred'].values, sample_weight=sw))
print(roc_auc_score(aaa['label'].values, aaa['new'].values, sample_weight=sw))
print(log_loss(aaa['label'].values, aaa['new'].values, sample_weight=sw))

aaa.to_csv('clique_data_0526.csv', index=False)
