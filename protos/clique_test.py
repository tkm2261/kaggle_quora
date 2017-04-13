import networkx as nx
from itertools import combinations
import copy
import pandas
import numpy


def preload(ppp=0.165):
    df = pandas.read_csv('../data/test.csv')
    submit = pandas.read_csv('submit.csv')
    df['pred'] = submit['is_duplicate'].values

    df2 = pandas.read_csv('../data/train.csv')
    pos_rate = df2['is_duplicate'].sum() / df2.shape[0]
    df2['pred'] = df2['is_duplicate'] / pos_rate * ppp
    map_dup = dict(((x[0], x[1]), x[2]) for x in df2[['question1', 'question2', 'is_duplicate']].values)

    G = nx.Graph()

    edges = [tuple(x) for x in df[['question1', 'question2', 'pred']].values]
    G.add_weighted_edges_from(edges)
    map_score = dict(((x[0], x[1]), x[2]) for x in df[['question1', 'question2', 'pred']].values)

    edges = [tuple(x) for x in df2[['question1', 'question2', 'pred']].values]
    G.add_weighted_edges_from(edges)
    map_score2 = dict(((x[0], x[1]), x[2]) for x in df2[['question1', 'question2', 'pred']].values)

    map_score.update(map_score2)
    return map_score, map_score2, G, edges, map_dup


if __name__ == '__main__':
    for ppp in [0.1, 0.12, 0.13, 0.14, 0.15, 0.16]:
        map_score, map_score2, G, edges, map_dup = preload(ppp)
        cnt = 0
        map_aaa = []
        map_result = copy.deepcopy(map_score)
        for cli in nx.find_cliques(G):
            if len(cli) < 3:
                continue
            keys = {}
            for (q1, q2) in combinations(cli, 2):
                if (q1, q2) in map_score:
                    keys[q1, q2] = map_score[q1, q2]
                elif (q2, q1) in map_score:
                    keys[q2, q1] = map_score[q2, q1]
            map_aaa.append([cli, keys])
        for cli, keys in map_aaa:

            vals = list(keys.values())
            val_max = numpy.max(vals)
            val_min = numpy.min(vals)
            val_avg = numpy.mean(vals)
            if val_avg > 0.4:
                val = val_max
            elif val_avg < 0.01:
                val = val_min
            else:
                val = val_max
            keys = {k: val for k in keys}
            map_result.update(keys)

        list_res = []
        for key, new in map_score2.items():
            new = map_result[key]
            try:
                label = map_dup[key]
            except:
                continue
            pred = map_score2[key]
            new_pred = new
            list_res.append((label, pred, new_pred))

        aaa = pandas.DataFrame(list_res, columns=['label', 'pred', 'new'])
        from tfidf_k import calc_weight
        from sklearn.metrics import log_loss, roc_auc_score
        sw = calc_weight(aaa['label'].values)
        print(ppp)
        print(roc_auc_score(aaa['label'].values, aaa['pred'].values, sample_weight=sw))
        print(log_loss(aaa['label'].values, aaa['pred'].values, sample_weight=sw))
        print(roc_auc_score(aaa['label'].values, aaa['new'].values, sample_weight=sw))
        print(log_loss(aaa['label'].values, aaa['new'].values, sample_weight=sw))
        print('-------')
