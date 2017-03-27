from nltk import word_tokenize
from nltk.stem import WordNetLemmatizer
from gensim.models.doc2vec import LabeledSentence
from gensim import models
import pandas
import numpy
import pickle
from multiprocessing import Pool

from spacy.en import English
parser = English()
wnl = WordNetLemmatizer()
from subject_object_extraction import findSVOs
from logging import getLogger
logger = getLogger(__name__)
from collections import Counter

# If a word appears only once, we ignore it completely (likely a typo)
# Epsilon defines a smoothing constant, which makes the effect of extremely rare words smaller


def get_weight(count, D, eps=1, min_count=2):
    if count < min_count:
        return 0
    else:
        return numpy.log(D / (count + eps))


def aaa(row):
    return [word.lower_ for word in parser(str(row).lower())]


def make_idf():
    print('enter')

    df = pandas.read_csv('../data/train.csv',
                         usecols=['question1', 'question2']).values

    df2 = pandas.read_csv('../data/test.csv',
                          usecols=['question1', 'question2']).values
    """
    p = Pool()
    _ret = p.map(aaa, df[:, 0])
    _ret = p.map(aaa, df[:, 1])
    _ret = p.map(aaa, df2[:, 0])
    _ret = p.map(aaa, df2[:, 1])
    p.close()
    p.join()
    ret = []
    for r in _ret:
        ret += r

    counts = Counter(ret)
    with open('svo_counter.pkl', 'wb') as f:
        pickle.dump(counts, f, -1)
    """
    with open('svo_counter.pkl', 'rb') as f:
        counts = pickle.load(f)

    weights = {word: get_weight(count, (df.shape[0] + df2.shape[0]) * 2) for word, count in counts.items()}
    print(len(weights))
    print('exit')
    return weights

weights = make_idf()


def feat(set1, set2):
    num_ent = 0
    val_ent = 0.
    sets = set1 & set2
    aaa = len(set1 | set2)
    if aaa > 0:
        rate_ent = len(sets) / aaa
    else:
        rate_ent = 0

    for word in sets:
        num_ent += 1  #
        val_ent += weights.get(word, 0)
    return num_ent, val_ent, rate_ent

import Levenshtein
from jellyfish._jellyfish import damerau_levenshtein_distance, jaro_distance, match_rating_comparison


def load_data(row):

    lev_dist = Levenshtein.distance(str(row[0]).lower(), str(row[1]).lower())
    jar_dist = jaro_distance(str(row[0]).lower(), str(row[1]).lower())
    dam_dist = damerau_levenshtein_distance(str(row[0]).lower(), str(row[1]).lower())

    q1 = parser(str(row[0]))
    q2 = parser(str(row[1]))

    set_ent1 = set([ele.label_.lower() for ele in q1.ents])
    set_ent2 = set([ele.label_.lower() for ele in q2.ents])

    num_ent, val_ent, rate_ent = feat(set_ent1, set_ent2)

    list_last1 = [wnl.lemmatize(ele.lower_) for ele in q1 if ele.pos_ != 'PUNCT']
    list_last2 = [wnl.lemmatize(ele.lower_) for ele in q2 if ele.pos_ != 'PUNCT']
    num_for = 0
    val_for = 0.
    for i in range(min(len(list_last1), len(list_last2))):
        if list_last1[i] == list_last2[i] or match_rating_comparison(list_last1[i], list_last2[i]):
            num_for += 1
            val_for += weights.get(list_last1[i], 0)
        else:
            break

    list_last1.reverse()
    list_last2.reverse()
    num_rev = 0
    val_rev = 0.
    for i in range(min(len(list_last1), len(list_last2))):
        if list_last1[i] == list_last2[i] or match_rating_comparison(list_last1[i], list_last2[i]):
            num_rev += 1
            val_rev += weights.get(list_last1[i], 0)
        else:
            break

    set_sub1 = set([wnl.lemmatize(ele.lower_) for ele in q1 if ele.dep_ == 'nsubj'])
    set_sub2 = set([wnl.lemmatize(ele.lower_) for ele in q2 if ele.dep_ == 'nsubj'])

    num_sub, val_sub, rate_sub = feat(set_sub1, set_sub2)

    set_root1 = set([wnl.lemmatize(ele.lower_) for ele in q1 if ele.dep_ == 'ROOT'])
    set_root2 = set([wnl.lemmatize(ele.lower_) for ele in q2 if ele.dep_ == 'ROOT'])

    num_root, val_root, rate_root = feat(set_root1, set_root2)

    set_advmod1 = set([wnl.lemmatize(ele.lower_) for ele in q1 if ele.dep_ == 'advmod'])
    set_advmod2 = set([wnl.lemmatize(ele.lower_) for ele in q2 if ele.dep_ == 'advmod'])

    num_advmod, val_advmod, rate_advmod = feat(set_advmod1, set_advmod2)

    set_advcl1 = set([wnl.lemmatize(ele.lower_) for ele in q1 if ele.dep_ == 'advcl'])
    set_advcl2 = set([wnl.lemmatize(ele.lower_) for ele in q2 if ele.dep_ == 'advcl'])

    num_advcl, val_advcl, rate_advcl = feat(set_advcl1, set_advcl2)

    set_aux1 = set([wnl.lemmatize(ele.lower_) for ele in q1 if ele.dep_ == 'aux'])
    set_aux2 = set([wnl.lemmatize(ele.lower_) for ele in q2 if ele.dep_ == 'aux'])

    num_aux, val_aux, rate_aux = feat(set_aux1, set_aux2)

    set_dobj1 = set([wnl.lemmatize(ele.lower_) for ele in q1 if ele.dep_ == 'dobj'])
    set_dobj2 = set([wnl.lemmatize(ele.lower_) for ele in q2 if ele.dep_ == 'dobj'])

    num_dobj, val_dobj, rate_dobj = feat(set_dobj1, set_dobj2)

    # set_poss1 = set([wnl.lemmatize(ele.lower_) for ele in q1 if ele.dep_ == 'poss'])
    # set_poss2 = set([wnl.lemmatize(ele.lower_) for ele in q2 if ele.dep_ == 'poss'])

    # num_poss, val_poss, rate_poss = feat(set_poss1, set_poss2)

    set_noun1 = set([wnl.lemmatize(ele.lower_) for ele in q1 if ele.pos_ == 'NOUN'])
    set_noun2 = set([wnl.lemmatize(ele.lower_) for ele in q2 if ele.pos_ == 'NOUN'])

    num_noun, val_noun, rate_noun = feat(set_noun1, set_noun2)

    set_verb1 = set([wnl.lemmatize(ele.lower_) for ele in q1 if ele.pos_ == 'VERB'])
    set_verb2 = set([wnl.lemmatize(ele.lower_) for ele in q2 if ele.pos_ == 'VERB'])

    num_verb, val_verb, rate_verb = feat(set_verb1, set_verb2)

    set_adv1 = set([wnl.lemmatize(ele.lower_) for ele in q1 if ele.pos_ == 'ADV'])
    set_adv2 = set([wnl.lemmatize(ele.lower_) for ele in q2 if ele.pos_ == 'ADV'])

    num_adv, val_adv, rate_adv = feat(set_adv1, set_adv2)

    # set_adj1 = set([wnl.lemmatize(ele.lower_) for ele in q1 if ele.pos_ == 'ADJ'])
    # set_adj2 = set([wnl.lemmatize(ele.lower_) for ele in q2 if ele.pos_ == 'ADJ'])

    # num_adj, val_adj, rate_adj = feat(set_adj1, set_adj2)

    set_svo1 = set([(ele[0].lower(), ele[1].lower(), ele[2].lower()) for ele in findSVOs(q1)])
    set_svo2 = set([(ele[0].lower(), ele[1].lower(), ele[2].lower()) for ele in findSVOs(q2)])

    set_svo1 = set([(wnl.lemmatize(ele[0]), wnl.lemmatize(ele[1]), wnl.lemmatize(ele[2]))
                    for ele in set_svo1])
    set_svo2 = set([(wnl.lemmatize(ele[0]), wnl.lemmatize(ele[1]), wnl.lemmatize(ele[2]))
                    for ele in set_svo2])

    num_svo, val_svo, rate_svo = feat(set_svo1, set_svo2)

    set_s1 = set(ele[0] for ele in set_svo1)
    set_v1 = set(ele[1] for ele in set_svo1)
    set_o1 = set(ele[2] for ele in set_svo1)

    set_s2 = set(ele[0] for ele in set_svo2)
    set_v2 = set(ele[1] for ele in set_svo2)
    set_o2 = set(ele[2] for ele in set_svo2)

    num_s, val_s, rate_s = feat(set_s1, set_s2)

    num_v, val_v, rate_v = feat(set_v1, set_v2)

    num_o, val_o, rate_o = feat(set_o1, set_o2)

    list_ret = [num_ent, num_rev, num_for, lev_dist, jar_dist, dam_dist,
                num_sub, num_root, num_advmod, num_advcl, num_aux,  # num_poss,
                num_noun, num_verb, num_adv,  # num_adj,
                num_svo, num_s, num_v, num_o]
    list_ret += [val_ent, val_rev, val_for,
                 val_sub, val_root, val_advmod, val_advcl, val_aux, val_dobj,  # val_poss,
                 val_noun, val_verb, val_adv,  # val_adj,
                 val_svo, val_s, val_v, val_o]
    list_ret += [rate_ent,
                 rate_sub, rate_root, rate_advmod, rate_advcl, rate_aux, rate_dobj,  # rate_poss,
                 rate_noun, rate_verb, rate_adv,  # rate_adj,
                 rate_svo, rate_s, rate_v, rate_o]

    return list_ret


if __name__ == '__main__':
    from logging import StreamHandler, DEBUG, Formatter, FileHandler

    log_fmt = Formatter('%(asctime)s %(name)s %(lineno)d [%(levelname)s][%(funcName)s] %(message)s ')
    handler = FileHandler('svo.py.log', 'w')
    handler.setLevel(DEBUG)
    handler.setFormatter(log_fmt)
    logger.setLevel(DEBUG)
    logger.addHandler(handler)

    handler = StreamHandler()
    handler.setLevel('INFO')
    handler.setFormatter(log_fmt)
    logger.setLevel('INFO')
    logger.addHandler(handler)
    p = Pool()
    df = pandas.read_csv('../data/train.csv',
                         usecols=['question1', 'question2']).values

    ret = numpy.array(list(p.map(load_data, df)))
    print(ret[:100])
    with open('train_svo.pkl', 'wb') as f:
        pickle.dump(ret, f, -1)
    logger.info('tran end')
    df = pandas.read_csv('../data/test.csv',
                         usecols=['question1', 'question2']).values

    ret = numpy.array(list(p.map(load_data, df)))
    with open('test_svo.pkl', 'wb') as f:
        pickle.dump(ret, f, -1)

    p.close()
    p.join()
