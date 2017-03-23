from train_sparse import feat
from scipy.sparse import vstack
import pickle
from sklearn.decomposition import TruncatedSVD

with open('train_sparse.pkl', 'rb') as f:
    idf, count, tfidf = pickle.load(f)
    x_train = feat(idf, count, tfidf)
with open('test_sparse.pkl', 'rb') as f:
    idf, count, tfidf = pickle.load(f)
    x_test = feat(idf, count, tfidf)

train_num = x_train.shape[0]

data = vstack([x_train, x_test])


svd = TruncatedSVD(n_components=300, random_state=0)
data = svd.fit_transform(data)


from features_tic import FEATURE as feat_tic
cols = feat_tic[:2000]
train_num = x_train.shape[0]

data = vstack([x_train, x_test])[:, cols]
from sklearn.decomposition import TruncatedSVD

svd = TruncatedSVD(n_components=300, random_state=0)
data = svd.fit_transform(data)

with open('train_svd_2000_300.pkl', 'wb') as f:
    pickle.dump(data[:train_num], f, -1)
with open('test_svd_2000_300.pkl', 'wb') as f:
    pickle.dump(data[train_num:], f, -1)
