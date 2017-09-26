#!/usr/bin/env python

"""scoring.py: Script that demonstrates the multi-label classification used."""

__author__      = "Bryan Perozzi"


import sys
import numpy
import cPickle as pickle
numpy.random.seed(43)
from tqdm import tqdm
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression
from itertools import izip, product
from sklearn.metrics import f1_score
from scipy.io import loadmat
from sklearn.utils import shuffle as skshuffle

from collections import defaultdict
import gensim
from gensim.models import Word2Vec

class TopKRanker(OneVsRestClassifier):
    def predict(self, X, top_k_list):
        assert X.shape[0] == len(top_k_list)
        probs = numpy.asarray(super(TopKRanker, self).predict_proba(X))
        all_labels = []
        for i, k in enumerate(top_k_list):
            probs_ = probs[i, :]
            labels = self.classes_[probs_.argsort()[-k:]].tolist()
            all_labels.append(labels)
        return all_labels

def sparse2graph(x):
    G = defaultdict(lambda: set())
    cx = x.tocoo()
    for i,j,v in izip(cx.row, cx.col, cx.data):
        G[i].add(j)
    return {str(k): [str(x) for x in v] for k,v in G.iteritems()}

# 0. Files
embeddings_file = sys.argv[1]
matfile = sys.argv[2]
method = sys.argv[3].lower()
if method == 'line':
    n_offset = 1
    binary = True
    idx_op = str
    loader = "w2v"
elif method == 'node2vec':
    n_offset = 1
    binary = False
    idx_op = str
    loader = "w2v"
else:
    n_offset = 1
    binary = False
    idx_op = int
    loader = "pkl"

# 1. Load Embeddings
if loader == "w2v":
    #model = Word2Vec.load_word2vec_format(embeddings_file, binary=binary)
    model = gensim.models.KeyedVectors.load_word2vec_format(embeddings_file, binary=binary)
else:
    with open(embeddings_file) as fp:
        model = pickle.load(fp)

# 2. Load labels
mat = loadmat(matfile)
A = mat['network']
graph = sparse2graph(A)
labels_matrix = mat['group']

# Map nodes to their features (note:  assumes nodes are labeled as integers 1:N)
features_matrix = numpy.asarray([model[idx_op(node + n_offset)] for node in range(len(graph))])
# features_matrix = numpy.asarray([model[idx_op(int(node) + n_offset)] for node in sorted(graph)])
#features_matrix = features_matrix[:10000]
#labels_matrix = labels_matrix[:10000]
import numpy as np
selector = np.array(labels_matrix.sum(axis=1) > 0).reshape(-1)
features_matrix = features_matrix[selector]
labels_matrix = labels_matrix[selector]

# 2. Shuffle, to create train/test groups
shuffles = []
number_shuffles = 1
print 'number_shuffles', number_shuffles
for x in range(number_shuffles):
  shuffles.append(skshuffle(features_matrix, labels_matrix))

# 3. to score each train/test group
all_results = defaultdict(list)

# 4. create multi_binarizer
y_ = shuffles[0][1]
y = [[] for x in xrange(y_.shape[0])]
cy =  y_.tocoo()
for i, j in izip(cy.row, cy.col):
    y[i].append(j)
assert sum(len(l) for l in y) == y_.nnz
from sklearn.preprocessing import MultiLabelBinarizer
multi_binarizer = MultiLabelBinarizer()
multi_binarizer.fit(y)

training_percents = [0.1]
#training_percents = [0.1, 0.5, 0.9]
print 'training_percents', training_percents
# uncomment for all training percents
#training_percents = numpy.asarray(range(1,10))*.1
#for train_percent in training_percents:
#  for shuf in shuffles:
for train_percent, shuf in tqdm(product(training_percents, shuffles)):

    X, y = shuf

    training_size = int(train_percent * X.shape[0])

    X_train = X[:training_size, :]
    y_train = y[:training_size]

    X_test = X[training_size:, :]
    y_test = y[training_size:]

    clf = TopKRanker(LogisticRegression(solver='liblinear', n_jobs=4))
    clf.fit(X_train, y_train)

    # find out how many labels should be predicted
    top_k_list = [l.getnnz() for l in y_test]
    preds = clf.predict(X_test, top_k_list)

    preds = multi_binarizer.transform(preds)

    results = {}
    averages = ["micro", "macro", "samples", "weighted"]
    for average in averages:
        results[average] = f1_score(y_test,  preds, average=average)

    all_results[train_percent].append(results)

print 'Results, using embeddings of dimensionality', X.shape[1]
print '-------------------'
for train_percent in sorted(all_results.keys()):
  print 'Train percent:', train_percent
  for x in all_results[train_percent]:
    print  x
  print '-------------------'
