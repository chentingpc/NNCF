import os
import sys
import numpy as np
import cPickle as pickle

root_dir = sys.argv[1]
val_ratio = 0.1

for parent, dirnames, filenames in os.walk(root_dir):
    for filename in filenames:
        path = os.path.join(parent, filename)
        if path.find('data_split_cold_item.pkl') < 0:
            continue
        with open(path) as fp:
            data = pickle.load(fp)
        train_items = data['train_items']
        train = data['train']
        num_train_items = len(train_items)
        num_train = train.shape[0]

        # create new val based on train
        np.random.shuffle(train_items)
        test_len = int(len(train_items) * val_ratio)
        test_items = train_items[:test_len]
        test_items_set = set(test_items)
        train_items = train_items[test_len:]
        tt_idx = np.array([each in test_items_set for each in train[:, 1]])
        test = train[tt_idx]
        train = train[~tt_idx]
        test_seen = train[:test.shape[0]]
        data['train_items'] = train_items
        data['test_items'] = test_items
        data['train'] = train
        data['test_seen'] = test_seen
        data['test'] = test

        # sanity check
        assert len(train_items) + len(test_items) == num_train_items
        assert data['train'].shape[0] + data['test'].shape[0] == num_train
        assert len(test_items_set.intersection(train_items)) == 0
        with open(path, 'w') as fp:
            pickle.dump(data, fp, 2)
