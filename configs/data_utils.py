import re
import random
import cPickle as pickle
import numpy as np
import pandas as pd
from numba import jit
from data_helper import DataHelper
from os.path import expanduser
from sampler import MultinomialSampler
from collections import defaultdict

home = expanduser('~')
data_root =  './data'

def get_data(data_name, conf, reverse_samping=False):
    # prepare data
    data_helper = _get_data(data_name)
    train = data_helper.data['train']
    test_seen = data_helper.data['test_seen']
    test = data_helper.data['test']
    C = data_helper.data['C']

    user_count = max(np.max(train[:, 0]), np.max(test[:, 0])) + 1
    word_count = np.max(C) + 1
    item_count = C.shape[0]
    max_content_len = C.shape[1]
    class DataSpec(object):
        def __init__(self, user_count, word_count,
                     item_count, max_content_len):
            self.user_count = user_count
            self.word_count = word_count
            self.item_count = item_count
            self.max_content_len = max_content_len
    data_spec = DataSpec(user_count, word_count,
                         item_count, max_content_len)
    if conf is not None:
        data_spec.W_pretrain, data_spec.C_pretrain = get_pretrained_vectors(conf, data_spec, data_helper)

        neg_dist = conf.neg_dist
        try:
            neg_sampling_power = conf.neg_sampling_power
            # print '[INFO] Setting neg_sampling_power to user defined {}'.format(neg_sampling_power)
        except:
            neg_sampling_power = 0.75
        sampler_dict = {}
        # user negative sampling (if needed)
        if reverse_samping:
            train = train[:, [1, 0, 2]]
            sample_u = get_sampler(train, neg_dist=neg_dist,
                                   neg_sampling_power=neg_sampling_power,
                                   batch_mode=False)
            sample_batch_u = get_sampler(train, neg_dist=neg_dist,
                                         neg_sampling_power=neg_sampling_power,
                                         batch_mode=True)
            train = train[:, [1, 0, 2]]
            sampler_dict['sample_u'] = sample_u
            sampler_dict['sample_batch_u'] = sample_batch_u
        # item negative sampling
        sample = get_sampler(train, neg_dist=neg_dist,
                             neg_sampling_power=neg_sampling_power,
                             batch_mode=False)
        sample_batch = get_sampler(train, neg_dist=neg_dist,
                                   neg_sampling_power=neg_sampling_power,
                                   batch_mode=True)
        sampler_dict['sample'] = sample
        sampler_dict['sample_batch'] = sample_batch
        data_helper.sampler_dict = sampler_dict
    data_helper.data_spec = data_spec

    return data_helper


def _get_data(data_name='citeulike_title_only'):
    data_helper = DataHelper()
    def error():
        assert False, '[ERROR] unseen data_name %s' % data_name
    sub_folder = ''
    fold = re.findall('fold(\d+)', data_name)
    if len(fold) == 1:
        sub_folder = 'fold%d' % int(fold[0])
    if data_name.startswith('citeulike'):
        if data_name.startswith('citeulike_title_only'):
            content_file = data_root + '/citeulike/title_only/%s/data_content.pkl' % sub_folder
            split_file = data_root + '/citeulike/title_only/%s/data_split_cold_item.pkl' % sub_folder
        elif data_name.startswith('citeulike_title_and_abstract'):
            content_file = data_root + '/citeulike/title_and_abstract/%s/data_content.pkl' % sub_folder
            split_file = data_root + '/citeulike/title_and_abstract/%s/data_split_cold_item.pkl' % sub_folder
        else:
            error()
    elif data_name.startswith('news'):
        if  data_name.startswith('news_title_only'):
            content_file = data_root + '/news/title_only/%s/data_content.pkl' % sub_folder
            split_file = data_root + '/news/title_only/%s/data_split_cold_item.pkl' % sub_folder
        elif  data_name.startswith('news_title_and_abstract'):
            content_file = data_root + '/news/title_and_abstract/%s/data_content.pkl' % sub_folder
            split_file = data_root + '/news/title_and_abstract/%s/data_split_cold_item.pkl' % sub_folder
    else:
        error()

    # data_helper.load_data(content_file)
    with open(split_file) as fp:
        split_data = pickle.load(fp)
    data_helper.data = split_data
    return data_helper


def get_pretrain_folder(data_name, aug=True):
    # used in conf
    if data_name.startswith('citeulike_title_only'):
        data_name_short = 'citeulike/title_only/'
    elif data_name.startswith('citeulike_title_and_abstract'):
        data_name_short = 'citeulike/title_and_abstract/'
    elif data_name.startswith('news_title_only'):
        data_name_short = 'news/title_only/'
    elif data_name.startswith('news_title_and_abstract'):
        data_name_short = 'news/title_and_abstract/'
    else:
        assert False, 'data_name %s unkown' % data_name

    if aug:
        aug = 'aug'
    else:
        aug = ''

    pretrain_folder = data_root + '/%s/pretrain/%s/' % (data_name_short, aug)
    return pretrain_folder


def get_pretrained_vectors(conf, data_spec, data_helper):
    # load pretrained word vectors, and sentence vectors
    if conf.pretrain:
        wordvec_filepath = conf.pretrain['wordvec_filepath']
        sentvec_filepath = conf.pretrain['sentvec_filepath']
    else:
        wordvec_filepath = None
        sentvec_filepath = None

    if wordvec_filepath:
        if wordvec_filepath.endswith('pkl'):
            # matrix in pkl format
            with open(wordvec_filepath) as fp:
                W_pretrain = pickle.load(fp)
        else:
            # original text format
            # discard the first line, each line is word emb0 emb1 ...
            # last column is NaN and discarded
            df = pd.read_csv(wordvec_filepath, skiprows=1, header=None, sep=' ')
            df.index = df[0]
            del df[0]
            del df[df.shape[1]]

            W_pretrain = np.zeros((data_spec.word_count, df.shape[1]))
            for word, wid in data_helper.word2id.iteritems():
                try:
                    w = df.loc[word]
                except:
                    continue
                W_pretrain[wid] = w
    else:
        W_pretrain = None

    if sentvec_filepath:
        if sentvec_filepath.endswith('pkl'):
            # matrix in pkl format
            with open(sentvec_filepath) as fp:
                C_pretrain = pickle.load(fp)
        else:
            # original text format
            # each line is id emb0 emb1 ...
            # last column is NaN and discarded
            df = pd.read_csv(sentvec_filepath, skiprows=0, header=None, sep=' ')
            df[0] = df[0].apply(lambda x: x[2:])
            df[0] = df[0].astype('int')
            df.index = df[0]
            del df[0]
            del df[df.shape[1]]
            item_count = data_spec.item_count
            C_pretrain = np.zeros((item_count, df.shape[1]))
            for row in df.iterrows():
                if row[0] < item_count:
                    C_pretrain[row[0]] = row[1]
    else:
        C_pretrain = None

    return W_pretrain, C_pretrain

@jit
def count_jit(ls, c):
    # count list into dict
    for l in ls: c[l] += 1

# prepare sampler for training
def get_sampler(ratings, neg_dist='unigram', neg_sampling_power=0.75,
                column=1, rand_seed=0, batch_mode=True):
    '''
    Return a sampling function

    Assuming the triplets are in shape of (user, item, rating), i.e. dst/item will be sampled as
        negatives
    '''
    neg_dist = neg_dist.split('_')[0]  # support suffix tricks
    assert neg_dist == 'uniform' or neg_dist == 'unigram', [neg_dist]

    dist = [0. for _ in range(np.max(ratings[:, column]) + 1)]
    count_jit(ratings[:, column], dist)
    dist = np.array(dist, dtype=float)
    if neg_dist == 'uniform':
        dist[dist > 0] = 1
        # dist[:] = 1

    s = MultinomialSampler(dist, dist.size, neg_sampling_power, rand_seed)
    if batch_mode:
        return s.sample_batch
    else:
        return s.sample


def group_shuffle_train(train, by='item', chop=0, iidx=None):
    ''' shuffle train randomly but group by items (or users), e.g.
        (2, 1, _), (1, 1, _), (44, 1,  _), (3, 0, _), (1, 0, _), ...
        train: in format of user item response(optional)
        chop: segment into size of chop, then shuffle.
            in expectation, when chop is small, chop will be close to
            average links per group/item/user in a mini-batch
        iidx : idx list of all users/items
    '''
    if by == 'user': col = 0
    else: col = 1
    if iidx is None:
        iidx = np.arange(np.max(train[:, col]) + 1)
    np.random.shuffle(iidx)
    np.random.shuffle(train)
    train = np.hstack((train, iidx[train[:, col]].reshape((train.shape[0], 1))))
    train = train[train[:, -1].argsort()][:, :-1]
    if chop > 0:
        bulk_len = (train.shape[0] // chop) * chop
        train_bulk = train[:bulk_len].reshape((-1, chop, train.shape[-1]))
        np.random.shuffle(train_bulk)
        train_bulk = train_bulk.reshape((-1, train_bulk.shape[-1]))
        train = np.vstack([train_bulk, train[bulk_len:]])
    return train


class GroupSampler(object):
    ''' First sample a group/item, then sample its positive members/users
        Followed by sampling negative members/users (whose number is decided)
    '''
    def __init__(self, train, group_by='item', chop=1,
                 neg_dist='unigram', neg_sign=0):
        ''' chop: int
            group size for each item/user

            neg_dist & neg_sign only used when sample_with_negs() called
        '''
        if neg_dist == 'uniform_no_correction':
            neg_dist = 'uniform'
            no_correction = True
        else:
            no_correction = False
        self.train = train
        self.group_by = group_by
        self.chop = chop
        self.neg_dist = neg_dist
        self.neg_sign = neg_sign

        if group_by == 'item':
            group_column = 1
            member_column = 0
        elif group_by == 'user':
            group_column = 0
            member_column = 1
        else:
            assert False, '[ERROR] Illegal group_by {}'.format(group_by)

        try:
            neg_sampling_power = conf.neg_sampling_power
            print '[INFO] In GroupSampler, setting neg_sampling_power' \
                  'to user defined {}'.format(neg_sampling_power)
        except:
            neg_sampling_power = 0.75
        self.group_sampler = get_sampler(train, neg_dist='unigram', \
            neg_sampling_power=1., column=group_column, batch_mode=True)
        self.member_neg_sampler = get_sampler(train, neg_dist=neg_dist, \
            neg_sampling_power=neg_sampling_power, column=member_column, \
            batch_mode=True)

        group_set = set(train[:, group_column])
        p_n_div_p_d = {}
        if neg_dist == 'unigram' or no_correction:
            for group in group_set:
                p_n_div_p_d[group] = 1.
        elif neg_dist == 'uniform':
            dist = [0. for _ in range(max(group_set) + 1)]
            count_jit(train[:, group_column], dist)
            dist /= np.sum(dist)
            group_set_size = len(group_set)
            for group in group_set:
                p_n_div_p_d[group] = 1. / group_set_size / dist[group]
        else:
            assert False, '[ERROR] Illegal neg_dist {}'.format(neg_dist)
        self.p_n_div_p_d = p_n_div_p_d

        group_members = defaultdict(list)
        for each in train:
            group_members[each[group_column]].append(each[member_column])
        self.group_members = group_members

    def sample(self, batch_size_p, strict_return_shape=True):
        ''' strict_return_shape: bool
            When set True, the returned batch size has to be batch_size_p
            Otherwise it can be smaller
        '''
        group_members = self.group_members
        if strict_return_shape:
            num_groups = int(np.ceil(float(batch_size_p) / self.chop))
        else:
            num_groups = batch_size_p // self.chop
        sampled_groups = self.group_sampler(num_groups)
        sampled_group_members = []
        for group in sampled_groups:
            gms = group_members[group]
            sampled_group_members.append( \
                np.random.choice(gms, self.chop))
        sampled_batch = np.vstack((np.hstack(sampled_group_members),
                                   np.repeat(sampled_groups, self.chop))).T
        sampled_batch = np.hstack((sampled_batch, \
            np.ones((sampled_batch.shape[0], 1), dtype=int)))
        if self.group_by == 'user':
            sampled_batch = sampled_batch[:, [1, 0, 2]]
        if strict_return_shape:
            return sampled_batch[:batch_size_p]
        else:
            return sampled_batch

    def sample_with_negs(self, batch_size_p, k, strict_return_shape=True):
        ''' k: int
            number of negative items per positive-link

            returned batch in following format:
                first ~batch_size_p are positives, with response = 1
                rest ~batch_size_p * k are negatives, with response = neg_sign
        '''

        group_members = self.group_members
        p_n_div_p_d = self.p_n_div_p_d
        batch_size_whole = batch_size_p * (1 + k)
        if strict_return_shape:
            num_groups = int(np.ceil(float(batch_size_p) / self.chop))
        else:
            num_groups = batch_size_p // self.chop
        sampled_groups = self.group_sampler(num_groups)
        sampled_groups_repeat_pos = []
        sampled_group_members_pos = []
        sampled_groups_repeat_neg = []
        sampled_group_members_neg = []

        def sample_members_for_groups(sampled_groups):
            for group in sampled_groups:
                # sample pos members
                sampled_group_members_pos.append( \
                    np.random.choice(group_members[group], self.chop))
                sampled_groups_repeat_pos.extend([group] * self.chop)

                # sample neg members
                random_rounding = lambda x: int(x) + (random.random() < x - int(x))
                num_negatives = random_rounding(k * self.chop * p_n_div_p_d[group])
                if num_negatives > 0:
                    sampled_groups_repeat_neg.extend([group] * num_negatives)
                    sampled_group_members_neg.append( \
                       self.member_neg_sampler(num_negatives))

        sample_members_for_groups(sampled_groups)

        while strict_return_shape:
            try: _here += 1
            except: _here = 1
            if _here > 10: assert False, \
                '[WARNING] the code here should be optimized if this is shown.'
            curt_batch_size = len(sampled_groups_repeat_pos) + \
                len(sampled_groups_repeat_neg)
            size_diff = batch_size_whole - curt_batch_size
            if size_diff <= 0:
                break
            if size_diff <= 100:
                num_groups_to_sample = size_diff
            else:
                num_groups_to_sample = size_diff // (self.chop * (1 + k)) + 100
            sampled_groups = self.group_sampler(num_groups_to_sample)
            sample_members_for_groups(sampled_groups)

        sampled_batch_pos = np.vstack((np.hstack(sampled_group_members_pos),
                                       sampled_groups_repeat_pos)).T
        sampled_batch_pos = np.hstack((sampled_batch_pos, \
            np.ones((sampled_batch_pos.shape[0], 1), dtype=int)))

        sampled_batch_neg = np.vstack((np.hstack(sampled_group_members_neg),
                                       sampled_groups_repeat_neg)).T
        sampled_batch_neg = np.hstack((sampled_batch_neg, self.neg_sign * \
            np.ones((sampled_batch_neg.shape[0], 1), dtype=int)))

        sampled_batch = np.vstack((sampled_batch_pos, sampled_batch_neg))

        if self.group_by == 'user':
            sampled_batch = sampled_batch[:, [1, 0, 2]]
        if strict_return_shape:
            return sampled_batch[:batch_size_whole]
        else:
            return sample_batch
