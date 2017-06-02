from os.path import expanduser
home = expanduser('~')
import numpy as np
import random
from data_utils import get_pretrain_folder
from conf_utils import get_conf_base
from keras.optimizers import Adam, SGD, RMSprop


class Conf(object):
    def __init__(self, data_name, param_dict=None):
        self.data_name = data_name
        self.max_epoch = 20
        self.num_negatives = 5
        self.batch_size_p = 64
        self.eval_topk = 50

        self.loss = 'skip-gram'
        #self.loss = 'max-margin'
        #self.loss_margin = 1
        #self.loss = 'log-loss'
        self.neg_loss_weight = 1
        self.use_content_id = False
        self.v_reg = 0
        self.interaction_multiplier = False
        self.interaction_bias = 'item'
        self.emb_normalization = None

        self.learn_rate = 0.01
        self.user_dim = self.item_dim = 50
        self.u_reg = 1e-5
        self.neg_dist = 'uniform'
        self.neg_sampling_power = 1
        self.chop_size = 1
        self.shuffle_st = 'by_item'

        self.evaluation_mode = False

        if param_dict is not None:
            self.__dict__.update(param_dict)
        self._post_init()

    def _post_init(self):
        if self.emb_normalization is None:
            self.emb_normalization = True \
                if self.loss == 'max-margin' or self.loss == 'log-loss' \
                else False

        if self.learn_rate > 0:
            self.optimizer = Adam(self.learn_rate)
        else:
            from optimizer import AdamOptimizer
            #import tensorflow as tf
            #self.optimizer = tf.train.GradientDescentOptimizer(1e4)
            self.optimizer = AdamOptimizer(-self.learn_rate)

        pretrain_folder = get_pretrain_folder(self.data_name, aug=False)
        sentvec_filepath = pretrain_folder + 'sentence_vectors_50d.txt'
        self.pretrain = {'wordvec_filepath': None,
                         'sentvec_filepath': sentvec_filepath,
                         'transform': True,
                         'pretrain_combine_dropout': 0.5,
                         'pretrain_combine_actv': 'relu',
                         'pretrain_combine_mode': None}
        self.pretrain['transform'] = False

        self.contextual_spatial_gated_input = None
        self.contextual_temporal_gated_input = None

def get_conf_default(data_name, param_dict=None):
    conf_seed, conf_var = get_conf_base(param_dict)
    conf = Conf(data_name, param_dict=param_dict)
    if conf_seed:
        conf.conf_seed = conf_seed
    if conf_var:
        conf.conf_var = conf_var
    return conf

def get_conf_evaluation(data_name, param_dict=None):
    # user/word dim is fixed to 50 for comparisons
    conf_seed, conf_var = get_conf_base(param_dict)
    conf = Conf(data_name, param_dict=param_dict)
    if conf_seed:
        conf.conf_seed = conf_seed
    if conf_var:
        conf.conf_var = conf_var
    conf.evaluation_mode = True
    conf.max_epoch = 3
    conf.interaction_bias = None
    conf.emb_normalization = False
    conf.optimizer = SGD(0)
    data_name = data_name.replace('_fold1', '')
    emb_folder = home + '/pbase/x/p607/results-aaai/baselines/CDL/results/{}/fold1/cdl/'.format(data_name)
    if data_name.startswith('citeulike'):
        user_emb_file = emb_folder + '10-U.dat'
        item_emb_file = emb_folder + '10-V.dat'
    else:
        user_emb_file = emb_folder + 'final-U.dat'
        item_emb_file = emb_folder + 'final-V.dat'
    user_emb = np.loadtxt(user_emb_file)
    item_emb = np.loadtxt(item_emb_file)
    conf.pretrain = {'wordvec_filepath': None,
                     'sentvec_filepath': None,
                     'user_emb': user_emb,
                     'item_emb': item_emb,
                     'transform': False,
                     'pretrain_combine_dropout': 0.,
                     'pretrain_combine_actv': 'linear',
                     'pretrain_combine_mode': None}
    return conf                    

def get_conf_best(data_name, param_dict=None):
    # user/word dim is fixed to 50 for comparisons
    conf_seed, conf_var = get_conf_base(param_dict)
    conf = Conf(data_name, param_dict=param_dict)
    if conf_seed:
        conf.conf_seed = conf_seed
    if conf_var:
        conf.conf_var = conf_var
    conf.c_reg = 0
    conf.max_epoch = 30
    conf.num_negatives = 10
    conf.interaction_bias = None
    conf.pretrain['transform'] = True
    if data_name.startswith('citeulike_title_only'):
        conf.u_reg = 1e-6
        conf.interaction_multiplier = False
        conf.pretrain['pretrain_combine_dropout'] = 0.3
    elif data_name.startswith('citeulike_title_and_abstract'):
        conf.u_reg = 1e-6
        conf.interaction_multiplier = False
        conf.pretrain['pretrain_combine_dropout'] = 0.3
    elif data_name.startswith('news_title_only'):
        conf.u_reg = 1e-5
        conf.interaction_multiplier = False
        conf.pretrain['pretrain_combine_dropout'] = 0.1
    elif data_name.startswith('news_title_and_abstract'):
        conf.u_reg = 1e-5
        conf.interaction_multiplier = False
        conf.pretrain['pretrain_combine_dropout'] = 0.1
    else:
        assert False, 'unknown data_name: %s' % data_name
    try:
        param_dict['reset_after_getconf']
        conf.__dict__.update(param_dict)
    except:
        pass
    return conf


def get_conf_random(data_name, param_dict=None):
    conf_seed, conf_var = get_conf_base(param_dict)
    if conf_seed:
        random.seed(conf_seed)
    conf = Conf(data_name, param_dict=param_dict)
    if conf_seed:
        conf.conf_seed = conf_seed
    if conf_var:
        conf.conf_var = conf_var
    conf.u_reg = random.choice([0., 1e-5, 1e-6, 1e-7])
    conf.pretrain['transform'] = random.choice([True, False])
    if conf.pretrain['transform']:
        conf.pretrain['pretrain_combine_dropout'] = [0.1, 0.3, 0.5]  # choose one?
        
    return conf

def get_conf(data_name, conf_choice, param_dict=None):
    if conf_choice == 'best':
        conf = get_conf_best(data_name, param_dict=param_dict)
    elif conf_choice == 'default':
        conf = get_conf_default(data_name, param_dict=param_dict)
    elif conf_choice == 'random':
        conf = get_conf_random(data_name, param_dict=param_dict)
    elif conf_choice == 'evaluation':
        conf = get_conf_evaluation(data_name, param_dict=param_dict)
    else:
        assert False, '[Error!] conf_choice {} unknonw'.format(conf_choice)
    return conf
