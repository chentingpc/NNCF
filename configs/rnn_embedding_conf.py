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

        # loss function related
        self.max_epoch = 40
        self.batch_size_p = 512
        self.num_negatives = 10
        self.loss = 'skip-gram'
        #self.loss = 'max-margin'
        #self.loss = 'log-loss'
        #self.loss = 'mse'
        self.learn_rate = 0.001 if self.loss == 'mse' else 0.01
        self.loss_gamma = 0.1 if self.loss == 'max-margin' else 10
        self.neg_loss_weight = 8 if self.loss == 'mse' else 128
        self.neg_dist = 'unigram'
        self.neg_sampling_power = 1
        self.emb_normalization = None  # set below

        # train scheme related
        self.shuffle_st = 'by_item_chop'
        self.chop_size = 2
        self.group_shuffling_trick = True

        # embedding related
        self.user_dim = self.item_dim = 50
        self.word_dim = 50
        self.u_reg = 1e-5
        self.c_reg = 0
        self.word_emb_dropout_rate = 0.

        # content model related
        self.rnn = 'lstm'
        self.bidirection = True
        self.lstm_dims = [64]
        self.lstm_w_dropout_rate = 0.
        self.lstm_u_dropout_rate = 0.
        self.lstm_o_dropout_rate = 0.3
        self.pooling = 'average'
        self.use_seq_for_dnn = True

        # uninterested
        self.eval_topk = 50
        self.interaction_bias = None
        self.use_content_id = False
        self.v_reg = 0

        if param_dict is not None:
            self.__dict__.update(param_dict)
        self._post_init()

    def _post_init(self):
        if self.emb_normalization is None:
            self.emb_normalization = True \
                if self.loss == 'max-margin' or self.loss == 'log-loss' \
                else False
            
        # optimizer
        if self.learn_rate > 0:
            self.optimizer = Adam(self.learn_rate)
        else:
            print 'using tensorflow optimizer'
            from optimizer import AdamOptimizer
            #import tensorflow as tf
            #self.optimizer = tf.train.GradientDescentOptimizer(1e4)
            self.optimizer = AdamOptimizer(-self.learn_rate)

        # pretrained content embedding
        pretrain_folder = get_pretrain_folder(self.data_name, aug=True)
        wordvec_filepath = pretrain_folder + 'word_vectors_50d.pkl'
        sentvec_filepath = pretrain_folder + 'sentence_vectors_50d.pkl'
        self.pretrain = {'wordvec_filepath': wordvec_filepath,
                         'sentvec_filepath': sentvec_filepath,
                         'pretrain_combine_dropout': 0.5,
                         'pretrain_combine_mode': 'concat',
                         'pretrain_combine_actv': 'relu'}

        # content model: embedding gating and dense layer
        self.contextual_spatial_gated_input = \
            None#{'gating_hidden_dim': 3, 'gating_hidden_actv': 'relu'}
        self.contextual_temporal_gated_input = \
            None#{'gating_hidden_dim': 5, 'gating_hidden_actv': 'relu', 'scale': False, 'choice': 'bn+nl'}
        self.item_dense_transform = \
            {'dense_hidden_dim': self.user_dim,
             'dense_hidden_dropout':0.,
             'dense_hidden_actv': 'relu'}
        

def get_conf_default(data_name, param_dict=None):
    conf_seed, conf_var = get_conf_base(param_dict)
    conf = Conf(data_name, param_dict=param_dict)
    if conf_seed:
        conf.conf_seed = conf_seed
    if conf_var:
        conf.conf_var = conf_var
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
    conf.num_negatives = 10
    conf.rnn = 'lstm'
    conf.lstm_dims = [64]
    if data_name.startswith('citeulike_title_only'):
        conf.max_epoch = 30
        conf.u_reg = 1e-7
        conf.word_emb_dropout_rate = 0.5
        conf.pretrain['pretrain_combine_dropout'] = 0.3
        conf.lstm_o_dropout_rate = 0.3
    elif data_name.startswith('citeulike_title_and_abstract'):
        conf.max_epoch = 30
        conf.u_reg = 1e-7
        conf.word_emb_dropout_rate = 0.5
        conf.pretrain['pretrain_combine_dropout'] = 0.3
        conf.lstm_o_dropout_rate = 0.3
    elif data_name.startswith('news_title_only'):
        conf.max_epoch = 20
        conf.u_reg = 1e-6
        conf.word_emb_dropout_rate = 0.5
        conf.pretrain['pretrain_combine_dropout'] = 0.3
        conf.lstm_o_dropout_rate = 0.3
    elif data_name.startswith('news_title_and_abstract'):
        conf.max_epoch = 20
        conf.u_reg = 1e-6
        conf.word_emb_dropout_rate = 0.0
        conf.pretrain['pretrain_combine_dropout'] = 0.3
        conf.lstm_o_dropout_rate = 0.3
    else:
        assert False, 'unknown data_name: %s' % data_name
    if conf_var == 'sup':
        # another baseline: supervised only
        conf.pretrain = None
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

    conf.max_epoch = 30
    #conf.pretrain = None
    conf.bidirection = True
    conf.use_seq_for_dnn = True
    u_reg_list = [1e-6, 1e-7]
    pretrain_combine_dropout_list = [0.1, 0.3, 0.5]
    lstm_dim_list = [[50]]
    #lstm_dim_list = [[50], [100]]
    #word_emb_dropout_rate_list = [0.3, 0.5]
    #lstm_o_dropout_rate_list = random.choice([0, 0.1, 0.3])
    word_emb_dropout_rate_list = [0.5]
    lstm_o_dropout_rate_list = [0.]

    import itertools    
    conf_i = int(conf_var)
    configs = [each for each in itertools.product( \
        u_reg_list, word_emb_dropout_rate_list, \
        pretrain_combine_dropout_list, lstm_dim_list, lstm_o_dropout_rate_list)]
    conf.u_reg, conf.word_emb_dropout_rate, \
        conf.pretrain['pretrain_combine_dropout'], \
        conf.lstm_dims, conf.lstm_o_dropout_rate = configs[conf_i]
    return conf


def get_conf(data_name, conf_choice, param_dict=None):
    if conf_choice == 'best':
        conf = get_conf_best(data_name, param_dict=param_dict)
    elif conf_choice == 'default':
        conf = get_conf_default(data_name, param_dict=param_dict)
    elif conf_choice == 'random':
        conf = get_conf_random(data_name, param_dict=param_dict)
    else:
        assert False, '[Error!] conf_choice {} unknonw'.format(conf_choice)
    return conf
