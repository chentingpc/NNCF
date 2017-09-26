import numpy as np
import tensorflow as tf
import keras.backend as K
from keras.models import Sequential, Model
from keras.layers import Input, Embedding, Lambda
from keras.layers import merge, Merge, Reshape
from utilities import activity_l2

from objectives import get_original_loss, get_neg_shared_loss
from objectives import get_sampled_neg_shared_loss, get_group_neg_shared_loss
from mean_pool import MeanPool
from cnn_model import CNN
from rnn_model import RNN
from vec2vec import ItemCombination
from interaction_dot import InteractionDot


def get_model(conf, data_helper, model_name):
    # retrieve model configurations
    max_epoch = conf.max_epoch
    num_negatives = conf.num_negatives
    batch_size_p = conf.batch_size_p
    eval_topk = conf.eval_topk
    optimizer = conf.optimizer
    loss = conf.loss
    user_dim = conf.user_dim
    item_dim = conf.item_dim
    data_spec = data_helper.data_spec
    user_count = data_spec.user_count
    item_count = data_spec.item_count
    word_count = data_spec.word_count
    emb_normalization = conf.emb_normalization
    max_content_len = data_spec.max_content_len
    support_groupping_for_all = True  # provide general speed-up

    # standard input & output of the model
    input_dtype = 'int32'
    row_cidx_prefx = tf.Variable(np.arange(batch_size_p, \
        dtype=input_dtype).reshape((batch_size_p, 1)))
    uid = Input(shape=(1,), dtype=input_dtype)
    cid = Input(shape=(1,), dtype=input_dtype)
    U_emb_given = Input(shape=(user_dim,), dtype='float32')
    C_emb_given = Input(shape=(item_dim,), dtype='float32')
    # unique cid, in two Lambda thanks to Keras, dummy
    cid_u = Lambda(lambda x: tf.reshape(tf.unique(x)[0], (-1, 1)), 
                   output_shape=(1,))(Reshape(())(cid))
    cid_x = Lambda(lambda x: tf.reshape(tf.unique(x)[1], (-1, 1)),
                   output_shape=(1,))(Reshape(())(cid))

    # retrieve content
    with tf.device('/cpu:0'):
        C = tf.Variable(data_helper.data['C'])
        get_content = lambda x: tf.reshape(tf.gather(C, x), 
                                           (-1, max_content_len))
        content = Lambda(get_content, output_shape=(max_content_len, ))(cid)
        content_u = Lambda(get_content, output_shape=(max_content_len, ))(cid_u)

    # user embedding: U_emb, U_emb_front (first batch_size_p)
    Emb_U = Embedding(user_count, user_dim, name='user_embedding',
                      activity_regularizer=activity_l2(conf.u_reg))
    U_emb = Reshape((user_dim, ))(Emb_U(uid))
    if emb_normalization:
        U_emb = Lambda(lambda x: tf.nn.l2_normalize(x, dim=-1))(U_emb)
    uid_front = Lambda(lambda x: x[:batch_size_p])(uid)  # thanks keras, dummy
    U_emb_front = Reshape((user_dim, ))(Emb_U(uid_front))

    # item embedding: C_emb_compact (no duplication), C_emb
    get_item_emb_combined_pretrain = ItemCombination().get_model()
    if model_name == 'pretrained':
        if conf.evaluation_mode:
            Emb_U = Embedding(user_count, user_dim, trainable=False,
                              weights=[conf.pretrain['user_emb']])
            U_emb = Reshape((user_dim, ))(Emb_U(uid))
            Emb_C = Embedding(item_count, item_dim, trainable=False,
                              weights=[conf.pretrain['item_emb']])
            C_emb = Reshape((item_dim, ))(Emb_C(cid))
        else:
            if conf.pretrain['transform']:
                C_emb = get_item_emb_combined_pretrain(None, cid, conf, data_spec)
            else:
                Emb_C = Embedding(item_count, item_dim, trainable=False,
                                  weights=[data_spec.C_pretrain])
                C_emb = Reshape((item_dim, ))(Emb_C(cid))
        C_emb_compact = C_emb
    elif model_name == 'mf':
        Emb_C = Embedding(item_count, item_dim, name='item_embedding')
        C_emb = Reshape((item_dim, ))(Emb_C(cid))
        C_emb_compact = C_emb  # remember to set model_group_neg_shared = model_neg_shared
        # C_emb_compact = Reshape((item_dim, ))(Emb_C(cid_u))  # can increase overhead.
    else:
        if model_name == 'basic_embedding':
            Content_model = MeanPool(data_spec, conf).get_model()
        elif model_name == 'cnn_embedding':
            Content_model = CNN(data_spec, conf).get_model()
        elif model_name == 'rnn_embedding':
            Content_model = RNN(data_spec, conf).get_model()
        else:
            assert False, '[ERROR] Model name {} unknown'.format(model_name)
        C_emb_compact = Content_model([content_u, cid_u])  # (None, item_dim)
        C_emb_compact = get_item_emb_combined_pretrain(C_emb_compact, cid_u, \
            conf, data_spec) # (None, item_dim)
        # C_emb_u only computes unique set of items, no duplication
        C_emb_u = Lambda( \
            lambda x: tf.reshape(tf.gather(x[0], x[1]), (-1, item_dim)), \
            output_shape=(item_dim, ))([C_emb_compact, cid_x])
        if support_groupping_for_all:
            C_emb = C_emb_u
        else:  # otherwise only support groupping for group_neg_shared
            C_emb = Content_model([content, cid])  # (None, item_dim)
        if emb_normalization:
            C_emb_compact = Lambda(lambda x: tf.nn.l2_normalize(x, dim=-1))(C_emb_compact)
            C_emb = Lambda(lambda x: tf.nn.l2_normalize(x, dim=-1))(C_emb)
    
    # item embedding more: C_emb_front, C_emb_back
    cid_front = Lambda(lambda x: x[:batch_size_p])(cid)
    cid_back = Lambda(lambda x: x[batch_size_p:])(cid)
    C_emb_front = Lambda(lambda x: x[:batch_size_p])(C_emb)
    C_emb_back = Lambda(lambda x: x[batch_size_p:])(C_emb)

    # interact (with or without bias)
    Interact = InteractionDot(bias=conf.interaction_bias, 
                              user_count=user_count, item_count=item_count)

    pred_score = Interact.set_form('mul')([U_emb, C_emb, uid, cid])

    pred_score_with_given = Interact.set_form('mul')([U_emb_given, C_emb_given,
                                                      uid, cid])

    pred_score_neg_shared = Interact.set_form('matmul')([U_emb, C_emb, 
                                                         uid, cid])

    pred_score_neg_shared_comp = Interact.set_form('matmul')([ \
        U_emb, C_emb_compact, uid, cid_u])
    pos_idxs = tf.concat([row_cidx_prefx, \
        tf.reshape(cid_x, (-1, 1))], 1)  # (batch_size_p, 2)
    loss_neg_shared_comp = get_group_neg_shared_loss( \
        pred_score_neg_shared_comp, pos_idxs, loss, batch_size_p, conf)

    pred_pos_sampled_neg_shared = Interact.set_form('mul')([ \
        U_emb_front, C_emb_front, uid_front, cid_front])  # (batch_size_p, 1)
    pred_neg_sampled_neg_shared = Interact.set_form('matmul')([ \
        U_emb_front, C_emb_back, uid_front, cid_back])  # (batch_size_p, num_negatives)
    pred_score_sampled_neg_shared = Lambda(lambda x: tf.concat([x[0], x[1]], 1))( \
        [pred_pos_sampled_neg_shared, pred_neg_sampled_neg_shared])

    # uid-cid element-wise interaction
    # during training, first batch_size_p assumed positive
    model = Model(input=[uid, cid], output=[pred_score])
    model.compile(optimizer=optimizer, \
        loss=get_original_loss(loss, batch_size_p, num_negatives, conf))

    # uid-cid complete pairwise interaction (produce prediction matrix)
    # during training, diag is assumed positive
    model_neg_shared = Model(input=[uid, cid], output=[pred_score_neg_shared])
    model_neg_shared.compile(optimizer=optimizer, \
        loss=get_neg_shared_loss(loss, batch_size_p, conf))

    # uid and compacted cid complete pairwise interactions
    model_group_neg_shared = Model(input=[uid, cid], \
        output=[pred_score_neg_shared_comp])
    model_group_neg_shared.compile(optimizer=optimizer, \
        loss=lambda y_true, y_pred: loss_neg_shared_comp)  # dummy
    if model_name == "mf":
        model_group_neg_shared = model_neg_shared
    
    # sampled negatives are shared
    # first batch_size_p pairs are positive ones, 
    # uid[:batch_size_p] and cid[batch_size_p:] are negative links
    model_sampled_neg_shared = Model(input=[uid, cid], \
        output=[pred_score_sampled_neg_shared])
    model_sampled_neg_shared.compile(optimizer=optimizer, \
        loss=get_sampled_neg_shared_loss(loss, batch_size_p, 
                                         num_negatives, conf))

    # test efficient methods with given (uid, cid) pairs
    model_user_emb = Model(input=[uid], output=[U_emb])
    model_item_emb = Model(input=[cid], output=[C_emb])
    model_pred_pairs = Model(input=[U_emb_given, C_emb_given, uid, cid], \
        output=[pred_score_with_given])

    # construct models for monitoring all types of losses during training
    def get_all_losses(input, output, loss):
        model_all_loss = {'skip-gram': None, 'mse': None,
                          'log-loss': None, 'max-margin': None}
        for lname in model_all_loss:
            from keras.optimizers import SGD
            m = Model(input=input, output=output)
            m.compile(optimizer=SGD(0.), loss=loss)
            model_all_loss[lname] = m
        return model_all_loss

    model_all_loss = get_all_losses([uid, cid], [pred_score], \
        get_original_loss(loss, batch_size_p, num_negatives, conf))
    model_neg_shared_all_loss = get_all_losses([uid, cid], \
        [pred_score_neg_shared], \
        get_neg_shared_loss(loss, batch_size_p, conf))

    model_dict = {'model': model,
                  'model_neg_shared': model_neg_shared,
                  'model_group_neg_shared': model_group_neg_shared,
                  'model_sampled_neg_shared': model_sampled_neg_shared,
                  'model_user_emb': model_user_emb,
                  'model_item_emb': model_item_emb,
                  'model_pred_pairs': model_pred_pairs,
                  'model_all_loss': model_all_loss,
                  'model_neg_shared_all_loss': model_neg_shared_all_loss
                  }

    return model_dict
