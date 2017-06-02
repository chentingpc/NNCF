import numpy as np
import pandas as pd
import cPickle as pickle
from datetime import datetime
import keras.backend as K
from keras.layers import Layer, RepeatVector, Flatten
from keras.regularizers import Regularizer
if K._backend == 'tensorflow':
    import tensorflow as tf
else:
    import theano
    import theano.tensor as T


####################
# common functions
####################

def get_cur_time():
    return datetime.now().strftime('%Y-%m-%d %H:%M:%S')

def get_key_from_val(val, dictionary, sanctity_check=False):
    ''' assuming val is unique '''
    if sanctity_check:
        val_len = len(dictionary.values)
        val_set_len = len(set(dictionary.values))
        assert val_len == val_set_len, [val_len, val_set_len]

    for key, val_ in dictionary.iteritems():
        if val == val_:
            return key

def nan_detection(val_name, val):
    if np.isnan(val):
        assert False, 'ERROR (nan)! {} is {}..'.format(val_name, val)

####################
# math functions
####################

def sigmoid(x):
    return 1./(np.exp(-x) + 1.)

def tanh(x):
    return 2 * sigmoid(2 * x) - 1

def softplus(x):
    return np.log(1 + np.exp(x))

def softmax(x):
    x -= np.max(x, axis=-1, keepdims=True)
    deno = np.sum(np.exp(x), axis=-1)
    return (np.exp(x).T / deno).T

def normalized_embedding(X):
    # n x dim
    return (1. / np.sqrt(np.sum(X**2, axis=1)) * X.T).T


############################
# Keras backend extensions
############################

def theano_reshape(x, new_shape):
    return x.reshape(new_shape)

def tensorflow_reshape(x, new_shape):
    return tf.reshape(x, new_shape)

def theano_diag(x):
    print "NOT implemented!"
    return None

def tensorflow_diag(x, size=None):
    ''' size: square matrix size
        tf.diag_part is very slow!
        so when size given, we use faster gather_nd
    '''
    if size is None:
        return tf.diag_part(x)
    else:
        diag_idx = np.vstack((np.arange(size), np.arange(size))).T
        return tf.gather_nd(x, diag_idx.astype(np.int32))

def theano_get_shape(x):
    return x.shape

def tensorflow_get_shape(x):
    shape = x._shape_as_list()
    assert shape.count(None) <= 1, '[Error!] not sure what to do with multiple None in tensor(flow) shape'
    shape = [-1 if s is None else s for s in shape]
    return shape

def dimshuffle_theano(x, shape):
    return x.dimshuffle(shape)

def dimshuffle_tensorflow(x, shape):
    # do not support degeneration of shape
    dims_to_permute = []
    dims_to_expand = []
    for i, s in enumerate(shape):
        if s == 'x':
            dims_to_expand.append(i)
        else:
            dims_to_permute.append(s)
    x = tf.transpose(x, dims_to_permute)
    for dim in dims_to_expand:
        x = tf.expand_dims(x, dim)
    return x

if K._backend == 'tensorflow':
    K.dimshuffle = dimshuffle_tensorflow
    K.reshape = tensorflow_reshape
    K.get_shape = tensorflow_get_shape
    K.diag = tensorflow_diag
else:
    K.dimshuffle = dimshuffle_theano
    K.reshape = theano_reshape
    K.get_shape = theano_get_shape
    K.diag = theano_diag

class L1L2Act(Regularizer):
    ''' reg normalized by number of samples '''

    def __init__(self, l1=0., l2=0.):
        self.l1 = K.cast_to_floatx(l1)
        self.l2 = K.cast_to_floatx(l2)

    def __call__(self, x):
        regularization = 0
        if self.l1:
            regularization += K.sum(self.l1 * K.mean(K.abs(x), axis=0))
        if self.l2:
            regularization += K.sum(self.l2 * K.mean(K.square(x), axis=0))
        return regularization

    def get_config(self):
        return {'name': self.__class__.__name__,
                'l1': float(self.l1),
                'l2': float(self.l2)}

def activity_l1(l=0.01):
    return L1L2Act(l1=l)

def activity_l2(l=0.01):
    return L1L2Act(l2=l)

def activity_l1l2(l1=0.01, l2=0.01):
    return L1L2Act(l1=l1, l2=l2)


################################
# Keras model related extensions
################################

def save_pred_result(pred_filename, pred_result):
    # save the prediction for test pairs
    # pred_result should be a list of tuples in the form of ('user', 'item', 'truth', 'pred')
    if pred_filename:
        pred_result_df = pd.DataFrame(pred_result, columns=['user', 'item', 'truth', 'pred'])
        pred_result_df.to_csv(pred_filename, index=False)

def save_model(model_filename, model):
    if model_filename:
        model.save_weights(model_filename)

def save_embeddings(emb_filename, get_embeddings, args):
    if emb_filename:
        emb_dict = get_embeddings(*args)
        with open(emb_filename, 'wb') as fp:
            pickle.dump(emb_dict, fp, 2)

def get_embeddings(model, conf, data_spec, C):
    # return a dictionary of important embeddings, such as user, item, word
    # make sure in model.layers, InteractionDot, user_embedding, word_embedding are named

    from keras.models import Sequential, Model

    interaction_layer = model.get_layer('InteractionDot')

    # get user embeddings
    user_io_idx = 0
    assert model.inputs[user_io_idx]._keras_shape[1] == 1, 'check if this is user dim'
    user_model = Model(input=model.inputs[user_io_idx], output=interaction_layer.input[user_io_idx])
    user_emb = user_model.predict_on_batch([np.array(range(data_spec.user_count))])
    if len(user_emb.shape) == 3:
        user_emb = user_emb.reshape((user_emb.shape[0], user_emb.shape[2]))
    # user_emb_norm = normalized_embedding(user_emb)

    # get content embeddings
    content_io_idx = 1
    assert model.inputs[content_io_idx]._keras_shape[1] > 1, 'check if this is content dim'
    content_model = Model(input=model.inputs[content_io_idx:], output=interaction_layer.input[content_io_idx])
    content_emb = content_model.predict_on_batch([np.array(range(C.shape[0]))])
    if len(content_emb.shape) == 3:
        content_emb = content_emb.reshape((content_emb.shape[0], content_emb.shape[2]))
    # content_emb_norm = normalized_embedding(content_emb)

    try:
        user_emb1 = model.get_layer('user_embedding').get_weights()[0]
    except:
        user_emb1 = None
    try:
        word_emb = model.get_layer('word_embedding').get_weights()[0]
    except:
        word_emb = None
    # user_emb_norm = normalized_embedding(user_emb)
    # word_emb_norm = normalized_embedding(word_emb)

    emb_dict = {'U': user_emb, 'V': content_emb, 'W': word_emb, 'U_1_just4check': user_emb1}
    return emb_dict

def pickle_dump(filename, data):
    with open(filename, 'w') as fp:
        pickle.dump(data, fp, 2)

def pickle_load(filename):
    with open(filename) as fp:
        return pickle.load(fp)