import numpy as np
import keras.backend as K
from keras.layers import Layer, Dense, Lambda, Merge, Reshape, Dropout
from keras.layers import Activation, BatchNormalization, AveragePooling1D
from keras.regularizers import l1, l2
from utilities import activity_l2


class DotMergeAdhoc(Layer):
    # an ad-hoc implementation of merge by element-wise dot with broadcast
    def __init__(self, scale=False, **kwargs):
        super(DotMergeAdhoc, self).__init__(**kwargs)
        self.scale = scale
        if self.scale:
            self.c = K.ones((1,), name='{}_c'.format('DotMergeAdhoc'))
            self.trainable_weights.append(self.c)

    def get_output_shape_for(self, input_shape):
        # [(None, step, v_dim), (None, 1, u_dim)]
        return (None, input_shape[0][1], input_shape[0][2])
    
    def call(self, x, mask=None):
        if self.scale:
            return x[0] * x[1] * self.c
        else:
            return x[0] * x[1]


class DotSumMergeAdhoc(Layer):
    # an ad-hoc implementation of merge by element-wise dot and then sum, with broadcast
    def __init__(self, **kwargs):
        super(DotSumMergeAdhoc, self).__init__(**kwargs)

    def get_output_shape_for(self, input_shape):
        # [(None, step, v_dim), (None, 1, v_dim)]
        return (None, input_shape[0][1], 1)
    
    def call(self, x, mask=None):
        return K.sum(x[0] * x[1], axis=-1, keepdims=True)


class ReshapeBatchAdhoc(Layer):
    # similar to Reshape, but merge first two dimension (i.e. size in the batch)
    def __init__(self, mid_dim=None, **kwargs):
        self.mid_dim = mid_dim
        super(ReshapeBatchAdhoc, self).__init__(**kwargs)

    def get_output_shape_for(self, input_shape):
        # (None, step, v_dim) => (None, v_dim), if mid_dim == None
        # (None, v_dim) => (NOne, mid_dim, v_dim), if mid_dim != None
        if self.mid_dim:
            return (None, self.mid_dim, input_shape[-1])
        else:
            return (None, input_shape[-1])
    
    def call(self, x, mask=None):
        if self.mid_dim:
            return K.reshape(x, (-1, self.mid_dim, x._keras_shape[-1]))
        else:
            return K.reshape(x, (-1, x._keras_shape[-1]))


def get_contextual_spatial_gated_input(X, conf_dict):
    # X: input to be gated, (None, steps, x_dim)
    # return X' = X * sigmoid(Dense(Average(f(X)))), f is a non-linear function.
    assert len(X._keras_shape) == 3, [X._keras_shape]
    seq_len, x_dim = X._keras_shape[1], X._keras_shape[2]
    gating_hidden_dim = conf_dict['gating_hidden_dim']
    gating_hidden_actv = conf_dict['gating_hidden_actv']

    Xp = ReshapeBatchAdhoc()(X)
    Xp = Dense(gating_hidden_dim, activation=gating_hidden_actv)(Xp)
    #Xp = Lambda(lambda x: x * 0)(Xp)
    Xp = ReshapeBatchAdhoc(mid_dim=seq_len)(Xp)
    Xp = AveragePooling1D(seq_len)(Xp)  # (None, 1, x_dim)
    Xp = Reshape((Xp._keras_shape[-1], ))(Xp)
    Xp = Dense(x_dim, activation='sigmoid')(Xp)
    Xp = Reshape((1, x_dim))(Xp)
    X = DotMergeAdhoc()([X, Xp])
    return X


def get_contextual_temporal_gated_input(X, conf_dict):
    # X: input to be gated, (None, steps, x_dim)
    # return X' = X * c * softmax(X.Average(f(X))), f is a non-linear function.
    assert len(X._keras_shape) == 3, [X._keras_shape]
    seq_len, x_dim = X._keras_shape[1], X._keras_shape[2]
    gating_hidden_dim = conf_dict['gating_hidden_dim']
    gating_hidden_actv = conf_dict['gating_hidden_actv']
    scale = conf_dict['scale']
    nl_choice = conf_dict['nl_choice']

    Xp = ReshapeBatchAdhoc()(X)
    Xp = Dense(gating_hidden_dim, activation=gating_hidden_actv)(Xp)
    Xp = ReshapeBatchAdhoc(mid_dim=seq_len)(Xp)
    Xp = AveragePooling1D(seq_len)(Xp)  # (None, 1, x_dim)
    Xp = Reshape((Xp._keras_shape[-1], ))(Xp)
    if nl_choice == 'nl':
        Xp = Dense(x_dim, activation='relu', bias=True)(Xp)
    elif nl_choice == 'bn+nl':
        Xp = BatchNormalization()(Xp)
        Xp = Dense(x_dim, activation='relu', bias=True)(Xp)
    elif nl_choice == 'bn+l':
        Xp = BatchNormalization()(Xp)
        Xp = Dense(x_dim, activation='linear', bias=True)(Xp)
    else:
        assert False, 'nonononon'
    Xp = Reshape((1, x_dim))(Xp)  # (None, 1, x_dim)
    Xp = DotSumMergeAdhoc()([X, Xp])  # (None, steps, 1)
    if True:  # debug
        Xp = Activation('sigmoid')(Xp)  # (None, steps, 1)
    else:
        # following can be uncomment to replace sigmoid with softmax
        Xp = Reshape((Xp._keras_shape[1], ))(Xp)  # (None, steps)
        Xp = Activation('softmax')(Xp)  # (None, steps)
        Xp = Reshape((Xp._keras_shape[-1], 1))(Xp)  # (None, steps, 1)
    X = DotMergeAdhoc(scale=scale)([X, Xp]) # (None, steps, x_dim)
    return X
