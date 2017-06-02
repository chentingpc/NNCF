import numpy as np
import tensorflow as tf
import keras.backend as K
from keras.layers import Layer, Embedding, Lambda, Merge, Reshape
from keras.layers import Dense, Activation, Flatten, initializations
from keras.regularizers import l1, l2
from utilities import activity_l2


def normalize_shape(x, final_dim=3):
    # normalize x (None, dim) to be (None, 1, dim) if final_dim=3
    # normalize x (None, 1, dim) to be (None, dim) if final_dim=2
    x_shape = x._keras_shape
    if final_dim == 3:
        if len(x_shape) == 2:
            return Reshape((1, x_shape[1]))(x)
    elif final_dim == 2:
        if len(x_shape) == 3:
            return Reshape((x_shape[-1],))(x)
    else:
        assert False
    return x


class InteractionDot(Layer):
    def __init__(self, form="mul", bias=None, name="InteractionDot", 
                 user_count=None, item_count=None, **kwargs):
        ''' form: str
                mul: element-wise matrix multiplication
                matmul: matrix mutiplication
            bias: str or None
                user, item, both
        '''
        self.form = form
        self.bias = bias
        self.user_count = user_count
        self.item_count = item_count

        self.init = 'zero'
        self.ubias_regularizer = None
        self.ubias_constraint = None
        self.cbias_regularizer = None
        self.cbias_constraint = None

        assert form in ['mul', 'matmul'], \
                "ERROR! Unknown interation form {}".format(form)
        assert bias in ['user', 'item', 'both', None], \
                "ERROR! Unknown interation bias {}".format(form)
        super(InteractionDot, self).__init__(name=name, **kwargs)

    def build(self, input_shape):
        bias = self.bias
        user_count = self.user_count
        item_count = self.item_count
        init = self.init
        ubias_regularizer = self.ubias_regularizer
        ubias_constraint = self.ubias_constraint
        cbias_regularizer = self.cbias_regularizer
        cbias_constraint = self.cbias_constraint

        init = initializations.get(init)
        if bias == 'user' or bias == 'both':
            self.ubias = self.add_weight((user_count, 1), initializer=init,
                                         name='{}_ubias'.format(self.name),
                                         regularizer=ubias_regularizer,
                                         constraint=ubias_constraint)
        if bias == 'item' or bias == 'both':
            self.cbias = self.add_weight((item_count, 1), initializer=init,
                                         name='{}_cbias'.format(self.name),
                                         regularizer=cbias_regularizer,
                                         constraint=cbias_constraint)
        self.built = True

    def get_output_shape_for(self, input_shape):
        # [(None, 1, u_dim), (None, 1, v_dim)] or
        # [(None, u_dim), (None, v_dim)]
        if self.form == "mul":
            return (None, 1)
        else:
            return (None, None)

    def call(self, x, mask=None):
        U, V = x[0], x[1]
        uid, cid = None, None
        if self.bias == 'both':
            uid, cid = x[2], x[3]
        elif self.bias == 'user':
            uid = x[2]
        elif self.bias == 'item':
            cid = x[3]

        if self.form == 'mul':
            U = normalize_shape(U, final_dim=2)
            V = normalize_shape(V, final_dim=2)
            R = tf.reduce_sum(U * V, axis=1, keep_dims=True)
            if uid is not None:
                R += tf.gather_nd(self.ubias, uid)
            if cid is not None:
                R += tf.gather_nd(self.cbias, cid)
        elif self.form == 'matmul':
            U = normalize_shape(U, final_dim=2)
            V = normalize_shape(V, final_dim=2)
            R = tf.matmul(U, tf.transpose(V))
            if uid is not None:
                R += tf.gather_nd(self.ubias, uid)
            if cid is not None:
                R += tf.transpose(tf.gather_nd(self.cbias, cid))

        return R

    def set_form(self, form):
        self.form = form
        return self
