import numpy as np
import keras.backend as K
from keras.layers import Layer


class LinearLayer(Layer):
    """ linear regression score by using ids of user/item """
    def __init__(self, num_user, num_item, **kwargs):
        super(LinearLayer, self).__init__(**kwargs)
        self.b_u = K.variable(np.zeros((num_user, 1)), name='{}_b_u'.format(self.name))
        self.b_v = K.variable(np.zeros((num_item, 1)), name='{}_b_v'.format(self.name))
        self.b_g = K.variable(0.0, name='{}_b_g'.format(self.name))
        self.trainable_weights = [self.b_u, self.b_v, self.b_g]

    def get_output_shape_for(self, input_shape):
        return (None, 1)

    def call(self, x, mask=None):
        uid, vid = x[0], x[1]
        # regression = self.b_u[uid] + self.b_v[vid] + self.b_g
        regression = K.gather(self.b_u, uid) + K.gather(self.b_v, vid) + self.b_g
        regression = K.reshape(regression, (-1, 1))
        return regression
