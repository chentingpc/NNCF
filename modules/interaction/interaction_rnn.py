import numpy as np
import keras.backend as K
from keras.layers import Layer, merge, RepeatVector


class InteractionRNN(Layer):
    """ output response of two input tensors """
    def __init__(self, RNN, num_steps, DNN=None, **kwargs):
        self.RNN = RNN
        self.num_steps = num_steps
        self.DNN = DNN
        super(InteractionRNN, self).__init__(**kwargs)

    def get_output_shape_for(self, input_shape):
        return (None, 1)

    def call(self, x, mask=None):
        U, V = x[0], x[1]
        x = merge([U, V], mode='concat', dot_axes=[1, 1])
        x = RepeatVector(self.num_steps)(x)
        response = self.RNN(x)
        if self.DNN is not None:
            response = self.DNN(response)
        return response
