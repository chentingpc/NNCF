import numpy as np
import keras.backend as K
from keras.layers import Layer, merge


class InteractionDNN(Layer):
    """ output response of two input tensors """
    def __init__(self, DNN, residule_layers=None, **kwargs):
        self.DNN = DNN
        self.residule_layers = residule_layers
        super(InteractionDNN, self).__init__(**kwargs)

    def get_output_shape_for(self, input_shape):
        return (None, 1)

    def call(self, x, mask=None):
        U, V = x[0], x[1]
        x = merge([U, V], mode='concat', dot_axes=[1, 1])
        response = x
        if self.residule_layers is not None:
            for i, layer in enumerate(self.residule_layers):
                response = layer(response) + x
        response = self.DNN(response)
        return response
