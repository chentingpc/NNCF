import keras.backend as K
from keras.layers import Layer, Dense, Embedding, Dropout
from keras.layers import Reshape, Activation, Flatten, merge, Merge
from keras.layers import MaxPooling1D, AveragePooling1D, BatchNormalization
from keras.regularizers import l1, l2
from utilities import activity_l2
from gatings import get_contextual_spatial_gated_input
from gatings import get_contextual_temporal_gated_input


class AverageEmbeddings(Layer):
    '''Average embeddings (ignore placehoder tokens with id 0).
    # Input shape
        3D tensor with shape: `(samples, steps, features)`.
        2D tensor with shape: `(samples, steps)`
    # Output shape
        3D tensor with shape: `(samples, 1, features)`.
    '''

    def __init__(self, **kwargs):
        super(AverageEmbeddings, self).__init__(**kwargs)

    def get_output_shape_for(self, input_shape):
        input_shape = input_shape[0]
        return (input_shape[0], 1, input_shape[2])

    def call(self, x, mask=None):
        U, content = x[0], x[1]
        #return K.mean(U, axis=1, keepdims=True)
        c = K.cast(K.not_equal(content, -1), 'float32')
        num_none_zeros = 1. / K.sum(c, axis=1, keepdims=True)
        num_none_zeros = K.reshape(num_none_zeros, (-1, 1, 1))
        return K.batch_dot(num_none_zeros, K.sum(U, axis=1, keepdims=True))


class MeanPool(object):
    def __init__(self, data_spec, conf):
        item_dense_transform = conf.item_dense_transform

        # variables containing learnable parameters
        self.Emb_C = None
        self.Emb_Cid = None
        self.Dense1 = None
        self.BatchNormalization1 = None

        def model(x):
            ''' word embedding => (gating) => pooling => dense '''
            content, cid = x[0], x[1]

            # get word embeddings
            if self.Emb_C is None:
                if data_spec.W_pretrain is None:
                    weights = None
                else:
                    weights = [data_spec.W_pretrain]
                self.Emb_C = Embedding(data_spec.word_count, conf.word_dim, \
                    input_length=data_spec.max_content_len, weights=weights, \
                    activity_regularizer=activity_l2(conf.c_reg), \
                    dropout=conf.word_emb_dropout_rate, name='word_embedding')
            C_emb = self.Emb_C(content)

            # word embedding gating
            if conf.contextual_temporal_gated_input:
                C_emb = get_contextual_temporal_gated_input(C_emb, \
                    conf.contextual_temporal_gated_input)
            if conf.contextual_spatial_gated_input:
                C_emb = get_contextual_spatial_gated_input(C_emb, \
                    conf.contextual_spatial_gated_input)

            # pooling into a single item embedding
            if conf.pooling == 'max':
                h = MaxPooling1D(data_spec.max_content_len)(C_emb)
            elif conf.pooling == 'average':
                # Keras pooling not consider placeholder tokens with id 0s 
                # h = AveragePooling1D(data_spec.max_content_len)(C_emb)
                h = AverageEmbeddings()([C_emb, content])
            else:
                assert False, '[ERROR] unknown pooling: %s' % conf.pooling

            # dense layer (with BN)
            if item_dense_transform:
                dense_h_dim = item_dense_transform['dense_hidden_dim']
                dense_h_actv = item_dense_transform['dense_hidden_actv']
                dense_h_dropout = item_dense_transform['dense_hidden_dropout']
                if len(h._keras_shape) == 3:
                    h = Reshape((h._keras_shape[-1], ))(h)
                if self.Dense1 is None:
                    self.Dense1 = Dense(dense_h_dim)
                h = self.Dense1(h)
                try:
                    assert conf.no_BN == True
                except:
                    if self.BatchNormalization1 is None:
                        self.BatchNormalization1 = BatchNormalization()
                    h = self.BatchNormalization1(h)
                h = Activation(dense_h_actv)(h)
                h = Dropout(dense_h_dropout)(h)
            else:
                if len(h._keras_shape) == 3:
                    h = Reshape((h._keras_shape[-1], ))(h)

            if conf.use_content_id:
                # content_id embedding is the same dim as user embedding
                if self.Emb_Cid is None:
                    self.Emb_Cid = Embedding(data_spec.item_count, conf.item_dim, \
                        activity_regularizer=activity_l2(conf.v_reg))
                Cid_emb = Reshape((conf.item_dim, ))(self.Emb_Cid(cid))
                h = Merge(mode='sum')([h, Cid_emb])

            return h

        self.model = model

    def get_model(self):
        return self.model
