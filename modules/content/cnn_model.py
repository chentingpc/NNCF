import keras.backend as K
from keras.layers import Layer, Dense, Flatten, merge, Merge, Reshape, Dropout
from keras.layers import Embedding, Activation, MaxPooling1D, AveragePooling1D
from keras.layers import Convolution1D, BatchNormalization
from keras.regularizers import l1, l2
from utilities import activity_l2
from gatings import get_contextual_spatial_gated_input
from gatings import get_contextual_temporal_gated_input


class CNN(object):
    def __init__(self, data_spec, conf):
        word_emb_dropout_rate = conf.word_emb_dropout_rate
        num_filters = conf.num_filters
        filter_lengths = conf.filter_lengths
        poolings = conf.poolings
        pool_lengths = conf.pool_lengths
        conv_dropout_rate = conf.conv_dropout_rate
        conv_activation = conf.conv_activation
        conv_batch_normalization = conf.conv_batch_normalization
        item_dense_transform = conf.item_dense_transform

        # variables containing learnable parameters
        self.Emb_C = None
        self.Emb_Cid = None
        if isinstance(filter_lengths[0], list):
            self.CNN1D_1 = [[None] * len(filter_lengths[0])] * \
                len(filter_lengths)
        else:
            self.CNN1D_2 = [None] * len(filter_lengths)
        self.dense1 = None
        self.BatchNormalization1 = [None] * len(filter_lengths)
        self.BatchNormalization2 = None

        def model(x):
            ''' embedding => (gating) => CNN => dense '''
            content, cid = x[0], x[1]

            # word embedding
            if self.Emb_C is None:
                if data_spec.W_pretrain is None:
                    weights = None
                else:
                    weights = [data_spec.W_pretrain]
                self.Emb_C = Embedding(data_spec.word_count, conf.word_dim,
                                       input_length=data_spec.max_content_len,
                                       activity_regularizer=activity_l2(conf.c_reg),
                                       dropout=word_emb_dropout_rate,
                                       weights=weights,
                                       name='word_embedding')
            C_emb = self.Emb_C(content)

            # word embedding gating
            if conf.contextual_temporal_gated_input:
                C_emb = get_contextual_temporal_gated_input(C_emb, \
                    conf.contextual_temporal_gated_input)
            if conf.contextual_spatial_gated_input:
                C_emb = get_contextual_spatial_gated_input(C_emb, \
                    conf.contextual_spatial_gated_input)

            # convolution: support multiple layers 
            h = C_emb
            for l, fl in enumerate(filter_lengths):
                if isinstance(fl, list):
                    # multiple lengths of filters
                    hs = []
                    for ll, fl_ in enumerate(fl):
                        if self.CNN1D_1[l][ll] is None:
                            cnn = Convolution1D(nb_filter=num_filters[l],
                                                filter_length=fl_,
                                                border_mode="same",
                                                activation="linear",
                                                subsample_length=1,
                                                init='he_normal')
                            self.CNN1D_1[l][ll] = cnn
                        hs.append(self.CNN1D_1[l][ll](h))
                    h = Merge(mode='concat', concat_axis=-1)(hs)
                    assert h._keras_shape[1] == data_spec.max_content_len, \
                        h._keras_shape
                else:
                    # single length of filters
                    if self.CNN1D_2[l] is None:
                        cnn = Convolution1D(nb_filter=num_filters[l],
                                            filter_length=fl,
                                            border_mode="same",
                                            activation="linear",
                                            subsample_length=1,
                                            init='he_normal')
                        self.CNN1D_2[l] = cnn
                    h = self.CNN1D_2[l](h)
                if conv_batch_normalization:
                    try:
                        assert conf.no_BN == True
                    except:
                        if self.BatchNormalization1[l] is None:
                            self.BatchNormalization1[l] = BatchNormalization(mode=0)
                        h = self.BatchNormalization1[l](h)
                h = Activation(conv_activation)(h)
                h = Dropout(conv_dropout_rate)(h)

                # pooling
                if pool_lengths[l] <= 0:
                    pool_length = h._keras_shape[1]
                else:
                    pool_length = pool_lengths[l]
                if poolings[l] == 'average':
                    h = AveragePooling1D(pool_length=pool_length)(h)
                elif poolings[l] == 'max':
                    h = MaxPooling1D(pool_length=pool_length)(h)
                else:
                    assert False, '[ERROR] unknown pooling %s' % poolings[l]

            # dense layer
            if item_dense_transform:
                dense_h_dim = item_dense_transform['dense_hidden_dim']
                dense_h_actv = item_dense_transform['dense_hidden_actv']
                dense_h_dropout = item_dense_transform['dense_hidden_dropout']
                h = Flatten()(h)
                if self.dense1 is None:
                    self.dense1 = Dense(dense_h_dim)
                h = self.dense1(h)
                try:
                    assert conf.no_BN == True
                except:
                    if self.BatchNormalization2 is None:
                        self.BatchNormalization2 = BatchNormalization(mode=0)
                    h = self.BatchNormalization2(h)
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
