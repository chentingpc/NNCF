import keras.backend as K
from keras.layers import Layer, Embedding, Dense, Dropout, Reshape, Lambda
from keras.layers import Activation, Flatten, Merge, BatchNormalization
from keras.layers import LSTM, GRU, MaxPooling1D, AveragePooling1D
from keras.regularizers import l1, l2
from utilities import activity_l2
from gatings import get_contextual_spatial_gated_input
from gatings import get_contextual_temporal_gated_input


class RNN(object):
    def __init__(self, data_spec, conf):
        if conf.rnn.lower() == 'lstm':
            RNN = LSTM
        elif conf.rnn.lower() == 'gru':
            RNN = GRU
        else:
            assert False, 'ERROR! check conf.rnn %s' % conf.rnn

        # variables containing learnable parameters
        self.Emb_C = None
        self.Emb_Cid = None
        self.dense1 = None
        self.BatchNormalization1 = None
        self.RNN_op = [None] * (int(conf.bidirection) + 1) * \
            len(conf.lstm_dims)

        def model(x):
            ''' embedding => (gating) => RNN => pooling => dense '''
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
                                       dropout=conf.word_emb_dropout_rate,
                                       weights=weights,
                                       name='word_embedding')
            C_emb = self.Emb_C(content)  # (None, max_content_len, word_dim)

            # word gating
            if conf.contextual_temporal_gated_input:
                C_emb = get_contextual_temporal_gated_input(C_emb, \
                    conf.contextual_temporal_gated_input)
            if conf.contextual_spatial_gated_input:
                C_emb = get_contextual_spatial_gated_input(C_emb, \
                    conf.contextual_spatial_gated_input)

            # input re-org
            if conf.bidirection:
                x_embs = [C_emb, C_emb]
                backwards = [False, True]
            else:
                x_embs = [C_emb]
                backwards = [False]

            # LSTM layers: support multiple layers
            k = 0
            x_embs_results = []
            for x_emb, backward in zip(x_embs, backwards):
                for i in range(len(conf.lstm_dims)):
                    if conf.use_seq_for_dnn:
                        return_sequences = True
                    else:
                        if i == len(conf.lstm_dims) - 1:
                            return_sequences = False
                        else:
                            # non-last layer should return sequences
                            return_sequences = True
                    if self.RNN_op[k] is None:
                        rnn = RNN(conf.lstm_dims[i],
                                  dropout_W=conf.lstm_w_dropout_rate,
                                  dropout_U=conf.lstm_u_dropout_rate,
                                  return_sequences=return_sequences,
                                  go_backwards=backward,
                                  inner_activation='hard_sigmoid',
                                  activation='tanh',
                                  init='glorot_uniform',
                                  inner_init='orthogonal')
                        self.RNN_op[k] = rnn
                    x_emb = self.RNN_op[k](x_emb)
                    x_emb = Dropout(conf.lstm_o_dropout_rate)(x_emb)
                    k += 1
                x_embs_results.append(x_emb)
            if conf.bidirection:
                h = Merge(mode='concat', concat_axis=-1)(x_embs_results)
            else:
                h = x_embs_results[0]

            # pooling
            if conf.use_seq_for_dnn:
                seq_length = h._keras_shape[1]
                if conf.pooling == 'average':
                    h = AveragePooling1D(pool_length=seq_length)(h)
                elif conf.pooling == 'max':
                    h = MaxPooling1D(pool_length=seq_length)(h)
                else:
                    assert False, 'pooling %s not recognized.' % conf.pooling

            # hidden layers:
            if conf.item_dense_transform:
                dense_h_dim = conf.item_dense_transform['dense_hidden_dim']
                dense_h_dropout = conf.item_dense_transform['dense_hidden_dropout']
                dense_h_actv = conf.item_dense_transform['dense_hidden_actv']
                h = Flatten()(h)
                if self.dense1 is None:
                    self.dense1 = Dense(dense_h_dim)
                h = self.dense1(h)
                try:
                    assert conf.no_BN == True
                except:
                    if self.BatchNormalization1 is None:
                        self.BatchNormalization1 = BatchNormalization(mode=0)
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
