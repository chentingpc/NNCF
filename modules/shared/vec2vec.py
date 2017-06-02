# this file contains common functions combining vectors to vectors
import numpy as np
import keras.backend as K
from keras.layers import Layer, Dense, Embedding, Merge, Reshape, Dropout
from keras.layers import Activation, BatchNormalization, AveragePooling1D
from keras.regularizers import l1, l2
from utilities import activity_l1, activity_l2


class ItemCombination(object):

    def __init__(self):
        self.Emb_C_pretrain = None
        self.Dense1 = None
        self.BatchNormalization1 = None
        
        def get_item_emb_combined_pretrain(C_emb, cid, conf, data_spec):
            # combine sup&unsup when C_pretrain is not None
            # consider sup as all zero is C_emb is None
            if data_spec.C_pretrain is not None:
                pretrain_dropout = conf.pretrain['pretrain_combine_dropout']
                pretrain_combine = conf.pretrain['pretrain_combine_mode']
                pretrain_actv = conf.pretrain['pretrain_combine_actv']
                output_dim = conf.user_dim
                item_count, item_dim = data_spec.C_pretrain.shape

                # placeholder for pretrained vectors
                if pretrain_dropout < 1:
                    if self.Emb_C_pretrain is None:
                        self.Emb_C_pretrain = Embedding(item_count, \
                            item_dim, weights=[data_spec.C_pretrain], \
                            dropout=pretrain_dropout, trainable=False)
                    C_emb_pretrain = self.Emb_C_pretrain(cid)
                    C_emb_pretrain = Reshape( \
                        (C_emb_pretrain._keras_shape[-1], ))(C_emb_pretrain)

                    # combining sup & unsup vectors
                    if C_emb is not None:
                        if len(C_emb._keras_shape) == 3:
                            C_emb = Reshape((C_emb._keras_shape[-1], ))(C_emb)
                        C_emb = Merge(mode=pretrain_combine)( \
                            [C_emb, C_emb_pretrain])
                    else:
                        C_emb = C_emb_pretrain
                else:
                    if len(C_emb._keras_shape) == 3:
                        C_emb = Reshape((C_emb._keras_shape[-1], ))(C_emb)

                if self.Dense1 is None:
                    self.Dense1 = Dense(output_dim)
                C_emb = self.Dense1(C_emb)
                conf.no_BN = True# debug
                try:
                    assert conf.no_BN == True
                except:
                    if self.BatchNormalization1 is None:
                        self.BatchNormalization1 = BatchNormalization()
                    C_emb = self.BatchNormalization1(C_emb)
                C_emb = Activation(pretrain_actv)(C_emb)
            else:
                if len(C_emb._keras_shape) == 3:
                    C_emb = Reshape((C_emb._keras_shape[-1], ))(C_emb)
            
            return C_emb

        self.get_item_emb_combined_pretrain = get_item_emb_combined_pretrain

    def get_model(self):
        return self.get_item_emb_combined_pretrain
