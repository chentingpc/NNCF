# neg_sharing (no additional neg sampling)
import numpy as np
import time
from utilities import get_cur_time, nan_detection
from data_utils import group_shuffle_train
from train_base import TrainerBase

class Trainer(TrainerBase):
    def __init__(self, model_dict, conf, data_helper, eval_scheme):
        super(Trainer, self).__init__(model_dict, conf, data_helper, eval_scheme)
        self.model_train = model_dict['model']
        self.neg_sign = np.array([-1], dtype='int32') \
            if conf.loss == 'skip-gram' else 0
        self.sample_batch = data_helper.sampler_dict['sample_batch']
        self.sample_batch_u = data_helper.sampler_dict['sample_batch_u']
        _num_in_train = np.max(data_helper.data['train'], axis=0) + 1
        self._iidx = {'user': np.arange(_num_in_train[0]),
                      'item': np.arange(_num_in_train[1])}
                      
        if conf.shuffle_st == 'reverse' or \
                conf.shuffle_st.startswith('by_item'):
            assert conf.loss not in ['log-loss', 'max-margin'], \
                "[ERROR] shuffle_st %s does not support pairwise losses" % \
                conf.shuffle_st

    def train(self, use_async_eval=True, emb_saveto=None):
        model_train = self.model_train
        neg_sign = self.neg_sign
        conf = self.conf
        num_negatives = conf.num_negatives
        batch_size = conf.batch_size_p * (1 + num_negatives)
        data_helper = self.data_helper
        eval_scheme = self.eval_scheme
        sample_batch = self.sample_batch
        sample_batch_u = self.sample_batch_u
        train_p = data_helper.data['train']
        C = data_helper.data['C']
        chop = conf.chop_size

        train_time_stamp = time.time()  # more accurate measure of real train time.
        train_time_stamp0 = train_time_stamp
        train_time = []
        for epoch in range(conf.max_epoch + 1):
            if emb_saveto is not None:
                self.save_emb(emb_saveto + '-epoch{}'.format(epoch))
            bb, b = 0, batch_size
            cost, it = 0, 0
            # pre-sample
            if conf.shuffle_st == 'original':
                # sampling as in original training scheme (given link, sample neg users)
                np.random.shuffle(train_p)
                train = train_p.repeat(1 + num_negatives, axis=0)
                train[:, 1] = sample_batch(train.shape[0])
                train[:, 2] = neg_sign
                train[::1 + num_negatives] = train_p
            elif conf.shuffle_st == 'reverse':
                np.random.shuffle(train_p)
                # sampling as in reverse training scheme (given link, sample neg users)
                train = train_p.repeat(1 + num_negatives, axis=0)
                train[:, 0] = sample_batch_u(train.shape[0])
                train[:, 2] = neg_sign
                train[::1 + num_negatives] = train_p
            else:  # random sampling and/or groupping
                train_n = train_p.repeat(num_negatives, axis=0)
                train_n[:, 1] = sample_batch(train_n.shape[0])
                train_n[:, 2] = neg_sign
                train = np.vstack((train_p, train_n))
                if conf.shuffle_st == 'random':
                    np.random.shuffle(train)
                elif conf.shuffle_st == 'by_user':
                    train = group_shuffle_train(train, by='user',
                                               iidx=self._iidx['user'])
                elif conf.shuffle_st == 'by_item':
                    train = group_shuffle_train(train, by='item',
                                               iidx=self._iidx['item'])
                elif conf.shuffle_st.startswith('by_user_chop'):
                    train = group_shuffle_train(train, by='user', chop=chop, 
                                               iidx=self._iidx['user'])
                elif conf.shuffle_st.startswith('by_item_chop'):
                    train = group_shuffle_train(train, by='item', chop=chop,
                                               iidx=self._iidx['item'])
                elif conf.shuffle_st.startswith('by_useritem_chop'):
                    if epoch % 2 == 0:
                        train = group_shuffle_train(train, by='user', chop=chop,
                                                   iidx=self._iidx['user'])
                    else:
                        train = group_shuffle_train(train, by='item', chop=chop,
                                                   iidx=self._iidx['item'])
                else:
                    assert False, 'ERROR: unknown shuffle strategy {}'.format(conf.shuffle_st)

            t_start = time.time()
            while epoch > 0 and bb < len(train):
                it += 1
                b = bb + batch_size
                if b > len(train):
                    # get rid of uneven tail, otherwise need to dynamically adjust batch_size
                    break
                train_batch = train[bb: b]
                user_batch = train_batch[:, 0]
                item_batch = train_batch[:, 1]
                response_batch = train_batch[:, 2]
                right_before_train = time.time()
                cost += model_train.train_on_batch([user_batch, item_batch],
                                                   [response_batch])
                right_after_train = time.time()
                train_time_stamp += right_after_train - right_before_train
                bb = b
            if epoch > 0:
                train_time.append(time.time() - t_start)
            print get_cur_time(), 'stamp %.2f epoch %d (%d it)' % ( \
                train_time_stamp - train_time_stamp0, epoch, it), \
                'cost %.5f' % (cost / it if it > 0 else -1),
            nan_detection('cost', cost)
            if eval_scheme is None:
                print ''
            else:
                async_eval = True \
                    if use_async_eval and epoch != conf.max_epoch else False
                try: ps[-1].join()
                except: pass
                ps = self.test(use_async_eval=async_eval)
        print 'Training time (sec) per epoch: {}, or {} (more accurate).'.format(
            np.mean(train_time), (train_time_stamp - train_time_stamp0) / conf.max_epoch)
