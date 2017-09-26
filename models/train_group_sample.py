# group-based sampling (by items)
import numpy as np
import time
from utilities import get_cur_time, nan_detection
from data_utils import group_shuffle_train, GroupSampler
from train_base import TrainerBase

class Trainer(TrainerBase):
    def __init__(self, model_dict, conf, data_helper, eval_scheme):
        super(Trainer, self).__init__(model_dict, conf, data_helper, eval_scheme)
        self.model_train = model_dict['model']
        self.neg_sign = np.array([-1], dtype='int32') \
            if conf.loss == 'skip-gram' else 0
        assert conf.loss not in ['log-loss', 'max-margin'], \
            "[ERROR] group_sample does not support pairwise losses"

        try: group_shuffling_trick = conf.group_shuffling_trick
        except: group_shuffling_trick = False
        self.group_shuffling_trick = group_shuffling_trick
        if group_shuffling_trick:
            if conf.neg_dist == 'uniform':
                print '[WARNING] group_shuffling_trick in group_sample' \
                      'does not fully support uniform neg_dist (no_correction).'
            self.sample_batch = data_helper.sampler_dict['sample_batch']
            self.sample_batch_u = data_helper.sampler_dict['sample_batch_u']
            _num_in_train = np.max(data_helper.data['train'], axis=0) + 1
            self._iidx = {'user': np.arange(_num_in_train[0]),
                          'item': np.arange(_num_in_train[1])}
        else:
            self.group_sampler = GroupSampler(data_helper.data['train'],
                                              group_by='item',
                                              chop=conf.chop_size,
                                              neg_dist=conf.neg_dist,
                                              neg_sign=self.neg_sign)
            self.group_sample_with_negs = self.group_sampler.sample_with_negs

    def train(self, use_async_eval=True, emb_saveto=None):
        model_train = self.model_train
        conf = self.conf
        data_helper = self.data_helper
        eval_scheme = self.eval_scheme
        num_negatives = conf.num_negatives
        batch_size_p = conf.batch_size_p
        train = data_helper.data['train']
        C = data_helper.data['C']
        group_shuffling_trick = self.group_shuffling_trick
        if group_shuffling_trick:
            # Support both user and item based sampling
            # Only when group_shuffling_trick is True
            shuffle_st = conf.shuffle_st
            if shuffle_st.startswith('by_user'):
                by = 'user'
            elif shuffle_st.startswith('by_item'):
                by = 'item'
            else:
                assert False
            print '[INFO] sampling group based on {}'.format(by)
        else:
            print '[INFO] sampling group based on item'

        train_time_stamp = time.time()  # more accurate measure of real train time.
        train_time_stamp0 = train_time_stamp
        train_time = []
        for epoch in range(conf.max_epoch + 1):
            if emb_saveto is not None:
                self.save_emb(emb_saveto + '-epoch{}'.format(epoch))
            bb, b = 0, batch_size_p
            cost, it = 0, 0
            if group_shuffling_trick:
                train = group_shuffle_train(train, by=by, \
                    chop=conf.chop_size, iidx=self._iidx[by])

            t_start = time.time()
            while epoch > 0 and bb < len(train):
                it += 1
                b = bb + batch_size_p
                if b > len(train):
                    # get rid of uneven tail so no need to dynamically adjust batch_size_p
                    break
                if group_shuffling_trick:
                    train_batch_p = train[bb: b]
                    train_batch_n = train_batch_p.repeat(num_negatives, axis=0)
                    if by == 'item':
                        train_batch_n[:, 0] = self.sample_batch_u( \
                            num_negatives * batch_size_p)
                    else:
                        train_batch[:, 1] = self.sample_batch( \
                            num_negatives * batch_size_p)
                    train_batch_n[:, 2] = self.neg_sign
                    train_batch = np.vstack((train_batch_p, train_batch_n))
                else:
                    train_batch = self.group_sample_with_negs(batch_size_p,
                                                              num_negatives)
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