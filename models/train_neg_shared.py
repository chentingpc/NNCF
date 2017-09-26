# neg_sharing (no additional neg sampling)
import random
import numpy as np
import time
from utilities import get_cur_time, nan_detection
from train_base import TrainerBase

class Trainer(TrainerBase):
    def __init__(self, model_dict, conf, data_helper, eval_scheme):
        super(Trainer, self).__init__(model_dict, conf, data_helper, eval_scheme)
        self.model_train = model_dict['model_neg_shared']
        self.model_all_loss = model_dict['model_neg_shared_all_loss']
        try:
            self.monitor_rate = conf.monitor_rate_all_loss
        except:
            self.monitor_rate = 0  # set this > 0 if want to monitor other losses
        
        if conf.neg_dist != 'unigram':
            print '[WARNING] Only unigram neg_dist is currently supported ' \
                'for group_neg_shared training. Set neg_dist = unigram.'

    def train(self, use_async_eval=True, emb_saveto=None):
        model_train = self.model_train
        model_all_loss = self.model_all_loss
        conf = self.conf
        data_helper = self.data_helper
        eval_scheme = self.eval_scheme
        train = data_helper.data['train']
        C = data_helper.data['C']

        train_time_stamp = time.time()  # more accurate measure of real train time.
        train_time_stamp0 = train_time_stamp    
        train_time = []
        for epoch in range(conf.max_epoch + 1):
            if emb_saveto is not None:
                self.save_emb(emb_saveto + '-epoch{}'.format(epoch))
            bb, b = 0, conf.batch_size_p
            cost, it = 0, 0
            monitor_losses = model_all_loss.keys()
            costs = dict(zip(monitor_losses, np.zeros(len(monitor_losses))))
            monitor_rate, monitor_it = self.monitor_rate, 0
            np.random.shuffle(train)

            t_start = time.time()
            while epoch > 0 and bb < len(train):
                it += 1
                b = bb + conf.batch_size_p
                if b > len(train):
                    # get rid of uneven tail so no need to dynamically adjust batch_size_p
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
                if random.random() < monitor_rate:
                    monitor_it += 1
                    for lname, m in model_all_loss.iteritems():
                        costs[lname] += m.train_on_batch([user_batch,
                                                          item_batch],
                                                         [response_batch])
                bb = b
            if epoch > 0:
                train_time.append(time.time() - t_start)
            print get_cur_time(), 'stamp %.2f epoch %d (%d it)' % ( \
                train_time_stamp - train_time_stamp0, epoch, it), \
                'cost %.5f' % (cost / it if it > 0 else -1),
            nan_detection('cost', cost)
            if monitor_rate > 0:
                print '(',
                for loss in monitor_losses:
                    print '{}={}'.format(loss, costs[loss] / monitor_it 
                                         if monitor_it > 0 else -1),
                print ')',
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
