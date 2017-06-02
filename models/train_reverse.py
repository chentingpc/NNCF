# naively reverse original training (which changes the negative link distribution)
import numpy as np
import time
from collections import Counter
from utilities import get_cur_time, nan_detection
from train_base import TrainerBase

class Trainer(TrainerBase):
    def __init__(self, model_dict, conf, data_helper):
        super(Trainer, self).__init__(model_dict, conf, data_helper)
        self.model_train = model_dict['model']
        self.neg_sign = np.array([-1], dtype='int32') \
            if conf.loss == 'skip-gram' else 0
        assert conf.loss not in ['log-loss', 'max-margin'], \
            "[ERROR] revrese does not support pairwise losses"
        self.sample_batch_u = data_helper.sampler_dict['sample_batch_u']

        def c_func(train, user_count=None, as_array=True):
            # user u -> cu[u]
            cu = Counter(train[:, 0])
            if user_count is None:
                user_count = np.max(train[:, 0]) + 1
            for u in cu:
                cu[u] = 1 + 10 * np.log(1 + cu[u] * 1e4)
            if as_array:
                cu_array = np.ones(user_count)
                for u, v in cu.iteritems():
                    cu_array[u] = v
                return cu_array
            else:
                return cu
        cu = c_func(data_helper.data['train'], self.data_spec.user_count)
        self.cu = cu

    def train(self, eval_scheme=None, use_async_eval=True):
        model_train = self.model_train
        neg_sign = self.neg_sign
        conf = self.conf
        data_helper = self.data_helper
        num_negatives = conf.num_negatives
        batch_size_p = conf.batch_size_p
        sample_batch_u = self.sample_batch_u
        train = data_helper.data['train']
        C = data_helper.data['C']

        train_time = []
        for epoch in range(conf.max_epoch + 1):
            bb, b = 0, batch_size_p
            cost, it = 0, 0
            np.random.shuffle(train)

            t_start = time.time()
            while epoch > 0 and bb < len(train):
                it += 1
                b = bb + batch_size_p
                if b > len(train):
                    # get rid of uneven tail so no need to dynamically adjust batch_size_p
                    break
                train_batch_p = train[bb: b]
                train_batch_n = train_batch_p.repeat(num_negatives, axis=0)
                train_batch_n[:, 0] = sample_batch_u(num_negatives * batch_size_p)
                train_batch_n[:, 2] = neg_sign
                train_batch = np.vstack((train_batch_p, train_batch_n))
                user_batch = train_batch[:, 0]
                item_batch = train_batch[:, 1]
                response_batch = train_batch[:, 2]
                cost += model_train.train_on_batch([user_batch, item_batch],
                                                   [response_batch])
                bb = b
            if epoch > 0:
                train_time.append(time.time() - t_start)
            print get_cur_time(), 'epoch %d (%d it)' % (epoch, it), \
                'cost %.5f' % (cost / it if it > 0 else -1),
            nan_detection('cost', cost)
            if eval_scheme is None:
                print ''
            else:
                async_eval = True \
                    if use_async_eval and epoch != conf.max_epoch else False
                try: ps[-1].join()
                except: pass
                ps = self.test(eval_scheme, use_async_eval=async_eval)
        print 'Training time (sec) per epoch:', np.mean(train_time)
