''' this file consists of two parts: (1) loss definitions, and (2) evalutaions'''
import numpy as np
import pandas as pd
import keras.backend as K
import tensorflow as tf
from utilities import get_cur_time, safer_log
from sklearn.metrics import roc_auc_score, average_precision_score


############################
#     Loss definitions     #
############################


def _get_neg_loss_weight(conf):
    try:
        neg_loss_weight = conf.neg_loss_weight
        if neg_loss_weight is None: raise
    except:
        neg_loss_weight = 1
        print '[warning] loss weight for negative instances not defined,' \
            + ' use value {}'.format(neg_loss_weight)
    return np.array(neg_loss_weight, dtype=np.float32)


def _get_gamma(conf, verbose=True):
    try:
        gamma = conf.loss_gamma
    except:
        gamma = 1
        print '[warning] loss gamma not defined, {} is used.'.format(gamma)
    return np.array(gamma, dtype=np.float32)


def get_original_loss(loss, batch_size_p, num_negatives, conf):
    ''' For pairwise objectives,
            assume first batch_size_p are positive, 
            and the rest batch_size_p * num_negatives are negative
        For skip-gram, assuming y_true is -1 or 1
        For mse, assuming y_true is 0 or 1
    '''
    # for keras loss calculation, which takes mean of ouput loss, dummy
    output_len = (1 + num_negatives) * batch_size_p
    loss_base = np.ones((output_len, 1), dtype=np.float32)

    def ranking_loss(y_true, y_pred):
        pos_interact_score = y_pred[:batch_size_p, :]
        neg_interact_score = y_pred[batch_size_p:, :]
        if loss == 'max-margin':
            margin = _get_gamma(conf)
            diff_mat = K.reshape(K.repeat(pos_interact_score, num_negatives),
                                 (-1, 1)) - neg_interact_score
            task_loss = K.mean(K.relu(margin - diff_mat))
        elif loss == 'log-loss':
            sscaling = _get_gamma(conf)
            diff_mat = K.reshape(K.repeat(pos_interact_score, num_negatives),
                                 (-1, 1)) - neg_interact_score
            task_loss = K.mean(-safer_log(K.sigmoid(sscaling * diff_mat)))
        elif loss == 'skip-gram':
            neg_loss_weight_v = _get_neg_loss_weight(conf) / num_negatives
            neg_loss_weight = np.ones((output_len, 1)) + \
                (1 - y_true) / 2.0 * (neg_loss_weight_v - 1)
            task_loss = -neg_loss_weight * safer_log(K.sigmoid(y_true * y_pred))
            task_loss = K.sum(task_loss) / batch_size_p
        elif loss == 'mse':
            neg_loss_weight_v = _get_neg_loss_weight(conf) / num_negatives
            neg_loss_weight = np.ones((output_len, 1)) + \
                (1 - y_true) * (neg_loss_weight_v - 1)
            task_loss = neg_loss_weight * (y_true - y_pred)**2
            task_loss = K.sum(task_loss) / batch_size_p
        else:
            assert False, '[ERROR!] loss %s not specified.' % loss
        return task_loss * loss_base

    return ranking_loss


def get_neg_shared_loss(loss, batch_size_p, conf):
    ''' assuming y_pred is a predicted matrix where on-digonal are positives,
        off-diagonal are negatives
        note: pairwise loss calculation is biased by 1/b due to diaganol
        
        # diff_mat: diags are 0, off-diags are pos - neg
        # label_mat: 1 in on-diagnoals, -1 or 0 in off-diagonals
    '''
    # for keras loss calculation, which takes mean of ouput loss, dummy
    loss_base = np.ones((batch_size_p, 1), dtype=np.float32)

    def ranking_loss(y_true, y_pred):
        if loss == 'max-margin':
            margin = _get_gamma(conf)
            diff_mat = K.diag(y_pred, batch_size_p) - y_pred
            margin_mat = np.diag([-margin] * batch_size_p) + margin
            task_loss = K.mean(K.relu(margin_mat - diff_mat))
        elif loss == 'log-loss':
            sscaling = _get_gamma(conf)
            diff_mat = K.diag(y_pred, batch_size_p) - y_pred
            task_loss = K.mean(-safer_log(K.sigmoid(sscaling * diff_mat)))
        elif loss == 'skip-gram':
            neg_loss_weight_v = _get_neg_loss_weight(conf) / (batch_size_p - 1)
            neg_loss_weight = np.diag([1 - neg_loss_weight_v] * batch_size_p) + \
                neg_loss_weight_v
            label_mat = np.diag([2] * batch_size_p) - 1 
            task_loss = -neg_loss_weight * safer_log(K.sigmoid(label_mat * y_pred))
            task_loss = K.sum(task_loss) / batch_size_p
        elif loss == 'mse':
            neg_loss_weight_v = _get_neg_loss_weight(conf) / (batch_size_p - 1)
            neg_loss_weight = np.diag([1 - neg_loss_weight_v] * batch_size_p) + \
                neg_loss_weight_v
            label_mat = np.diag([1] * batch_size_p)
            task_loss = neg_loss_weight * (y_pred - label_mat)**2
            task_loss = K.sum(task_loss) / batch_size_p
        else:
            assert False, '[ERROR!] loss %s not specified.' % loss
        return task_loss * loss_base

    return ranking_loss


def get_sampled_neg_shared_loss(loss, batch_size_p, num_negatives, conf):
    ''' assuming first batch_size_p are positives,
        and next num_negatives are negatives
    '''
    # for keras loss calculation, which takes mean of ouput loss, dummy
    output_len = batch_size_p + num_negatives
    loss_base = np.ones((output_len, 1), dtype=np.float32)

    def ranking_loss(y_true, y_pred):
        if loss == 'max-margin':
            margin = _get_gamma(conf)
            diff_mat = K.reshape(y_pred[:, 0], (-1,1)) - y_pred[:, 1:]
            task_loss = K.mean(K.relu(margin - diff_mat))
        elif loss == 'log-loss':
            sscaling = _get_gamma(conf)
            diff_mat = K.reshape(y_pred[:, 0], (-1,1)) - y_pred[:, 1:]
            task_loss = K.mean(-safer_log(K.sigmoid(sscaling * diff_mat)))
        elif loss == 'skip-gram':
            neg_loss_weight_v = _get_neg_loss_weight(conf) / num_negatives
            neg_loss_weight = np.ones([batch_size_p, 1 + num_negatives], 
                                      dtype=np.float32)
            neg_loss_weight[:, 1:] = neg_loss_weight_v
            label_mat = np.ones((batch_size_p, 1 + num_negatives),
                                dtype=np.float32)
            label_mat[:, 1:] = -1
            task_loss = -neg_loss_weight * safer_log(K.sigmoid(label_mat * y_pred))
            task_loss = K.sum(task_loss) / batch_size_p
        elif loss == 'mse':
            neg_loss_weight_v = _get_neg_loss_weight(conf) / num_negatives
            neg_loss_weight = np.ones([batch_size_p, 1 + num_negatives],
                                      dtype=np.float32)
            neg_loss_weight[:, 1:] = neg_loss_weight_v
            label_mat = np.ones((batch_size_p, 1 + num_negatives), 
                                dtype=np.float32)
            label_mat[:, 1:] = 0
            task_loss = neg_loss_weight * (y_pred - label_mat)**2
            task_loss = K.sum(task_loss) / batch_size_p
        else:
            assert False, '[ERROR!] loss %s not specified.' % loss
        return task_loss * loss_base

    return ranking_loss

def get_group_neg_shared_loss(pred, pos_idxs, loss, batch_size_p, conf):
    ''' pred: (batch_size_p, variable_size)
        pos_idxs: (batch_size_p, 2)
    '''
    # for keras loss calculation, which takes mean of ouput loss, dummy
    loss_base = np.ones((batch_size_p, 1), dtype=np.float32)
    zero = tf.constant(np.zeros((batch_size_p, batch_size_p), 
                                dtype=np.float32))
    label_mat = tf.Variable(np.zeros((batch_size_p, batch_size_p),
                                     dtype=np.float32), name='label_mat')
    negw_mat = tf.Variable(np.zeros((batch_size_p, batch_size_p),
                                    dtype=np.float32), name='negw_mat')
    pred_pos = tf.reshape(tf.gather_nd(pred, pos_idxs), (-1, 1))
    pred_shape = tf.shape(pred)
    num_negatives = tf.cast(pred_shape[1] - 1, tf.float32)

    def create_mask(var, output_shape, pos_idxs, pos_val, neg_val, zero):
        # output_shape is a tensor of shape
        # zero is the zero constant with the shape shape as var
        var = tf.assign(var, zero)
        if isinstance(pos_val, int) or isinstance(pos_val, float):
            pos_val = np.array(pos_val, dtype=np.float32)
        if isinstance(neg_val, int) or isinstance(neg_val, float):
            neg_val = np.array(neg_val, dtype=np.float32)
        var_subset = tf.gather_nd(var, pos_idxs) + \
            pos_val - neg_val
        var = tf.scatter_nd_add(var, pos_idxs, var_subset)
        var += neg_val
        var = tf.slice(var, [0, 0], output_shape)
        return var

    if loss == 'max-margin':
        margin = _get_gamma(conf)
        diff_mat = pred_pos - pred
        task_loss = K.mean(K.relu(margin - diff_mat))
    elif loss == 'log-loss':
        sscaling = _get_gamma(conf)
        diff_mat = pred_pos - pred
        task_loss = K.mean(-safer_log(K.sigmoid(sscaling * diff_mat)))
    elif loss == 'skip-gram':
        neg_loss_weight_v = _get_neg_loss_weight(conf) / num_negatives
        label_mat = create_mask(label_mat, pred_shape, pos_idxs,
                                1, -1, zero)
        neg_loss_weight = create_mask(negw_mat, pred_shape, pos_idxs,
                                      1, neg_loss_weight_v, zero)
        task_loss = -neg_loss_weight * safer_log(K.sigmoid(label_mat * pred))
        task_loss = K.sum(task_loss) / batch_size_p
    elif loss == 'mse':
        neg_loss_weight_v = _get_neg_loss_weight(conf) / num_negatives
        label_mat = create_mask(label_mat, pred_shape, pos_idxs,
                                1, 0, zero)
        neg_loss_weight = create_mask(negw_mat, pred_shape, pos_idxs,
                                      1, neg_loss_weight_v, zero)
        task_loss = neg_loss_weight * (pred - label_mat)**2
        task_loss = K.sum(task_loss) / batch_size_p
    else:
        assert False, '[ERROR!] loss %s not specified.' % loss
    return task_loss * loss_base


############################
#        Evaluations       #
############################


from metrics_ranking import eval_multiple, eval_multiple_original, eval_apk


def test_eval(tests, topk, model_dict, user_count, item_count,
              content, per_user=False, return_result=False,
              predict_only=False):
    # evaluation for test performance
    # test is a list of (uid, truth, score)
    # content is the item-content matrix

    if not isinstance(tests, list):
        tests = [tests]
    for test in tests:
        user_emb = model_dict['model_user_emb'].predict_on_batch( \
            [np.array(range(user_count), dtype='int32')])
        item_emb = model_dict['model_item_emb'].predict_on_batch( \
            [np.array(range(item_count), dtype='int32')])
        pred = model_dict['model_pred_pairs'].predict( \
            [user_emb[test[:, 0]], item_emb[test[:, 1]], \
             test[:, 0], test[:, 1]], batch_size=4096)
        result = zip(test[:, 0], test[:, 2], pred[:, 0])  # (uid, truth, score)
        if predict_only:
            yield result
        elif return_result:
            result_save = zip(test[:, 0], test[:, 1], test[:, 2], pred[:, 0])
            yield evaluate(result, topk, per_user), result_save
        else:
            yield evaluate(result, topk, per_user)

def evaluate(result, topk, per_user=False, rounding=16):
    '''
    result is a list of three lists with ['uid', 'truth', 'pred'] columns

    rounding: rounding after decimal point
    '''
    if isinstance(result, pd.DataFrame):
        assert list(result.columns) == ['uid', 'truth', 'pred']
        result = result.groupby('uid')
    else:
        result = pd.DataFrame(result,
                              columns=['uid', 'truth', 'pred']).groupby('uid')
    map_k, recall_k, precision_k, auc = [], [], [], []
    result_per_user = {}
    for each in result:
        uid = each[1]['uid'].iloc[0]
        true_scores, pred_scores = each[1]['truth'], each[1]['pred']
        ap, recall, precision = eval_multiple_original( \
            true_scores, pred_scores, topk) if topk == -1 \
            else eval_multiple(true_scores, pred_scores, topk)
        _auc = roc_auc_score(true_scores, pred_scores)
        result_per_user[uid] = [ap, recall, precision, _auc]
        map_k.append(ap)
        recall_k.append(recall)
        precision_k.append(precision)
        auc.append(_auc)

    if per_user == True:
        return result_per_user, \
                {'map@%d' % topk: round(np.mean(map_k), rounding),
                'recall@%d' % topk: round(np.mean(recall_k), rounding),
                'precision@%d' % topk: round(np.mean(precision_k), rounding),
                'auc': round(np.mean(auc), rounding)}
    else:
        return {'map@%d' % topk: round(np.mean(map_k), rounding),
                'recall@%d' % topk: round(np.mean(recall_k), rounding),
                'precision@%d' % topk: round(np.mean(precision_k), rounding),
                'auc': round(np.mean(auc), rounding)}

def test_eval_mat(model_predict, eval_topk, true_mat, target_users,
                  target_items, content, predict_only=False, batch_size=1024):
    ''' assuming model_predict is neg_shared model,
        which computes complete interactions
    '''
    if not isinstance(target_users, np.ndarray):
        target_users = np.array(target_users, dtype='int32')
    if not isinstance(target_items, np.ndarray):
        target_items = np.array(target_items, dtype='int32')
    if batch_size > 0:
        pred_mat = []
        b = 0
        while b < target_items.shape[0]:
            target_items_b = target_items[b: b + batch_size]
            pm = model_predict.predict_on_batch([target_users, target_items_b])
            pred_mat.append(pm)
            b += batch_size
        pred_mat = np.hstack(pred_mat)
    else:
        pred_mat = model_predict.predict_on_batch([target_users, target_items])
    idx = np.sum(true_mat, axis=1) > 0
    if predict_only:
        return (true_mat[idx], pred_mat[idx])
    else:
        return evaluate_mat(true_mat[idx], pred_mat[idx],
                            eval_topk, compute_auc=False)

def evaluate_mat_thread(train_info, test_info, topk):
    truth_mat, pred_mat = train_info
    train_results = evaluate_mat(truth_mat, pred_mat, topk, compute_auc=False)
    truth_mat, pred_mat = test_info
    test_results = evaluate_mat(truth_mat, pred_mat, topk, compute_auc=False)
    print 'train recall/map', train_results['recall@%s' % topk], \
        train_results['map@%s' % topk], \
        'test recall/map', test_results['recall@%s' % topk], \
        test_results['map@%s' % topk]

def evaluate_mat(truth_mat, pred_mat, topk, per_user=False,
                 users_of_interests=None, compute_auc=True):
    '''
    each row of mat is a prediction for items of a user
    '''
    num_users = truth_mat.shape[0]
    if users_of_interests is None:
        users_of_interests = set(range(num_users))
    map_k, recall_k, precision_k, auc = [], [], [], []
    result_per_user = {}
    for uid in range(num_users):
        if uid not in users_of_interests:
            continue
        true_scores, pred_scores = truth_mat[uid, :], pred_mat[uid, :]
        ap, recall, precision = eval_multiple_original( \
            true_scores, pred_scores, topk) if topk == -1 \
            else eval_multiple(true_scores, pred_scores, topk)
        if compute_auc:
            _auc = roc_auc_score(true_scores, pred_scores)
        else:
            _auc = -1
        result_per_user[uid] = [ap, recall, precision, _auc]
        map_k.append(ap)
        recall_k.append(recall)
        precision_k.append(precision)
        auc.append(_auc)

    if per_user == True:
        return result_per_user, \
                {'map@%d' % topk: np.mean(map_k),
                'recall@%d' % topk: np.mean(recall_k),
                'precision@%d' % topk: np.mean(precision_k),
                'auc': np.mean(auc)}
    else:
        return {'map@%d' % topk: np.mean(map_k),
                'recall@%d' % topk: np.mean(recall_k),
                'precision@%d' % topk: np.mean(precision_k),
                'auc': np.mean(auc)}


class Evaluator(object):
    def __init__(self, data_helper, data_spec, conf, eval_scheme):
        self.data_helper = data_helper
        self.data_spec = data_spec
        self.conf = conf
        self.eval_scheme = eval_scheme
        if eval_scheme is not None:
            self._check_eval_sheme(eval_scheme)

        self.train = data_helper.data['train']
        self.test_seen = data_helper.data['test_seen']
        self.test = data_helper.data['test']
        self.C = data_helper.data['C']
        self.user_count = data_spec.user_count
        self.item_count = data_spec.item_count
        self.eval_topk = conf.eval_topk
        if eval_scheme == "whole":
            # Prepare only in case of whole evaluation, due to memory consumption.
            self._prepare_for_whole_eval()

    def _check_eval_sheme(self, eval_scheme):
        assert eval_scheme == 'given' or eval_scheme == 'whole', \
            '[Error] Unknown eval_scheme {}.'.format(eval_scheme)

    def _prepare_for_whole_eval(self):
        train = self.train
        test = self.test
        user_count = self.user_count

        train_items = sorted(set(train[:, 1]))
        test_items = sorted(set(test[:, 1]))
        train_items_dict, test_items_dict = {}, {}
        for i, item in enumerate(train_items):
            train_items_dict[item] = i
        for i, item in enumerate(test_items):
            test_items_dict[item] = i

        train_true_mat = np.zeros((user_count, len(train_items)),
                                  dtype='int32')
        test_true_mat = np.zeros((user_count, len(test_items)),
                                 dtype='int32')
        for user, item, _ in train:
            train_true_mat[user, train_items_dict[item]] = 1
        for each in test[test[:, 2] == 1]:
            user, item = each[0], each[1]
            test_true_mat[user, test_items_dict[item]] = 1

        self.train_items = train_items
        self.test_items = test_items
        self.train_true_mat = train_true_mat
        self.test_true_mat = test_true_mat

    def run(self, model, predict_only=False, verbose=True, 
            eval_scheme=None, batch_size=1024, use_async_eval=False):
        ''' predict_only: bool
                if True, return prediction; otherwise, evaluate normally
            eval_scheme: given or whole
                if given, evaluate with given test list
                if whole, predict for all items
            batch_size: int
                > 0 if want to use batch to predict
                <= 0 if want to use whole to predict at once
            use_async_eval: bool
                if True, will create new thread for running part of eval func,
                    so that GPU will not be left empty
        '''
        if eval_scheme is None:
            eval_scheme = self.eval_scheme
        self._check_eval_sheme(eval_scheme)
        C = self.C
        test_seen = self.test_seen
        test = self.test
        eval_topk = self.eval_topk
        user_count = self.user_count
        item_count = self.item_count

        if eval_scheme == 'given':
            model_dict = model
            test_eval_gen = test_eval([test_seen, test], eval_topk, \
                model_dict, user_count, item_count, C, \
                predict_only=predict_only)
            train_results = test_eval_gen.next()
            test_results = test_eval_gen.next()
            if verbose and not predict_only:
                print get_cur_time(), 'train map/auc', \
                    train_results['map@%s' % eval_topk], train_results['auc'], \
                    'test map/auc', test_results['map@%s' % eval_topk], test_results['auc']
            return train_results, test_results
        elif eval_scheme == 'whole':
            assert eval_topk > 0, \
                '[ERROR] eval_top {} must > 0'.format(eval_topk) + \
                'when eval_scheme=whole'
            train_true_mat = self.train_true_mat
            test_true_mat = self.test_true_mat
            train_items = self.train_items
            test_items = self.test_items

            model_predict = model
            predict_only_local = True \
                if predict_only or use_async_eval else False
            train_results = test_eval_mat(model_predict, eval_topk, \
                train_true_mat, range(user_count), train_items, C, \
                predict_only=predict_only_local, batch_size=batch_size)
            test_results = test_eval_mat(model_predict, eval_topk, \
                test_true_mat, range(user_count), test_items, C, \
                predict_only=predict_only_local, batch_size=batch_size)

            if verbose and not predict_only:
                if use_async_eval:
                    import threading
                    t = threading.Thread(target=evaluate_mat_thread,
                                         args=(train_results, test_results,
                                               eval_topk))
                    print get_cur_time(),
                    t.start()
                    train_results = test_results = t
                else:
                    print get_cur_time(), \
                        'train recall/map', train_results['recall@%s' % eval_topk], \
                            train_results['map@%s' % eval_topk], \
                        'test recall/map', test_results['recall@%s' % eval_topk], \
                            test_results['map@%s' % eval_topk]
            return train_results, test_results
