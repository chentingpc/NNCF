# neg_sharing (no additional neg sampling)
import numpy as np
import cPickle as pickle
from utilities import get_cur_time, pickle_dump
from objectives import Evaluator

class TrainerBase(object):
    def __init__(self, model_dict, conf, data_helper, eval_scheme):
        self.model_dict = model_dict
        self.conf = conf
        self.data_helper = data_helper
        self.eval_scheme = eval_scheme
        self.data_spec = data_helper.data_spec
        self.model_predict = model_dict['model_neg_shared']
        self.evaluater = Evaluator(data_helper, self.data_spec, conf, eval_scheme)
        print '[INFO] Timestamps below are recorded at the end of training/evaluation respectively'

    def test(self, predict_only=False, use_async_eval=False):
        ''' predict_only: bool
                if True, only return truth/prediction; else evaluate normally
            use_async_eval: bool
                if Ture, will run some eval funcs with a new thread,
                which saves time, but it may need to small print layout issue
                (train/test result being put after next iteration train cost)
                if training is finished faster than test
        '''
        model_dict = self.model_dict
        model_predict = self.model_predict
        eval_scheme = self.eval_scheme
        if eval_scheme == 'given':
            result = self.evaluater.run(model_dict, eval_scheme=eval_scheme,
                                        predict_only=predict_only,
                                        use_async_eval=use_async_eval)
        elif eval_scheme == 'whole':
            result = self.evaluater.run(model_predict, eval_scheme=eval_scheme,
                                        predict_only=predict_only,
                                        use_async_eval=use_async_eval)
        return result

    def predict(self, pred_saveto=None):
        result = self.test(self.eval_scheme, predict_only=True)
        if pred_saveto is not None:
            pickle_dump(pred_saveto, result)
        return result

    def save_emb(self, emb_saveto=None, pkl_format=True):
        weights = self.model_train.get_weights()
        names = [weight.name for weight in self.model_train.weights]
        main_emb = None
        context_emb = None
        for name, weight in zip(names, weights):
            if name.startswith("item_embedding"):
                if context_emb is not None:
                    raise ValueError("multiple item_embedding {}".format(name))
                context_emb = weight
            elif name.startswith("user_embedding"):
                if main_emb is not None:
                    raise ValueError("multiple user_embedding {}".format(name))
                main_emb = weight
        if main_emb is None or context_emb is None:
            raise ValueError("both main_emb {} and context_emb {} shouldn't be None".format(
                main_emb, context_emb))
        # Concat main and context embeddings.
        # emb = np.hstack((main_emb, context_emb))
        emb = main_emb

        # Save to txt format.
        if pkl_format:
            with open(emb_saveto, "wb") as fp:
                pickle.dump(emb, fp, 2)
        else:
            np.savetxt(emb_saveto, emb)
