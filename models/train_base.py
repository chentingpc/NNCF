# neg_sharing (no additional neg sampling)
import numpy as np
from utilities import get_cur_time, pickle_dump
from objectives import Evaluator

class TrainerBase(object):
    def __init__(self, model_dict, conf, data_helper):
        self.model_dict = model_dict
        self.conf = conf
        self.data_helper = data_helper
        self.data_spec = data_helper.data_spec
        self.model_predict = model_dict['model_neg_shared']
        self.evaluater = Evaluator(data_helper, self.data_spec, conf)
        print '[INFO] Timestamps below are recorded at the end of training/evaluation respectively'

    def test(self, eval_scheme, predict_only=False, use_async_eval=False):
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
        if eval_scheme == 'given':
            result = self.evaluater.run(model_dict, eval_scheme=eval_scheme,
                                        predict_only=predict_only,
                                        use_async_eval=use_async_eval)
        elif eval_scheme == 'whole':
            result = self.evaluater.run(model_predict, eval_scheme=eval_scheme,
                                        predict_only=predict_only,
                                        use_async_eval=use_async_eval)
        return result

    def predict(self, eval_scheme, pred_saveto=None):
        result = self.test(eval_scheme, predict_only=True)
        if pred_saveto is not None:
            pickle_dump(pred_saveto, result)
        return result