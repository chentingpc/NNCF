import os, sys
import numpy as np
import pandas as pd
import argparse

code_base = './'
for sub in ['', 'sampler', 'configs', 'models',
            'utils', 'modules', 'modules/interaction',
            'modules/content', 'modules/shared']:
    sys.path.insert(0, code_base + sub)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

parser = argparse.ArgumentParser()
parser.add_argument('--data_name', required=True)
parser.add_argument('--model_choice', required=True)
parser.add_argument('--conf_choice', required=True)
parser.add_argument('--train_scheme', default='original')
parser.add_argument('--eval_scheme', default='given')
parser.add_argument('--param_dict', default=None)
parser.add_argument('--pred_name', default=None)
parser.add_argument('--save_emb_name', default=None)
parser.add_argument('--gpu', default=None, type=str)
args_config = parser.parse_args()
data_name = args_config.data_name
model_choice = args_config.model_choice
conf_choice = args_config.conf_choice
train_scheme = args_config.train_scheme
eval_scheme = args_config.eval_scheme
if eval_scheme.lower() == "none":
    eval_scheme = None
pred_filename = args_config.pred_name
save_emb_filename = args_config.save_emb_name
if args_config.param_dict is None:
    param_dict = None
else:
    import ast
    param_dict = ast.literal_eval(args_config.param_dict)
if args_config.gpu is not None:
    os.environ['CUDA_VISIBLE_DEVICES'] = args_config.gpu
print 'model_choice: %s \nconf_choice: %s' % (model_choice, conf_choice)

# tensorflow memory configuration
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
set_session(tf.Session(config=config))

# load confs and related
if model_choice == 'pretrained':
    from pretrained_conf import get_conf
elif model_choice == 'mf':
    from pretrained_conf import get_conf
elif model_choice == 'basic_embedding':
    from basic_embedding_conf import get_conf
elif model_choice == 'cnn_embedding':
    from cnn_embedding_conf import get_conf
elif model_choice == 'rnn_embedding':
    from rnn_embedding_conf import get_conf
else:
    assert False, 'model choice %s not defined' % model_choice
conf = get_conf(data_name, conf_choice, param_dict)
# basic postprocessing
if eval_scheme is not None and eval_scheme.find('@') > 0:
    p = eval_scheme.find('@')
    conf.eval_topk = int(eval_scheme[p + 1:])
    eval_scheme = eval_scheme[:p]

# load data and related
from data_utils import get_data
data_helper = get_data(data_name, conf, reverse_samping=True)
data_spec = data_helper.data_spec

train = data_helper.data['train']
test_seen = data_helper.data['test_seen']
test = data_helper.data['test']
C = data_helper.data['C']
user_count = data_spec.user_count
item_count = data_spec.item_count

print conf.__dict__
#print data_spec.__dict__

# load model
from model_framework import get_model
model_dict = get_model(conf, data_helper, model_choice)

# start training
if train_scheme == 'original':
    from train_original import Trainer
elif train_scheme == 'reverse':
    from train_reverse import Trainer
elif train_scheme == 'presample':
    from train_presample import Trainer
elif train_scheme == 'neg_shared':
    from train_neg_shared import Trainer
elif train_scheme == 'group_neg_shared':
    from train_group_neg_shared import Trainer
elif train_scheme == 'group_sample':
    from train_group_sample import Trainer
elif train_scheme == 'sampled_neg_shared':
    from train_sampled_neg_shared import Trainer
else:
    assert False, '[ERROR] Unknown train_scheme {}'.format(train_scheme)

trainer = Trainer(model_dict, conf, data_helper, eval_scheme)
trainer.train(emb_saveto=save_emb_filename)
if pred_filename is not None:
    _ = trainer.predict(pred_filename)
if save_emb_filename is not None:
    _ = trainer.save_emb(save_emb_filename)

