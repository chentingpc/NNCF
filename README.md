# NNCF

Introduction
----------------

This repository contains code for paper "On Sampling Strategies for Neural Network-based Collaborative Filtering", which propose (1) a general NNCF framework incorporates both interaction and content information, and (2) sampling strategies for speed up the process.


Model parameters
--------------

**Loss Functions**

1. pointwise: loss=skip-gram,  mse
2. pairwise: loss=log-loss, max-margin


**Content Embedding**

1. CNN: model_choice=cnn_embedding
2. RNN: model_choice=rnn_embedding
3. Mean of word vectors: model_choice=basic_embedding


**Interaction Module**

1. dot product


**Sampling strategies**

IID sampling: train_scheme=presample + shuffle_st=random

Negative sampling: train_scheme=original

Stratified sampling: train_scheme=group_sample + shuffle_st=item

Negative sharing: train_scheme=neg_shared

Stratified sampling with negative sharing: train_scheme=group_neg_shared


**Other parameters explained**

eval_scheme: whole@k for using all test items as candidates,  given@-1 for using true items + random sampled 10 items as candidates.

neg_loss_weight & gamma: adjustment in functions

chop_size: number of positive links per item in a batch. Only approximately when set group_shuffling_trick=True (which is recommended).


Setup and Run
--------------

1. unzip data; and go to ./code/sampler, and run ./make.sh
2. run using scripts under ./code/scripts/demos, which are prepared for each
   of the sampling strategies.
3. after running, the results are stored in ./results folder

Requirements
----------------

0. Unix system with python 2.7, GCC 4.8.x and GSL
1. Keras 1.2.2
2. Tensorflow 1.0

The code may or may not be working properly with other versions.

__Tips__

* GSL can be installed with ``sudo apt-get install libgsl-dev``
* Keras can be installed by first downloading zip file of version 1.2.2 and
  then installing with command ``python setup.py install``


Cite
-----------------

```
@inproceedings{chen2017onsampling,
	title={On Sampling Strategies for Neural Network-based Collaborative Filtering},
	author={Chen, Ting and Sun, Yizhou and Shi, Yue and Hong, Liangjie},
	booktitle={Proceedings of the 23th ACM SIGKDD international conference on Knowledge discovery and data mining},
	year={2017},
	organization={ACM}
}
```

