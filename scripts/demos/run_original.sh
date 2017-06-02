#!/bin/bash
cd ../../

result_root_folder=./results
none_op='pretrain'  # set to pretrain to disable joint text embedding
eval_scheme=whole@50
max_epoch=30
gpu=0

for train_scheme in original ; do
for model_choice in cnn_embedding rnn_embedding; do
#for model_choice in basic_embedding cnn_embedding rnn_embedding; do
for data_name in citeulike_title_and_abstract ; do
#for data_name in citeulike_title_only citeulike_title_and_abstract ; do
for loss in skip-gram log-loss max-margin mse ; do
for chop_size in 0; do
for numneg in 10 ; do
for neg_dist in unigram; do
for neg_loss_weight in 128; do
for gamma in 10; do
for batch_size_p in 512; do
#for batch_size_p in 512 256 128 64; do
for learn_rate in 0.01 ; do
    if [ $loss == mse ]; then
        neg_loss_weight=8
        learn_rate=0.001
    fi
	if [ $loss == max-margin ]; then
		gamma=0.1
	fi
	if [ $model_choice == rnn_embedding ]; then
		numneg=5
	fi

    echo ******, $train_scheme, $model_choice, $data_name, $loss, $batch_size_p, $learn_rate, $neg_dist, $chop_size
    data_name_final=$data_name\_fold1
    result_folder=$result_root_folder/$data_name/$model_choice/$train_scheme/
	if [ ! -d $result_folder ]; then
		mkdir -p $result_folder
	fi
    log=$result_folder/log-eval$eval_scheme\-loss$loss\-numneg$numneg\-negw$neg_loss_weight\-gamma$gamma\-lr$learn_rate\-bsize$batch_size_p\-ndist$neg_dist\-chop$chop_size
    param_dict="{'reset_after_getconf': True, 'max_epoch': $max_epoch, 'loss': '$loss', 'num_negatives': $numneg, 'neg_loss_weight': $neg_loss_weight, 'loss_gamma': $gamma, 'learn_rate': $learn_rate, 'neg_dist': '$neg_dist', 'neg_sampling_power': 1, 'batch_size_p': $batch_size_p, 'chop_size': $chop_size, '$none_op': None}"

    stdbuf -oL -eL python main.py --data_name $data_name_final --model_choice $model_choice --conf_choice best --train_scheme $train_scheme --eval_scheme $eval_scheme --param_dict "$param_dict" --gpu $gpu> $log
done
done
done
done
done
done
done
done
done
done
done
