#!/bin/bash


dataset=blogcatalog
#dataset=flickr
#dataset=youtube
result_folder=/storage/NNCF/results/
mat_file=/home/chentingpc/dbase/network/network_embedding/original/$dataset.mat
for method in group_sample group_neg_shared neg_shared original ; do
#for method in group_sample group_neg_shared neg_shared original presample; do
for epoch in {0..30} final; do
	if [ $epoch == final ]; then
		emb_file=$result_folder/network-$dataset/mf/$method/emb
	else
		emb_file=$result_folder/network-$dataset/mf/$method/emb-epoch$epoch
	fi
	while [ ! -e $emb_file ]; do
		sleep 5
		echo waiting for $emb_file
	done
	log_file=$result_folder/network-$dataset/mf/$method/log-classification
	echo ============== $emb_file ================ >> $log_file
	python scoring.py $emb_file $mat_file "sampling" >>$log_file
done
done


<<comment
# For LINE
emb_file=/home/chentingpc/cbase/dl/network_embedding/LINE/linux/result/vec_all.txt
#emb_file=/home/chentingpc/cbase/dl/network_embedding/LINE/linux/result/vec_1st.txt
#emb_file=/home/chentingpc/cbase/dl/network_embedding/LINE/linux/result/vec_2nd.txt
#emb_file=/home/chentingpc/cbase/dl/network_embedding/LINE/linux/result/vec_2nd_wo_norm.txt
#emb_file=/home/chentingpc/cbase/dl/network_embedding/LINE/linux/result/vec_1st_wo_norm.txt
mat_file=/home/chentingpc/dbase/network/network_embedding/original/blogcatalog.mat
mat_file=/home/chentingpc/dbase/network/network_embedding/original/youtube.mat
python scoring.py $emb_file $mat_file "line"
comment
exit


