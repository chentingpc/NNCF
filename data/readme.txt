The essential data is "data_split_cold_item.pkl" file, which contains a dictionary with keys:

	['C', 'test_items', 'train_items', 'test', 'test_seen', 'train']

	C is the content matrix, each row is a sequence of text, with zero padding in the begining.
	train_items/test_items: list of items in the train/test. They should be non-overlapping.
	train: a matrix of positive interactions for train_items, each row is "user item score", where score is 1.
	test_seen: a random subset of train with sampled negative interactions for training monitoring.
	test: a matrix of positive interactions for test_items, with sampled negative interactions. 

The augmented pretrained text embedding is also provided, but it is not required.

Caveats:
1. neg_shared and group_neg_shared methods usually require "earlier stop" in order to avoid over-fitting, as they converge faster (thus over-fit faster).
2. Validation set is not included here, but the script for creating validation set is included. To do so, first copy data/ to data_val/, then run ``python create_val .`` in data_val/, it will replace pkl files in data_val with randomly generated validation data, finally edit data_utils.py file to redirect data folder to data_val.
