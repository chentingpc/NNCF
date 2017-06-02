#!/bin/bash

# this script help run things in batch
echo WARNING: ACCESSING SPEED, CANNOT RUN PARALLEL IN ONE GPU
echo ./run_presample_random.sh
./run_presample_random.sh
echo ./run_original.sh
./run_original.sh
echo ./run_group_sample.sh
./run_group_sample.sh
echo ./run_group_neg_shared.sh
./run_group_neg_shared.sh
echo ./run_neg_shared.sh
./run_neg_shared.sh
