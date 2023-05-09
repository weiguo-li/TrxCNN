#!/bin/bash

nohup python -u train.py > ./train_log_withBucket/Layer4_Filters800_Bsize32Cuda1_1.txt 2>&1 &
# nohup python -u train.py > ./test_log/test_wihtaccuray77_7_top10.txt 2>&1 &
