#!/bin/bash

set -e

gpus=${1:-4}

echo "start query generation"
mkdir -p top300k_result
mkdir -p log

ITER_NUM=`expr $gpus - 1`

for ITER in $(seq 0 $ITER_NUM )
do
nohup python -u qg.py --idx $ITER --cuda_device $ITER \
    --pretrain_model_path castorini/doc2query-t5-base-msmarco \
    --input_file_path ../msmarco_document/msmarco-docs-sents.top.300k.json \
    --partition_num $gpus --return_num 10 --max_len 256 \
    --output_file_path top300k_result/ms_top300k.$ITER.tsv \
    > log/ms_top300k_$ITER.log 2>&1 &
done

echo "the generation progress will not show in the shell, you can trace it in ./log/"
echo "only after finishing qg, you can continue to run ../ms_top300k_process2.sh"