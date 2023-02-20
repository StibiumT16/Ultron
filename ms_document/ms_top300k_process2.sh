#!/bin/bash
set -e

gpus=${1:-4}

cp  qg/top300k_result/ms_top300k.0.tsv qg/ms_top300k_fakequery10.tsv
ITER_NUM=`expr $gpus - 1`
for ITER in $(seq 1 $ITER_NUM )
do
cat qg/top300k_result/ms_top300k.$ITER.tsv >> qg/ms_top300k_fakequery10.tsv
done

echo "concat done!"

mkdir -p dataset/top300k/tmp

python src/starter.py  --encoding url  --cur_data stage1 \
    --data_path msmarco_document/msmarco-docs-sents.top.300k.json \
    --docid_path dataset/top300k/encoded_docid.url.tsv \
    --output_path dataset/top300k \
    --code_path src/gen_train_data.py \
    --pretrain_model_path  /share/project/webbrain-zhouyujia/transfer/transformers_models/t5-base 


python src/starter.py  --encoding url  --cur_data stage2 \
    --data_path msmarco_document/msmarco-docs-sents.top.300k.json \
    --docid_path dataset/top300k/encoded_docid.url.tsv \
    --fake_query_path qg/ms_top300k_fakequery10.tsv \
    --output_path dataset/top300k \
    --code_path src/gen_train_data.py \
    --pretrain_model_path  /share/project/webbrain-zhouyujia/transfer/transformers_models/t5-base 


python src/starter.py  --encoding url  --cur_data stage3 \
    --data_path msmarco_document/msmarco-docs-sents.top.300k.json \
    --docid_path dataset/top300k/encoded_docid.url.tsv \
    --query_path msmarco_document/msmarco-doctrain-queries.tsv \
    --qrels_path msmarco_document/msmarco-doctrain-qrels.tsv \
    --output_path dataset/top300k \
    --code_path src/gen_train_data.py \
    --pretrain_model_path  /share/project/webbrain-zhouyujia/transfer/transformers_models/t5-base 

python src/starter.py  --encoding url  --cur_data eval \
    --data_path msmarco_document/msmarco-docs-sents.top.300k.json \
    --docid_path dataset/top300k/encoded_docid.url.tsv \
    --query_path msmarco_document/msmarco-docdev-queries.tsv \
    --qrels_path msmarco_document/msmarco-docdev-qrels.tsv \
    --output_path dataset/top300k \
    --code_path src/gen_dev_data.py \
    --pretrain_model_path  /share/project/webbrain-zhouyujia/transfer/transformers_models/t5-base 

