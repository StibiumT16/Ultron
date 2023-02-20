#!/bin/bash

set -e

python src/gen_ms_sents.py \
    --output_path msmarco_document/msmarco-docs-sents.top.300k.json \
    --doc_file_path msmarco_document/msmarco-docs.tsv \
    --qrels_train_path msmarco_document/msmarco-doctrain-qrels.tsv \
    --scale 300k

mkdir -p dataset/top300k

python src/gen_encoded_docid.py \
        --pretrain_model_path /share/project/webbrain-zhouyujia/transfer/transformers_models/t5-base \
        --encoding url \
        --input_doc_path msmarco_document/msmarco-docs-sents.top.300k.json \
        --output_path dataset/top300k/encoded_docid.url.tsv \
        --scale top_300k 
