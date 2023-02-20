# !/bin/bash

set -e

gpus=${1:-4}
topk=${2:-10}

python src/gen_msmarco_sent.py \
        --output_path corpus/docs-sents.json \
        --doc_file_path raw_data/collection.tsv \
        --qrels_train_path raw_data/qrels.train.tsv \
        --qrels_dev_path raw_data/qrels.dev.small.tsv 

python src/gen_doc_emb.py --input_file corpus/docs-sents.json --output_file corpus/emb_gtr_t5.txt

python src/gen_encoded_docid.py \
        --input_doc_path corpus/docs-sents.json \
        --input_embed_path corpus/emb_gtr_t5.txt \
        --encoding pq \
        --output_path encoded_docid/pq.txt                  

echo "start query generation"
if [ $gpus = 1 ]
then
     distributed_cmd=" "
else
    master_port=$(eval "shuf -i 10000-15000 -n 1")
    distributed_cmd=" -m torch.distributed.launch --nproc_per_node $gpus --master_port=$master_port "
fi

outputpath="qg/fakequery${topk}.txt"

python $distributed_cmd qg/qg.py \
        --model_path castorini/doc2query-t5-base-msmarco \
        --output_path $outputpath \
        --output_dir qg/temp \
        --per_device_eval_batch_size 32 \
        --max_length 256 \
        --valid_file corpus/docs-sents.json \
        --dataloader_num_workers 10 \
        --num_return_sequences $topk \
        --q_max_length 64

python starter.py --encoding pq --cur_data general_pretrain  --data_path corpus/docs-sents.json
python starter.py --encoding pq --cur_data search_pretrain --topk $topk  --data_path corpus/docs-sents.json
python starter.py --encoding pq --cur_data finetune  --data_path corpus/docs-sents.json
python starter.py --encoding pq --cur_data eval  --data_path corpus/docs-sents.json
# python starter.py --encoding pq --cur_data fake_finetune --topk $topk