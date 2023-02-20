import os

## encoding config  # 319927/321631/231695
atomic_config = {"encoding": "atomic", "add_doc_num":319927, "max_docid_length":1}
semantic_config = {"encoding": "semantic_structured", "add_doc_num":100, "max_docid_length":9}
pq24_config = {"encoding": "pq24", "add_doc_num":6144, "max_docid_length":24}
url_title_config = {"encoding": "url_title", "add_doc_num":0, "max_docid_length":100}
pred_title_config = {"encoding": "pred_title", "add_doc_num":0, "max_docid_length":60}

config = semantic_config
encoding, add_doc_num, max_docid_length = config["encoding"], config["add_doc_num"], config["max_docid_length"]

## training settings
code_dir = "../"
scale = "300k"   # 100k/300k/300w
top_or_rand = "top"  # top/rand

gpus = 4

print("start finetune...")
model = "t5_128_1"  # the data for current training
load_model = "t5_128_10"  # the data to be loaded
all_data = "finetune"  # all data used for training  # pretrain_post_finetune
cur_data = "query"  # the data used for current training  # pretrain / rank_pretrain / finetune
stage = "finetune"  # pretrain / post_pretrain / finetune
load_ckpt = "False"  # True if load checkpoint, go to load_ckpt_path
operation = "training"  # training / pair_training
max_seq_length = 128
save_every_n_epoch = 4

os.system(f"cd {code_dir}/pretrain && \
    python -m torch.distributed.launch --nproc_per_node {gpus} --master_port=12345 run.py \
    --epoch 20 \
    --per_gpu_batch_size  32 \
    --learning_rate 1e-3 \
    --save_path {code_dir}/outputs/{top_or_rand}_{scale}/Trainer.{all_data}_{encoding}/ \
    --log_path {code_dir}/logs/{top_or_rand}_{scale}/Trainer.{stage}.{encoding}.{all_data}.log \
    --eval_log_path {code_dir}/logs/{top_or_rand}_{scale}/EvalTrainer.{stage}.{encoding}.{all_data}.log \
    --doc_file_path {code_dir}/data/msmarco-data/msmarco-docs-sents.{top_or_rand}.{scale}.json \
    --pretrain_model_path {code_dir}/transformers_models/t5-base \
    --docid_path {code_dir}/data/encoded_docid/t5_{encoding}_{top_or_rand}_{scale}.txt \
    --train_file_path {code_dir}/data/train_data_{top_or_rand}_{scale}/{cur_data}.{model}.{encoding}.{scale}.json \
    --test_file_path {code_dir}/data/test_data_{top_or_rand}_{scale}/query_dev.{model}.{encoding}.{scale}.json \
    --dataset_script_dir ../data_scripts \
    --dataset_cache_dir ../../negs_tutorial_cache \
    --add_doc_num {add_doc_num} \
    --max_seq_length {max_seq_length} \
    --max_docid_length {max_docid_length} \
    --load_ckpt {load_ckpt} \
    --load_ckpt_path {code_dir}/outputs/top_300k/pretrain_pq24/model_19.pkl")

print("write success")
