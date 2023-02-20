import os

## encoding config  # 319927/321631/231695
atomic_config = {"encoding": "atomic", "add_doc_num":319927, "max_docid_length":1}
semantic_config = {"encoding": "semantic_structured", "add_doc_num":100, "max_docid_length":9}
pq24_config = {"encoding": "pq", "add_doc_num":6144, "max_docid_length":24}
url_title_config = {"encoding": "url_title", "add_doc_num":0, "max_docid_length":100}
pred_title_config = {"encoding": "pred_title", "add_doc_num":0, "max_docid_length":60}

config = pq24_config
encoding, add_doc_num, max_docid_length = config["encoding"], config["add_doc_num"], config["max_docid_length"]

## training settings
code_dir = "../"

print("start training...")
max_seq_length = 64
batch_size = 128
epochs = 20

all_data = "pretrain_post_finetune"  # all data used for training  # pretrain_post_finetune
stage = "finetune"  # pretrain / post_pretrain / finetune
feature_num = 1
load_ckpt = "True"
last_stage = "pretrain_post"
last_feature_num = 10
lr = 1e-5

os.system(f"cd {code_dir}/pretrain && python runT5.py \
    --epoch {epochs} \
    --per_gpu_batch_size {batch_size}\
    --learning_rate {lr} \
    --save_path {code_dir}/outputs/ms_psg/{all_data}_{encoding}_{feature_num}/ \
    --log_path {code_dir}/logs/ms_psg/{stage}.{encoding}.{all_data}.{feature_num}.log \
    --doc_file_path {code_dir}/ms_passage/docs-sents.json \
    --pretrain_model_path {code_dir}/transformers_models/t5-base \
    --docid_path {code_dir}/ms_passage/encoded_docid/{encoding}.txt \
    --train_file_path {code_dir}/ms_passage/train_data/{encoding}/{stage}_{max_seq_length}_{feature_num}.json \
    --dataset_script_dir ../data_scripts \
    --dataset_cache_dir ../../negs_tutorial_cache \
    --add_doc_num {add_doc_num} \
    --max_seq_length {max_seq_length} \
    --max_docid_length {max_docid_length} \
    --load_ckpt {load_ckpt} \
    --load_ckpt_path {code_dir}/outputs/ms_psg/{last_stage}_{encoding}_{last_feature_num}/model_19.pkl \
    --output_every_n_step 5000 \
    --save_every_n_epoch 4 \
    --operation training")

print("write success")
