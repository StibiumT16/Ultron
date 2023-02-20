import os

## encoding config  # 319927/321631/231695
atomic_config = {"encoding": "atomic", "add_doc_num":319927, "max_docid_length":1}
semantic_config = {"encoding": "semantic", "add_doc_num":100, "max_docid_length":7}
pq24_config = {"encoding": "pq", "add_doc_num":6144, "max_docid_length":24}
url_title_config = {"encoding": "url_title", "add_doc_num":0, "max_docid_length":100}

config = semantic_config
encoding, add_doc_num, max_docid_length = config["encoding"], config["add_doc_num"], config["max_docid_length"]

## training settings
code_dir = "../"
max_seq_length = 64
epochs = 20
batch_size = 128 #per gpu


print("start pretrain...")
feature_num1 = 10
all_data = "pretrain"  # all data used for training  # pretrain_post_finetune
stage = "pretrain"  # pretrain / post_pretrain / finetune

os.system(f"cd {code_dir}/pretrain && python runT5.py \
    --epoch {epochs} \
    --per_gpu_batch_size {batch_size}\
    --learning_rate 1e-3 \
    --save_path {code_dir}/outputs/ms_psg/{all_data}_{encoding}_{feature_num1}/ \
    --log_path {code_dir}/logs/ms_psg/{stage}.{encoding}.{all_data}.{feature_num1}.log \
    --doc_file_path {code_dir}/ms_passage/docs-sents.json \
    --pretrain_model_path {code_dir}/transformers_models/t5-base \
    --docid_path {code_dir}/ms_passage/encoded_docid/{encoding}.txt \
    --train_file_path {code_dir}/ms_passage/train_data/{encoding}/{stage}_{max_seq_length}_{feature_num1}.json \
    --dataset_script_dir ../data_scripts \
    --dataset_cache_dir ../../negs_tutorial_cache \
    --add_doc_num {add_doc_num} \
    --max_seq_length {max_seq_length} \
    --max_docid_length {max_docid_length} \
    --load_ckpt False \
    --output_every_n_step 5000 \
    --save_every_n_epoch 4 \
    --operation training")


print("start post pretrain...")
feature_num2 = 10
all_data = "pretrain_post"  # all data used for training  # pretrain_post_finetune
stage = "post_pretrain"  # pretrain / post_pretrain / finetune

os.system(f"cd {code_dir}/pretrain && python runT5.py \
    --epoch 20 \
    --per_gpu_batch_size {batch_size}\
    --learning_rate 1e-3 \
    --save_path {code_dir}/outputs/ms_psg/{all_data}_{encoding}_{feature_num2}/ \
    --log_path {code_dir}/logs/ms_psg/{stage}.{encoding}.{all_data}.{feature_num2}.log \
    --doc_file_path {code_dir}/ms_passage/docs-sents.json \
    --pretrain_model_path {code_dir}/transformers_models/t5-base \
    --docid_path {code_dir}/ms_passage/encoded_docid/{encoding}.txt \
    --train_file_path {code_dir}/ms_passage/train_data/{encoding}/{stage}_{max_seq_length}_{feature_num2}.json \
    --dataset_script_dir ../data_scripts \
    --dataset_cache_dir ../../negs_tutorial_cache \
    --add_doc_num {add_doc_num} \
    --max_seq_length {max_seq_length} \
    --max_docid_length {max_docid_length} \
    --load_ckpt True \
    --load_ckpt_path {code_dir}/outputs/ms_psg/pretrain_{encoding}_{feature_num1}/model_{epochs-1}.pkl \
    --output_every_n_step 5000 \
    --save_every_n_epoch 4 \
    --operation training")


print("start finetune...")
feature_num3 = 1
all_data = "pretrain_post_finetune"  # all data used for training  # pretrain_post_finetune
stage = "finetune"  # pretrain / post_pretrain / finetune

os.system(f"cd {code_dir}/pretrain && python runT5.py \
    --epoch {epochs} \
    --per_gpu_batch_size {batch_size}\
    --learning_rate 1e-3 \
    --save_path {code_dir}/outputs/ms_psg/{all_data}_{encoding}_{feature_num3}/ \
    --log_path {code_dir}/logs/ms_psg/{stage}.{encoding}.{all_data}.{feature_num3}.log \
    --doc_file_path {code_dir}/ms_passage/docs-sents.json \
    --pretrain_model_path {code_dir}/transformers_models/t5-base \
    --docid_path {code_dir}/ms_passage/encoded_docid/{encoding}.txt \
    --train_file_path {code_dir}/ms_passage/train_data/{encoding}/{stage}_{max_seq_length}_{feature_num3}.json \
    --dataset_script_dir ../data_scripts \
    --dataset_cache_dir ../../negs_tutorial_cache \
    --add_doc_num {add_doc_num} \
    --max_seq_length {max_seq_length} \
    --max_docid_length {max_docid_length} \
    --load_ckpt True \
    --load_ckpt_path {code_dir}/outputs/ms_psg/pretrain_post_{encoding}_{feature_num2}/model_{epochs-1}.pkl \
    --output_every_n_step 5000 \
    --save_every_n_epoch 4 \
    --operation training")

print("write success")
