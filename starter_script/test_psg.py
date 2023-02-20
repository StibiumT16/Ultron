import os
atomic_config = {"encoding": "atomic", "add_doc_num":319927, "max_docid_length":1}
semantic_config = {"encoding": "semantic", "add_doc_num":100, "max_docid_length":7}
pq24_config = {"encoding": "pq", "add_doc_num":6144, "max_docid_length":24}
url_title_config = {"encoding": "url_title", "add_doc_num":0, "max_docid_length":100}

config = semantic_config
encoding, add_doc_num, max_docid_length = config["encoding"], config["add_doc_num"], config["max_docid_length"]

## test settings
all_data = "pretrain_post_finetune"
max_seq_length = 64
feature_num = 1

batch_size = 8
num_beams = 100


code_dir = "../"
def main():
    for epoch in [19]: #[3,7,11,15,19]:
        os.system(f"cd {code_dir}/pretrain && python runT5.py \
            --per_gpu_batch_size {batch_size} \
            --save_path {code_dir}/outputs/ms_psg/{all_data}_{encoding}_{feature_num}/model_{epoch}.pkl \
            --log_path {code_dir}/logs/ms_psg/eval.{encoding}.{all_data}.log \
            --doc_file_path {code_dir}/ms_passage/docs-sents.json \
            --pretrain_model_path {code_dir}/transformers_models/t5-base \
            --docid_path {code_dir}/ms_passage/encoded_docid/{encoding}.txt \
            --test_file_path {code_dir}/ms_passage/test_data/{encoding}/dev_{max_seq_length}_1.json \
            --dataset_script_dir ../data_scripts \
            --dataset_cache_dir ../../negs_tutorial_cache \
            --num_beams {num_beams} \
            --add_doc_num {add_doc_num} \
            --max_seq_length {max_seq_length} \
            --max_docid_length {max_docid_length} \
            --operation testing \
            --use_docid_rank False")

    print("write success")

if __name__ == '__main__':
    main()