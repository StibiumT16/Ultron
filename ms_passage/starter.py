import os
import json
import argparse
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument("--encoding", default="pq", type=str, help="docid method atomic/pq/url")
parser.add_argument("--cur_data", default="general_pretrain", type=str, help="general_pretrain/search_pretrain/finetune")
parser.add_argument("--topk", default=10, type=int, help="number of fake-queries for one document.")
parser.add_argument("--max_seq_length", default=64, type=int, help="max sequence length")
parser.add_argument("--data_path", default="", type=str, help="data path")
args = parser.parse_args()

cur_data = args.cur_data
encoding = args.encoding 
max_seq_length = args.max_seq_length
msmarco_or_nq = "msmarco"

def train_data():
    if cur_data == "general_pretrain":
        target_data = [("passage", 10), ("sampled_terms", 1)]
    elif cur_data == "search_pretrain":
        target_data = [("fake_query", args.topk)]
    elif cur_data == "finetune":
        target_data = [("query", 1)]
    elif cur_data == "fake_finetune":
        os.system(f"cp -r train_data/{encoding}/post_pretrain_{max_seq_length}_{args.topk}.json train_data/{encoding}/fake_finetune_{max_seq_length}_{args.topk}.json")
        os.system(f"cat train_data/{encoding}/finetune_{max_seq_length}_1.json >> train_data/{encoding}/fake_finetune_{max_seq_length}_{args.topk}.json")
        return
    else:
        print("unknown train stage!")
        return
    
    for data_name, sample_for_one_doc in target_data:
        print(f"generating {data_name} ...")
        os.system(f"python src/gen_train_data.py \
            --max_seq_length {max_seq_length} \
            --pretrain_model_path ../../transformers_models/t5-base \
            --data_path {args.data_path} \
            --docid_path encoded_docid/{encoding}.txt \
            --query_path raw_data/queries.train.tsv \
            --qrels_path raw_data/qrels.train.tsv \
            --output_path train_data/{encoding}/{data_name}_{max_seq_length}_{sample_for_one_doc}.json \
            --fake_query_path qg/fakequery{args.topk}.txt\
            --sample_for_one_doc {sample_for_one_doc} \
            --current_data {data_name}")
    
    if cur_data == "general_pretrain":
        passage_input = f"train_data/{encoding}/passage_{max_seq_length}_10.json"
        sampled_input = f"train_data/{encoding}/sampled_terms_{max_seq_length}_1.json"
        merge_output = f"train_data/{encoding}/pretrain_{max_seq_length}_10.json"
        fout = open(merge_output, "w")
        total_count = 0
        with open(passage_input, "r") as fr:
            for line in tqdm(fr, desc="loading passage input"):
                fout.write(line)
                total_count += 1
        with open(sampled_input, "r") as fr:
            for line in tqdm(fr, desc="loading sampled terms input"):
                fout.write(line)
                total_count += 1
        fout.close()
        print("total number of pretrain samples: ", total_count)

    elif cur_data == "search_pretrain":
        fakequery_input = f"train_data/{encoding}/fake_query_{max_seq_length}_{args.topk}.json"
        merge_output = f"train_data/{encoding}/post_pretrain_{max_seq_length}_{args.topk}.json"
        fout = open(merge_output, "w")
        total_count = 0
        with open(fakequery_input, "r") as fr:
            for line in tqdm(fr, desc="loading fakequery input"):
                fout.write(line)
                total_count += 1
        fout.close()
        print("total number of search pretrain samples: ", total_count)
        os.system(f"rm {fakequery_input}")
        
    elif cur_data == "finetune":
        query_input = f"train_data/{encoding}/query_{max_seq_length}_1.json"
        merge_output = f"train_data/{encoding}/finetune_{max_seq_length}_1.json"
        fout = open(merge_output, "w")
        total_count = 0
        with open(query_input, "r") as fr:
            for line in tqdm(fr, desc="loading query input"):
                fout.write(line)
                total_count += 1
        fout.close()
        print("total number of finetune samples: ", total_count)
        os.system(f"rm {query_input}")

    print("write success")

def eval_data():
    print("start generating dev data")
    sample_for_one_doc = 1
    os.system(f"python src/gen_eval_data.py \
        --max_seq_length {max_seq_length} \
        --pretrain_model_path ../../transformers_models/t5-base \
        --data_path corpus/docs-sents.json \
        --docid_path encoded_docid/{encoding}.txt \
        --query_path raw_data/queries.dev.small.tsv \
        --qrels_path raw_data/qrels.dev.small.tsv \
        --output_path test_data/{encoding}/dev_{max_seq_length}_{sample_for_one_doc}.json \
        --current_data query_dev")

if __name__ == '__main__':
    if args.cur_data == 'eval':
        eval_data()
    else:
        train_data()