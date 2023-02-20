import os
import json
import argparse
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument("--encoding", default="url", type=str, help="docid method atomic/pq/url")
parser.add_argument("--cur_data", default=None, type=str, help="current stage: general_pretrain/search_pretrain/finetune")
parser.add_argument("--data_path", default="msmarco-data/msmarco-docs-sents.top.300k.json", type=str, help='data path')
parser.add_argument("--pretrain_model_path", default="t5-base", type=str, help='bert model path')
parser.add_argument("--output_path", default="DocRetrieval/top300k/", type=str, help='output model path')
parser.add_argument("--docid_path", default="DocRetrieval/top300k/encoded_docid.url.tsv", type=str, help='bert model path')
parser.add_argument("--query_path", default="msmarco_document/msmarco-doctrain-queries.tsv", type=str, help='data path')
parser.add_argument("--qrels_path", default="msmarco_document/msmarco-doctrain-qrels.tsv", type=str, help='data path')
parser.add_argument("--fake_query_path", default=None, type=str, help='fake query path')
parser.add_argument("--code_path", default="src/gen_train_data.py", type=str, help='code path')
parser.add_argument("--max_seq_length", default=128, type=int, help='max input seq length')
args = parser.parse_args()

cur_data = args.cur_data
encoding = args.encoding 


def main():
    if cur_data == "stage1":
        target_data = [("passage", 10), ("sampled_terms", 1)]
    elif cur_data == "stage2":
        target_data = [("fake_query", 10)]
    elif cur_data == "stage3":
        target_data = [("query", 1)]
    else:
        os.system(f"python {args.code_path} \
        --max_seq_length {args.max_seq_length} \
        --pretrain_model_path {args.pretrain_model_path} \
        --data_path {args.data_path} \
        --docid_path {args.docid_path} \
        --query_path {args.query_path} \
        --qrels_path {args.qrels_path} \
        --output_path {args.output_path}/dev.{args.max_seq_length}.{encoding}.json \
        --current_data {cur_data}")
        return

    
    for data_name, sample_for_one_doc in target_data:
        print(f"generating {data_name} ...")
        os.system(f"python {args.code_path} \
            --max_seq_length {args.max_seq_length} \
            --pretrain_model_path {args.pretrain_model_path} \
            --data_path {args.data_path} \
            --docid_path {args.docid_path} \
            --query_path {args.query_path} \
            --qrels_path {args.qrels_path} \
            --output_path {args.output_path}/tmp/{data_name}.{sample_for_one_doc}.json \
            --fake_query_path {args.fake_query_path} \
            --sample_for_one_doc {sample_for_one_doc} \
            --current_data {data_name}")
    
    if cur_data == "stage1":
        passage_input = f"{args.output_path}/tmp/passage.10.json"
        sampled_input = f"{args.output_path}/tmp/sampled_terms.1.json"
        merge_output = f"{args.output_path}/stage1.{args.max_seq_length}.{encoding}.json"
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

    elif cur_data == "stage2":
        fakequery_input = f"{args.output_path}/tmp/fake_query.10.json"
        merge_output = f"{args.output_path}/stage2.{args.max_seq_length}.{encoding}.json"
        fout = open(merge_output, "w")
        total_count = 0
        with open(fakequery_input, "r") as fr:
            for line in tqdm(fr, desc="loading fakequery input"):
                fout.write(line)
                total_count += 1
        fout.close()
        print("total number of search pretrain samples: ", total_count)
        os.system(f"rm {fakequery_input}")
        
    elif cur_data == "stage3":
        query_input = f"{args.output_path}/tmp/query.1.json"
        merge_output = f"{args.output_path}/stage3.{args.max_seq_length}.{encoding}.json"
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

if __name__ == '__main__':
    main()