import os
import re
import nltk
import json
import argparse
from tqdm import tqdm
from collections import defaultdict

parser = argparse.ArgumentParser()
parser.add_argument("--output_path", default=None, type=str)
parser.add_argument("--doc_file_path", default="./msmarco_document/msmarco-docs.tsv", type=str)
parser.add_argument("--qrels_train_path", default=None, type=str)
parser.add_argument("--scale", default="300k", type=str)
args = parser.parse_args()


def generate_top_documents(doc_click_count, scale):  # all clicked documents in the train set, almost 10%
    print(f"generating top {scale} dataset.")
    input_path = "./msmarco_document/msmarco-docs-sents-all.json"
    output_path = args.output_path
    count = 0
    with open(input_path, "r") as fr:
        with open(output_path, "w") as fw:
            for line in fr:
                docid = json.loads(line)["docid"]
                if doc_click_count[docid] <= 0:
                    continue
                fw.write(line)
                count += 1
    print(f"count of top {scale}: ", count)


if __name__ == '__main__':
    doc_file_path = args.doc_file_path
    qrels_train_path = args.qrels_train_path
    fout = open("./msmarco_document/msmarco-docs-sents-all.json", "w")
    id_to_content = {}
    doc_click_count = defaultdict(int)
    content_to_id = {}
    
    with open(doc_file_path) as fin:
        for i, line in tqdm(enumerate(fin)):
            cols = line.split("\t")
            if len(cols) != 4:
                continue
            docid, url, title, body = cols
            body = re.sub('<[^<]+?>', '', body).replace('\n', '').strip()
            sents = nltk.sent_tokenize(body)
            id_to_content[docid] = {"docid": docid, "url": url, "title": title, "body": body, "sents": sents}
            doc_click_count[docid] = 0

    print("Total number of unique documents: ", len(doc_click_count))
    
    with open(qrels_train_path, "r") as fr:
        for line in tqdm(fr):
            queryid, _, docid, _ = line.strip().split()
            doc_click_count[docid] += 1

    # 所有doc按照点击query的数量(popularity)由高到低选择，优先使用点击次数多的doc  
    sorted_click_count = sorted(doc_click_count.items(), key=lambda x:x[1], reverse=True)
    # print("sorted_click_count: ", sorted_click_count[:100])
    for docid, count in sorted_click_count:
        if docid not in id_to_content:
            continue
        fout.write(json.dumps(id_to_content[docid])+"\n")

    fout.close()

    # generate top 100k/200k/300k dataset
    generate_top_documents(doc_click_count, scale = args.scale)