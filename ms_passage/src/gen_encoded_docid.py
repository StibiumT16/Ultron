import os
import re
import math
import time
import json
import nanopq
import pickle
import random
import argparse
import collections
import numpy as np
from tqdm import tqdm
from sklearn.cluster import KMeans
from collections import Counter
from collections import defaultdict
from transformers import T5Tokenizer, T5Model, T5ForConditionalGeneration

MaskedLmInstance = collections.namedtuple("MaskedLmInstance", ["index", "label"])

parser = argparse.ArgumentParser()
parser.add_argument("--input_doc_path", default="docs-sents.json", type=str)
parser.add_argument("--input_embed_path", default=None, type=str)
parser.add_argument("--encoding", default="pq", type=str, help="docid method atomic/pq/url")
parser.add_argument("--sub_space", default=24, type=int, help='The number of sub-spaces for 768-dim vector.')
parser.add_argument("--cluster_num", default=256, type=int, help="The number of clusters in each sub-space.")
parser.add_argument("--output_path", default="encoded_docid/atomic.txt", type=str, help='output path')
parser.add_argument("--pretrain_model_path", default="../../transformers_models/t5-base", type=str, help='t5 model path')
args = parser.parse_args()

def load_doc_vec(input_path):
    docid_2_idx, idx_2_docid = {}, {}
    idx_2_docid = {}
    doc_embeddings = []

    with open(input_path, "r") as fr:
        for line in tqdm(fr, desc="loading doc vectors..."):
            did, demb = line.strip().split('\t')
            d_embedding = [float(x) for x in demb.split(',')]

            docid_2_idx[did] = len(docid_2_idx)
            idx_2_docid[docid_2_idx[did]] = did
            
            doc_embeddings.append(d_embedding)

    print("successfully load doc embeddings.")
    return docid_2_idx, idx_2_docid, np.array(doc_embeddings, dtype=np.float32)

## Encoding documents with atomic unique docid
def atomic_docid(input_path, output_path):
    print("generating atomic docids...")
    model = T5ForConditionalGeneration.from_pretrained(args.pretrain_model_path)
    vocab_size = model.config.vocab_size
    encoded_docids = {}

    with open(input_path) as fin:
        for doc_index, line in tqdm(enumerate(fin), desc='loading all docids'):
            doc_item = json.loads(line)
            docid = doc_item['docid'].lower()
            docid = "[{}]".format(docid)
            
            encoded_docids[docid] = vocab_size + doc_index
    
    with open(output_path, "w") as fw:
        for docid, code in encoded_docids.items():
            doc_code = str(code)
            fw.write(docid + "\t" + doc_code + "\n")

## Encoding documents with product quantization
def product_quantization_docid(args, docid_2_idx, idx_2_docid, doc_embeddings, output_path):
    print("generating product quantization docids...")
    model = T5ForConditionalGeneration.from_pretrained(args.pretrain_model_path)
    vocab_size = model.config.vocab_size

    # Instantiate with M sub-spaces, Ks clusters in each sub-space
    pq = nanopq.PQ(M=args.sub_space, Ks=args.cluster_num)

    # Train codewords
    print("training codewords...")
    pq.fit(doc_embeddings)
    print(np.array(pq.codewords).shape)

    # Encode to PQ-codes
    print("encoding doc embeddings...")
    X_code = pq.encode(doc_embeddings)  # [#doc, sub_space] with dtype=np.uint8

    with open(output_path, "w") as fw:
        for idx, doc_code in tqdm(enumerate(X_code), desc="writing doc code into the file..."):
            docid = idx_2_docid[idx]
            new_doc_code = [int(x) for x in doc_code]
            for i, x in enumerate(new_doc_code):
                new_doc_code[i] = int(x) + i*256
            code = ','.join(str(x + vocab_size) for x in new_doc_code)
            fw.write(docid + "\t" + code + "\n")    

## Encoding documents with token-id in url
def url_docid(input_path, output_path):
    print("generating url title docids...")
    model = T5ForConditionalGeneration.from_pretrained(args.pretrain_model_path)
    vocab_size = model.config.vocab_size
    tokenizer = T5Tokenizer.from_pretrained(args.pretrain_model_path)
    encoded_docids = {}
    max_docid_len = 99

    urls = {}
    with open(input_path) as fin:
        for doc_index, line in tqdm(enumerate(fin), desc='loading all docids'):
            doc_item = json.loads(line)
            docid = doc_item['docid'].lower()
            docid = "[{}]".format(docid)

            url = doc_item['url'].lower()
            title = doc_item['title'].lower().strip()

            url = url.replace("http://","").replace("https://","").replace("-"," ").replace("_"," ").replace("?"," ").replace("="," ").replace("+"," ").replace(".html","").replace(".php","").replace(".aspx","").strip()
            reversed_url = url.split('/')[::-1]
            url_content = " ".join(reversed_url[:-1])
            domain = reversed_url[-1]
            
            url_content = ''.join([i for i in url_content if not i.isdigit()])
            url_content = re.sub(' +', ' ', url_content).strip()

            if len(title.split()) <= 2:
                url = url_content + " " + domain
            else:
                url = title + " " + domain
            
            encoded_docids[docid] = tokenizer(url).input_ids[:-1][:max_docid_len] + [1]  # max docid length
    with open(output_path, "w") as fw:
        for docid, code in encoded_docids.items():
            doc_code = ','.join([str(x) for x in code])
            fw.write(docid + "\t" + doc_code + "\n")

def semantic_docid(args, docid_2_idx, idx_2_docid, doc_embeddings, output_path): # max 32238
    print("generating semantic structured docids...")
    model = T5ForConditionalGeneration.from_pretrained(args.pretrain_model_path)
    vocab_size = model.config.vocab_size
    encoded_docids = {}
    for docid in docid_2_idx:
        encoded_docids[docid] = []
    complete = set()
    max_cluster, max_layer = [0], [0]

    k, c = 10, 100

    def reverse(cur_idx_2_docid, embeddings, layer):
        clusters = KMeans(n_clusters=k, random_state=42).fit(embeddings)
        labels = clusters.labels_
        for i in range(k):
            one_cluster = np.where(labels == i)[0]
            cluster_size = len(one_cluster)
            if layer == 0:
                print(f"processing layer {layer}, cluster {i}, cluster_size {cluster_size}.")
            for index in one_cluster:
                encoded_docids[cur_idx_2_docid[index]].append(i)

            if cluster_size > c and len(set(labels)) > 1:
                temp_embeddings = embeddings[labels == i]
                temp_idx_2_docid = {}
                for index in one_cluster:
                    temp_idx_2_docid[len(temp_idx_2_docid)] = cur_idx_2_docid[index]
                reverse(temp_idx_2_docid, temp_embeddings, layer+1)
            else:
                random.shuffle(one_cluster)
                max_layer[0] = max(max_layer[0], layer)

                if cluster_size > c:
                    print("duplicate embedding: ", cluster_size)
                    total_pos, temp = 1, cluster_size // c
                    while(temp > 0):
                        total_pos += 1
                        temp //= 10
                    print("total position: ", total_pos)
                    
                    for index in range(cluster_size):
                        code = [0] * total_pos
                        code[-1] = index % 100
                        code_idx, temp = -2, index // 100
                        while(temp > 0):
                            code[code_idx] = temp
                            temp //= 10
                            code_idx -= 1

                        encoded_docids[cur_idx_2_docid[one_cluster[index]]].extend(code[:])
                        assert cur_idx_2_docid[one_cluster[index]] not in complete
                        complete.add(cur_idx_2_docid[one_cluster[index]])
                else:
                    for index in range(cluster_size):
                        encoded_docids[cur_idx_2_docid[one_cluster[index]]].append(index)
                        assert cur_idx_2_docid[one_cluster[index]] not in complete
                        complete.add(cur_idx_2_docid[one_cluster[index]])

    reverse(idx_2_docid, doc_embeddings, 0)
    with open(output_path, "w") as fw:
        for docid, code in encoded_docids.items():
            max_layer[0] = max(max_layer[0], len(code))
            doc_code = ','.join([str(x+vocab_size) for x in code]+["1"])
            fw.write(docid + "\t" + doc_code + "\n")
    print("max cluster: ", max_cluster, "max layer: ", max_layer)


if __name__ == "__main__":
    print("start generate encoded docid")
    input_doc_path = args.input_doc_path
    input_embed_path = args.input_embed_path
    output_path = args.output_path
    
    if args.encoding == "atomic":
        atomic_docid(input_doc_path, output_path)
    elif args.encoding == "pq":
        docid_2_idx, idx_2_docid, doc_embeddings = load_doc_vec(input_embed_path)
        product_quantization_docid(args, docid_2_idx, idx_2_docid, doc_embeddings, output_path)
    elif args.encoding == "url":
        url_docid(input_doc_path, output_path)
    elif args.encoding == "semantic":
        docid_2_idx, idx_2_docid, doc_embeddings = load_doc_vec(input_embed_path)
        semantic_docid(args, docid_2_idx, idx_2_docid, doc_embeddings, output_path)
    else:
        print("Wrong docid type!")
        