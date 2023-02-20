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
parser.add_argument("--encoding", default="pq", type=str, help="docid method atomic/pq/url")
parser.add_argument("--scale", default="top_300k", type=str, help="top/rand_100k/200k/300k.")
parser.add_argument("--sub_space", default=24, type=int, help='The number of sub-spaces for 768-dim vector.')
parser.add_argument("--cluster_num", default=256, type=int, help="The number of clusters in each sub-space.")
parser.add_argument("--output_path", default="", type=str, help='output path')
parser.add_argument("--pretrain_model_path", default="t5-base", type=str, help='t5 model path')
parser.add_argument("--input_doc_path", default="msmarco_document/msmarco-docs-sents-all.json", type=str, help='doc sents path')
parser.add_argument("--input_embed_path", default="", type=str, help='doc embedding path')
args = parser.parse_args()

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
            docid = doc_item['docid']

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

if __name__ == "__main__":
    top_or_rand, scale = args.scale.split("_")
    input_doc_path = args.input_doc_path
    input_embed_path = args.input_embed_path
    output_path = args.output_path
    
    if args.encoding == "url":
        url_docid(input_doc_path, output_path)