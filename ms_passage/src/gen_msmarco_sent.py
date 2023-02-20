import re
import nltk
import json
import argparse
import random
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument("--output_path", default="docs-sents.json", type=str)
parser.add_argument("--doc_file_path", default="raw_data/collection.tsv", type=str)
parser.add_argument("--qrels_train_path", default="", type=str)
parser.add_argument("--qrels_dev_path", default="", type=str)
args = parser.parse_args()

random.seed(123)

id_to_content = {}

print("start process corpus")

if "all" in args.output_path:
    print("full corpus")
    with open(args.doc_file_path) as fin,  open(args.output_path, "w") as fout:
        for i, line in tqdm(enumerate(fin)):
            cols = line.split("\t")
            if len(cols) != 2:
                continue
            docid, body = cols
            body = re.sub('<[^<]+?>', '', body).replace('\n', '').strip()
            fout.write(json.dumps({'docid' : docid, 'body' : body, 'sents' : nltk.sent_tokenize(body)})+"\n")
else:
    print("sub corpus")
    corpus = []
    with open(args.qrels_train_path, "r") as fr:
        for line in tqdm(fr):
            queryid, _, docid, _ = line.strip().split()
            corpus.append(docid)

    with open(args.qrels_dev_path, "r") as fr:
        for line in tqdm(fr):
            queryid, _, docid, _ = line.strip().split()
            corpus.append(docid)
    
    print(len(corpus))
    corpus = list(dict.fromkeys(corpus))
    print(len(corpus))

    with open(args.doc_file_path) as fin:
        for i, line in tqdm(enumerate(fin)):
            cols = line.split("\t")
            if len(cols) != 2:
                continue
            docid, body = cols
            id_to_content[docid] = {"docid": docid, "body": body}
    
    with open(args.output_path, "w") as fout:
        for docid in tqdm(corpus):
            if docid not in id_to_content:
                print("docid: " + str(docid) + "not found!" )
                continue
            body = id_to_content[docid]['body']
            body = re.sub('<[^<]+?>', '', body).replace('\n', '').strip()
            fout.write(json.dumps({'docid' : docid, 'body' : body, 'sents' : nltk.sent_tokenize(body)})+"\n")
    