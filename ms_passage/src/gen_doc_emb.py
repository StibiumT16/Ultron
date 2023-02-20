from sentence_transformers import SentenceTransformer
import json
import argparse
from tqdm import tqdm
model = SentenceTransformer('sentence-transformers/gtr-t5-base')

print("start generate doc embeddings")
print("model name: gtr-t5-base")

parser = argparse.ArgumentParser()
parser.add_argument("--input_file", default="docs-sents.json", type=str)
parser.add_argument("--output_file", default="docemb_gtr_t5.txt", type=str)
args = parser.parse_args()

input_file = args.input_file
output_file = args.output_file
docid = []
sentences = []

with open(input_file, "r") as fin:
    for i, line in enumerate(tqdm(fin)):
        line = json.loads(line)
        docid.append(line['docid'])
        sentences.append(line['body'])
        
embeddings = model.encode(sentences)

with open(output_file, "w") as fout:
    for i in range(len(docid)):
        lst = [str(x) for x in embeddings[i]]
        str_num = ",".join(lst)
        fout.write("[" + str(docid[i]) + "]\t" + str_num +"\n")

