import torch
import numpy as np
import json
from tqdm import tqdm, trange
from typing import Any, List, Sequence, Callable
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import argparse

def pad_sequence_to_length(
        sequence: Sequence,
        desired_length: int,
        default_value: Callable[[], Any] = lambda: 0,
        padding_on_right: bool = True,
) -> List:
    sequence = list(sequence)
    if padding_on_right:
        padded_sequence = sequence[:desired_length]
    else:
        padded_sequence = sequence[-desired_length:]
    pad_length = desired_length - len(padded_sequence)
    values_to_pad = [default_value()] * pad_length
    if padding_on_right:
        padded_sequence = padded_sequence + values_to_pad
    else:
        padded_sequence = values_to_pad + padded_sequence
    return padded_sequence

def main(args):

    device=torch.device(f"cuda:{args.cuda_device}")
    ##  You can also download from Hugging Face. This folder should be in the same path as the notebook.
    model = AutoModelForSeq2SeqLM.from_pretrained(args.pretrain_model_path).to(f"cuda:{args.cuda_device}")
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(args.pretrain_model_path)
    id_doc_dict = {}

    with open(args.input_file_path, 'r') as f:
        for line in f:
            line = json.loads(line)
            docid, content = line['docid'], line['body']
            id_doc_dict[docid] = content
    
    text_id_all = list(id_doc_dict.keys())
    text_list_all = [id_doc_dict[id_] for id_ in text_id_all]

    base = int(len(text_list_all) / args.partition_num)
    
    text_partitation = []
    text_partitation_id = []

    text_partitation.append(text_list_all[:base])
    text_partitation_id.append(text_id_all[:base])
    
    for i in range(args.partition_num-2):
        text_partitation.append(text_list_all[(i+1)*base: (i+2)*base])
        text_partitation_id.append(text_id_all[(i+1)*base: (i+2)*base])

    text_partitation.append(text_list_all[(i+2)*base:  ])
    text_partitation_id.append(text_id_all[(i+2)*base:  ])

    output_qg = []
    output_docid = []

    for i in trange(len(text_partitation[args.idx])):

        next_n_lines = text_partitation[args.idx][i]
        batch_input_ids = []
        sen = next_n_lines[:args.max_len]

        batch_input_ids.append(tokenizer.encode(text=sen, add_special_tokens=True))

        max_len = max([len(sen) for sen in batch_input_ids] )
        batch_input_ids = [
            pad_sequence_to_length(
                sequence=sen, desired_length=max_len, default_value=lambda : tokenizer.pad_token_id,
                padding_on_right=False
            ) for sen in batch_input_ids
        ]
        batch_input_ids = torch.tensor(data=batch_input_ids, dtype=torch.int64, device=device)

        generated = model.generate(
            input_ids=batch_input_ids,
            max_length=256,
            do_sample=True,
            num_return_sequences=args.return_num,
        )

        generated = tokenizer.batch_decode(sequences=generated.tolist(), skip_special_tokens=True)
        for index, g in enumerate(generated):
            output_docid.append(text_partitation_id[args.idx][i])
            output_qg.append(g)
    

    with open(args.output_file_path, 'w') as fout:
        for i in range(len(output_qg)):
            fout.write(output_docid[i] + '\t' + output_qg[i] + '\n')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--idx", type=int, default=0, help="partitation")
    parser.add_argument("--partition_num", type=int, default=8, help="partitation")
    parser.add_argument("--input_file_path", type=str, default='', help="input file path")
    parser.add_argument("--output_file_path", type=str, default='', help="output file path")
    parser.add_argument("--pretrain_model_path", type=str, default='castorini/doc2query-t5-base-msmarco', help="pretrain model path")
    parser.add_argument("--max_len", type=int, default=64, help="max length")
    parser.add_argument("--return_num", type=int, default=10, help="return num")
    parser.add_argument("--cuda_device", type=int, default=0, help="cuda")

    args = parser.parse_args()
    print(args)

    main(args)
