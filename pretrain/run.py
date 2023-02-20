import os
import time
import torch
import random
import argparse
from utils import *
from tqdm import tqdm
import torch.nn as nn
from trie import Trie
from T5Trainer import T5Trainer, EvalCallback, torch_default_data_collator
from pretrain_dataset import PretrainDataForT5
from transformers import T5Tokenizer, T5ForConditionalGeneration, TrainingArguments

parser = argparse.ArgumentParser()
### training settings
parser.add_argument("--epochs", default=2, type=int, help="Total number of training epochs to perform.")
parser.add_argument("--per_gpu_batch_size", default=25, type=int, help="The batch size.")
parser.add_argument("--learning_rate", default=5e-5, type=float, help="The initial learning rate for Adam.")
parser.add_argument("--warmup_ratio", default=0, type=float, help="The ratio of warmup steps.")
parser.add_argument("--evaluate_every_n_step", default=100, type=int, help="compute metrics every n steps in evaluation process")
parser.add_argument("--add_doc_num", type=int, help="the number of docid to be added.")
parser.add_argument("--max_seq_length", type=int, default=512, help="the max length of input sequences.")
parser.add_argument("--max_docid_length", type=int, default=1, help="the max length of docid sequences.")
parser.add_argument("--num_beams", default=10, type=int, help="the number of beams.")

parser.add_argument("--load_ckpt", default="False", type=str, help="whether to load a trained model checkpoint.")
parser.add_argument("--save_path", default="./model/", type=str, help="The path to save trained models.")
parser.add_argument("--log_path", default="./log/", type=str, help="The path to save log.")
parser.add_argument("--eval_log_path", default="./log/", type=str, help="The path to save my eval log.")
parser.add_argument("--docid_path", default=None, type=str, help='path of the encoded docid.')
parser.add_argument("--train_file_path", type=str, default=None, help="the path/directory of the training file.")
parser.add_argument("--test_file_path", type=str, default=None,  help="the path/directory of the testing file.")
parser.add_argument("--pretrain_model_path", type=str, help="path of the pretrained model checkpoint")
parser.add_argument("--load_ckpt_path", default="./model/", type=str, help="The path to load ckpt of a trained model.")
parser.add_argument("--dataset_script_dir", type=str, help="The path of dataset script.")
parser.add_argument("--dataset_cache_dir", type=str, help="The path of dataset cache.")

parser.add_argument('--local_rank', type=int, default=0)
parser.add_argument("--local_world_size", type=int, default=1)

args = parser.parse_args()
tokenizer = T5Tokenizer.from_pretrained(args.pretrain_model_path)

def load_data(file_path):
    """
        function: 从传入的文件或者目录下加载数据
        args: file_path  -- a directory or a specific file
    """
    if os.path.isfile(file_path):
        fns = [file_path]
    else:
        data_dir = file_path
        fns = [os.path.join(data_dir, fn) for fn in os.listdir(data_dir)]
    # print("file path: ", fns)
    return fns

def load_encoded_docid(docid_path):
    encode_2_docid = {}
    encoded_docid = []
    with open(docid_path, "r") as fr:
        for line in fr:
            docid, encode = line.strip().split("\t")
            docid = docid.lower()
            if "semantic" in docid_path:
                encode = [min(int(x),32227) for x in encode.split(",")]
            else:
                encode = [int(x) for x in encode.split(",")]
            encoded_docid.append(encode)
            encode = ','.join([str(x) for x in encode])
            if encode not in encode_2_docid:
                encode_2_docid[encode] = [docid]
            else:
                encode_2_docid[encode].append(docid)
    return encoded_docid, encode_2_docid

def train_model(train_data, test_data):
    model = T5ForConditionalGeneration.from_pretrained(args.pretrain_model_path)
    model.resize_token_embeddings(model.config.vocab_size + args.add_doc_num)

    encoded_docid, encode_2_docid = load_encoded_docid(args.docid_path)
    docid_trie = Trie([[0] + item for item in encoded_docid])
    
    if args.load_ckpt == "True": # 基于之前的checkpoint开始训练
        save_model = load_model(os.path.join(args.load_ckpt_path))
        vocab_size = save_model["shared.weight"].shape[0]
        if vocab_size < model.config.vocab_size:
            current_model = model.state_dict()
            for key, value in save_model.items():
                if key in ["shared.weight", "encoder.embed_tokens.weight", "decoder.embed_tokens.weight", "lm_head.weight"]:
                    current_model[key][:value.shape[0], :] = value
                else:
                    current_model[key] = value
            save_model = current_model
        model.load_state_dict(save_model)

    train_dataset = PretrainDataForT5(train_data, args.max_seq_length, args.max_docid_length, tokenizer, args.dataset_script_dir, args.dataset_cache_dir, args) 
    test_dataset = PretrainDataForT5(test_data, args.max_seq_length, args.max_docid_length, tokenizer, args.dataset_script_dir, args.dataset_cache_dir, args)
    logger = open(args.eval_log_path, "w")
    
    training_args = TrainingArguments(
        output_dir=args.save_path,
        logging_dir=args.log_path,
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.per_gpu_batch_size,
        evaluation_strategy='no',
        num_train_epochs=args.epochs,
        dataloader_drop_last=False,
        warmup_ratio=args.warmup_ratio,
        save_strategy='epoch', # epoch / steps / no
        logging_strategy='epoch', # epoch / steps / no
        dataloader_num_workers=10,
        max_grad_norm = 5.0, 
        report_to='none', # wandb / none
    )

    trainer = T5Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=None,
        callbacks=[EvalCallback(test_dataset, logger, docid_trie, args.max_docid_length+1, args.num_beams, training_args)],
        data_collator=torch_default_data_collator,
    )
    
    trainer.train()
    logger.close()

if __name__ == '__main__':
    train_data = load_data(args.train_file_path) 
    test_data = load_data(args.test_file_path)
    set_seed() # 控制各种随机种子
    train_model(train_data, test_data) # 开始预训练
