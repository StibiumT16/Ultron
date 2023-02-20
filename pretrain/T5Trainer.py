
import torch
from torch import nn
from tqdm import tqdm
import numpy as np
from collections.abc import Mapping
from evaluate import evaluator
from torch.utils.data import DataLoader
from transformers.trainer_utils import is_main_process
from transformers import TrainingArguments, TrainerCallback
from typing import Dict, List, Tuple, Optional, Any, Union
from transformers.trainer import Trainer


class T5Trainer(Trainer):
    def __init__(self, **kwds):
        super().__init__(**kwds)
    
    def compute_loss(self, model, inputs, return_outputs=False):
        loss = model.forward(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"], labels=inputs['labels']).loss
        if return_outputs:
            return loss, [None, None]
        return loss



class EvalCallback(TrainerCallback):
    def __init__(self, test_dataset, logger, docid_trie, max_length, num_beams, args: TrainingArguments):
        self.logger = logger
        self.args = args
        self.test_dataset = test_dataset
        self.docid_trie = docid_trie
        self.max_length = max_length
        self.num_beams = num_beams
        self.evaluator = evaluator()
        self.dataloader = DataLoader(
            test_dataset,
            batch_size=self.args.per_device_eval_batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=self.args.dataloader_num_workers,
        )
        
    def _prefix_allowed_tokens_fn(self, batch_id, sent):
        return self.docid_trie.get(sent.tolist())
    
    def _docid2string(self, docid):
        x_list = []
        for x in docid:
            if x != 0:
                x_list.append(str(x))
            if x == 1:
                break
        return ",".join(x_list)
    
    def on_epoch_end(self, args, state, control, **kwargs):
        if is_main_process(self.args.local_rank):
            model = kwargs['model'].eval()
            
            truth, prediction = [], []
            
            for testing_data in tqdm(self.dataloader, desc='Evaluating dev queries'):
                input_ids = testing_data["input_ids"]
                attention_mask = testing_data["attention_mask"]
                labels = testing_data["labels"]
                truth.extend([[self._docid2string(docid)] for docid in labels.cpu().numpy().tolist()])
                
                with torch.no_grad():
                    outputs = model.generate(
                        input_ids=input_ids.to(model.device),
                        attention_mask=attention_mask.to(model.device),
                        max_length=self.max_length,
                        num_beams=self.num_beams,
                        prefix_allowed_tokens_fn=self._prefix_allowed_tokens_fn,
                        num_return_sequences=self.num_beams,
                        do_sample=False,)
                    
                    for j in range(input_ids.shape[0]):
                        doc_rank = []
                        batch_output = outputs[j*self.num_beams:(j+1)*self.num_beams].cpu().numpy().tolist()
                        for docid in batch_output:
                            doc_rank.append(self._docid2string(docid))
                        prediction.append(doc_rank)
            
            result =  self.evaluator.evaluate_ranking(truth, prediction)   
            _mrr10, _mrr, _ndcg10, _ndcg20, _ndcg100, _map20, _p1, _p10, _p20, _p100, _r1, _r10, _r100, _r1000 = result
            
            print(f"mrr@10:{_mrr10}, mrr:{_mrr}, r@1:{_r1}, r@10:{_r10}, r@100:{_r100}")
            self.logger.write(f"mrr@10:{_mrr10}, mrr:{_mrr}, r@1:{_r1}, r@10:{_r10}, r@100:{_r100} \n")
            self.logger.flush()



def torch_default_data_collator(features) -> Dict[str, Any]:

    if not isinstance(features[0], Mapping):
        features = [vars(f) for f in features]
    first = features[0]
    batch = {}

    # Special handling for labels.
    # Ensure that tensor is created with the correct type
    # (it should be automatically the case, but let's make sure of it.)
    if "label" in first and first["label"] is not None:
        label = first["label"].item() if isinstance(first["label"], torch.Tensor) else first["label"]
        dtype = torch.long if isinstance(label, int) else torch.float
        batch["labels"] = torch.tensor([f["label"] for f in features], dtype=dtype)
    elif "label_ids" in first and first["label_ids"] is not None:
        if isinstance(first["label_ids"], torch.Tensor):
            batch["labels"] = torch.stack([f["label_ids"] for f in features])
        else:
            dtype = torch.long if type(first["label_ids"][0]) is int else torch.float
            batch["labels"] = torch.tensor([f["label_ids"] for f in features], dtype=dtype)

    # Handling of all other possible keys.
    # Again, we will use the first element to figure out which key/values are not None for this model.
    for k, v in first.items():
        if k not in ("label", "label_ids") and v is not None and not isinstance(v, str):
            if isinstance(v, torch.Tensor):
                batch[k] = torch.stack([f[k] for f in features])
            else:
                batch[k] = torch.tensor(np.array([f[k] for f in features]))

    return batch