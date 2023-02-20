import json
import torch
import datasets
import numpy as np
from tqdm import tqdm
from torch import nn
from dataclasses import dataclass, field
from torch.utils.data import Dataset, DataLoader
from transformers import PreTrainedTokenizer, DataCollatorWithPadding
from typing import Dict, List, Tuple, Optional, Any, Union
from transformers.trainer import Trainer
from transformers import (
    PreTrainedTokenizer, 
    DataCollatorWithPadding,
    T5Tokenizer,
    T5TokenizerFast,
    T5ForConditionalGeneration,
    TrainingArguments,
    HfArgumentParser,
    set_seed,
)
set_seed(313)


class GenerateDataset(Dataset):
    def __init__(
            self,
            path_to_data,
            max_length: int,
            cache_dir: str,
            tokenizer: PreTrainedTokenizer,
    ):
        self.data = []
        with open(path_to_data, 'r') as f:
            for data in f:
                data = json.loads(data)
                docid, passage = data['docid'], data['body']
                self.data.append((docid, f'{passage}'))

        self.max_length = max_length
        self.tokenizer = tokenizer
        self.total_len = len(self.data)


    def __len__(self):
        return self.total_len

    def __getitem__(self, item):
        docid, text = self.data[item]
        input_ids = self.tokenizer(text, return_tensors="pt").input_ids[0]
        return input_ids, int(docid)

@dataclass
class QueryEvalCollator(DataCollatorWithPadding):
    def __call__(self, features):
        input_ids = [{'input_ids': x[0]} for x in features]
        labels = [x[1] for x in features]
        inputs = super().__call__(input_ids)
        return inputs, labels
    
class DocTqueryTrainer(Trainer):
    def __init__(self, do_generation: bool, **kwds):
        super().__init__(**kwds)
        self.do_generation = do_generation

    def prediction_step(
        self,
        model: nn.Module,
        inputs: Dict[str, Union[torch.Tensor, Any]],
        prediction_loss_only: bool,
        ignore_keys: Optional[List[str]] = None,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:

        if not self.do_generation:
            return super().prediction_step(
                model, inputs, prediction_loss_only=prediction_loss_only, ignore_keys=ignore_keys
            )

        outputs = self.model.generate(
            input_ids=inputs[0]['input_ids'].to(self.args.device),
            attention_mask=inputs[0]['attention_mask'].to(self.args.device),
            max_length=self.max_length,
            do_sample=True,
            top_k=self.top_k,
            num_return_sequences=self.num_return_sequences)
        labels = torch.tensor(inputs[1], device=self.args.device).repeat_interleave(self.num_return_sequences)

        if outputs.shape[-1] < self.max_length:
            outputs = self._pad_tensors_to_max_len(outputs, self.max_length)
        return (None, outputs.reshape(inputs[0]['input_ids'].shape[0], self.num_return_sequences, -1),
                labels.reshape(inputs[0]['input_ids'].shape[0], self.num_return_sequences, -1))

    def _pad_tensors_to_max_len(self, tensor, max_length):
        if self.tokenizer is not None and hasattr(self.tokenizer, "pad_token_id"):
            # If PAD token is not defined at least EOS token has to be defined
            pad_token_id = (
                self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else self.tokenizer.eos_token_id
            )
        else:
            if self.model.config.pad_token_id is not None:
                pad_token_id = self.model.config.pad_token_id
            else:
                raise ValueError("Pad_token_id must be set in the configuration of the model, in order to pad tensors")

        padded_tensor = pad_token_id * torch.ones(
            (tensor.shape[0], max_length), dtype=tensor.dtype, device=tensor.device
        )
        padded_tensor[:, : tensor.shape[-1]] = tensor
        return padded_tensor

    def predict(
            self,
            test_dataset: Dataset,
            ignore_keys: Optional[List[str]] = None,
            metric_key_prefix: str = "test",
            max_length: Optional[int] = None,
            num_return_sequences: Optional[int] = None,
            top_k: Optional[int] = None,
    ):

        self.max_length = max_length
        self.num_return_sequences = num_return_sequences
        self.top_k = top_k
        return super().predict(test_dataset, ignore_keys=ignore_keys, metric_key_prefix=metric_key_prefix)

@dataclass
class RunArguments:
    model_path: str = field(default=None)
    max_length: Optional[int] = field(default=32)
    valid_file: str = field(default=None)
    output_path : str = field(default="fakequery.txt")
    top_k: Optional[int] = field(default=10)
    num_return_sequences: Optional[int] = field(default=10)
    q_max_length: Optional[int] = field(default=32)

if __name__ == "__main__":
    parser = HfArgumentParser((TrainingArguments, RunArguments))
    training_args, run_args = parser.parse_args_into_dataclasses()
    tokenizer = T5Tokenizer.from_pretrained(run_args.model_path, cache_dir='cache')
    fast_tokenizer = T5TokenizerFast.from_pretrained(run_args.model_path, cache_dir='cache')
    model = T5ForConditionalGeneration.from_pretrained(run_args.model_path, cache_dir='cache')

    
    generate_dataset = GenerateDataset(path_to_data=run_args.valid_file,
                                           max_length=run_args.max_length,
                                           cache_dir='cache',
                                           tokenizer=tokenizer)

    trainer = DocTqueryTrainer(
        do_generation=True,
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        data_collator=QueryEvalCollator(tokenizer, padding='longest'))
    
    predict_results = trainer.predict(generate_dataset,
                                        top_k=run_args.top_k,
                                        num_return_sequences=run_args.num_return_sequences,
                                        max_length=run_args.q_max_length)
    
    with open(run_args.output_path, 'w') as f:
        for batch_tokens, batch_ids in tqdm(zip(predict_results.predictions, predict_results.label_ids), desc="Writing file"):
                for tokens, docid in zip(batch_tokens, batch_ids):
                    f.write('[' + str(docid.item()) + ']\t' + fast_tokenizer.decode(tokens, skip_special_tokens=True) + '\n')