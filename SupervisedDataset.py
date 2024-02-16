from datasets import Dataset
import json
from typing import Dict, Optional
from dataclasses import dataclass, field

import torch
from torch.utils.data import Dataset

import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer




class SupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(
        self,
        data_path,
        tokenizer,
        model_max_length=4096,
        user_tokens=[1786, 4194, 95388],
        assistant_tokens=[1786, 10850, 95388],
    ):
        super(SupervisedDataset, self).__init__()
        self.data = json.load(open(data_path))
        self.tokenizer = tokenizer
        self.model_max_length = model_max_length
        self.user_tokens = user_tokens
        self.assistant_tokens = assistant_tokens
        self.ignore_index = -100
        item = self.preprocessing(self.data[0])
        # print("input:", self.tokenizer.decode(item["input_ids"]))
        labels = []
        for id_ in item["label_ids"]:
            if id_ == -100:
                continue

            labels.append(id_)
        # print("label:", self.tokenizer.decode(labels))

    def __len__(self):
        return len(self.data)

    def preprocessing(self, example):
        input_ids = [self.tokenizer.bos_token_id]
        label_ids = []

        for message in example["messages"]:
            role = message["role"]
            content = message["content"]
            content_ids = self.tokenizer.encode(content, add_special_tokens=False)

            if role == "user":
                input_ids += self.user_tokens + content_ids
                label_ids += [self.ignore_index] * len(self.user_tokens) + [
                    self.ignore_index
                ] * len(content_ids)
            else:
                input_ids += self.assistant_tokens + content_ids
                label_ids += (
                    [self.ignore_index] * len(self.assistant_tokens)
                    + content_ids
                    + [self.tokenizer.eos_token_id]
                )

        input_ids = input_ids[: self.model_max_length]
        label_ids = label_ids[: self.model_max_length]
        # input_ids += [self.tokenizer.eos_token_id] * (len(label_ids) - len(input_ids))
        input_ids += [self.tokenizer.eos_token_id] * (
            self.model_max_length - len(input_ids)
        )
        label_ids += [self.ignore_index] * (self.model_max_length - len(label_ids))
        input_ids = torch.LongTensor(input_ids)
        label_ids = torch.LongTensor(label_ids)
        # print(f"len input_ids: {len(input_ids)}, len label_ids: {len(label_ids)}")
        attention_mask = input_ids.ne(self.tokenizer.eos_token_id)
        return {
            "input_ids": input_ids,
            "label_ids": label_ids,
            "attention_mask": attention_mask,
        }

    def __getitem__(self, idx) -> Dict[str, torch.Tensor]:
        return self.preprocessing(self.data[idx])

