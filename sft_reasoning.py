import os
import re
import json
import torch

import copy
import random
import pathlib
import pandas as pd
from tqdm import tqdm
from typing import Any, Callable, Optional, Union

from datasets import load_dataset
from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM
from transformers.data.data_collator import DataCollatorMixin
from dataclasses import dataclass, field
from datasets import Dataset, IterableDataset
from trl import (
    ScriptArguments,
    ModelConfig,
    SFTConfig,
    SFTTrainer,
    TrlParser,
    get_peft_config
)
from trl.trainer.utils import pad

from mimiciv_dataset.mimiciv import MIMICIV
from mimiciv_dataset.multidataset import MultipleDataset
from models.base_model import LOCAL_MODEL_PATHS

try:
    local_rank = int(os.environ["LOCAL_RANK"])
except:
    local_rank = 0

IGNORE_INDEX = 0

@dataclass
class SFTScriptArguments(ScriptArguments):
    root_dir: str = field(metadata={"help": "dataset root"}, default="./datas")
    load_dataset_mode: str = field(metadata={"help": "mode for load dataset, choose from [lazzy, direct]"}, default="direct")
    eval_dataset_name: str = field(metadata={"help": "dataset name for evaluation"}, default="")
    add_consultation_data: bool = field(metadata={"help": "whether load consultation data"}, default=False)
    curriculum: bool = field(metadata={"help": "mix multiple dataset with curriculum learning, only work when dataset_num=2."}, default=False)
    length_curriculum: bool = field(metadata={"help": "mix multiple dataset with curriculum learning, only work when dataset_num=2."}, default=False)

def format_mimic_data(data, tokenizer, completion_only_loss, max_seq_length=-1):
        # raw_dataset: class MIMICiV
        no_think_input_prompt = "{input}\n{instruction}/no_think"
        think_input_prompt = "{input}\n{instruction}\n/think"
        output_prompt = "<think>\n{reasoning}\n</think>\n\n{prediction}"

        reasoning = data["task_info"].get("reasoning", "")
        try:
            reasoning = eval(reasoning)
            if isinstance(reasoning, list):
                reasoning = reasoning[0]
            elif isinstance(reasoning, dict):
                reasoning = reasoning["choices"][0]["message"]["content"]
            else:
                raise NotImplementedError
        except:
            reasoning = reasoning
        
        if reasoning:
            input_prompt = think_input_prompt
        else:
            input_prompt = no_think_input_prompt

        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": input_prompt.format(input=data["input"], instruction=data["instruction"])},
            {"role": "assistant", "content": output_prompt.format(reasoning=reasoning, prediction=data["output"])},
        ]   

        total_inputs = tokenizer.apply_chat_template(messages, tokenize=False)
        clean_data = tokenizer(total_inputs)

        if completion_only_loss:
            outputs = [message["content"] for message in messages if message["role"] == "assistant"][-1]
            outputs_begin_index = total_inputs.find(outputs)
            inputs = total_inputs[:outputs_begin_index]
            inputs_token_len = len(tokenizer(inputs)["input_ids"])
            clean_data["completion_mask"] = [IGNORE_INDEX] * inputs_token_len + clean_data["input_ids"][inputs_token_len:]

        if max_seq_length > 0:
            for data_name in clean_data:
                clean_data[data_name] = clean_data[data_name][-max_seq_length:]

        return clean_data

def obtain_dataset_lazzy(script_args, dataset_name, tokenizer, num_workers=8):

    def gen_mimic_data(shards, tokenizer, completion_only_loss, max_seq_length):
        for dataset in shards:
            for data in dataset:
                sample = format_mimic_data(data, tokenizer, completion_only_loss, max_seq_length)
                yield sample 


    dataset_list = [
        MIMICIV(
        root_dir=script_args.root_dir,
        sample_info_path=dataset_path,
        log=False,
        lazzy_mode=True,
    ) for dataset_path in dataset_name.split(",")]

    if script_args.add_consultation_data:
        with open("/dnn_training_sys/users/longquan.lys/datas/consultation_datas/SFT_data_full_trainset.json", "r") as f:
            consultation_datas = json.load(f)
            random.shuffle(consultation_datas)

        print(f"Loading {len(consultation_datas)} training sample from consultation datas...")
        dataset_list.append(consultation_datas)
    
    dataset_len = [len(dataset) for dataset in dataset_list]
    print(f"Load {len(dataset_len)} dataset in total, with length list: {dataset_len}")

    print(f"Curriculum setting is {script_args.curriculum}!")
    shards = [
        MultipleDataset(dataset_list, curriculum=script_args.curriculum)
    ]
    total_size = len(shards[0])

    dataset = IterableDataset.from_generator(gen_mimic_data, gen_kwargs={"shards": shards, "tokenizer":tokenizer, "completion_only_loss": script_args.completion_only_loss, "max_seq_length": script_args.max_seq_length})
    print(f"Loading Dataset Finish, dataset.num_shards={dataset.n_shards}")
    # dataset = dataset.shuffle(seed= 42 , buffer_size= 10_000 )
    # print("Shuffle Dataset Finish.")
    # data_list = Parallel(n_jobs=-1)(delayed(format_mimic_data)(data) for data in tqdm(raw_dataset) if data)

    # event_log = Counter([data["event"] for data in data_list])
    # print("The event type distribution in the training data:")
    # print(event_log)

    # dataset = Dataset.from_list(data_list)
    return dataset, total_size

def obtain_dataset(script_args, dataset_name, tokenizer, num_workers=8):
    raw_dataset = MIMICIV(
        root_dir=script_args.root_dir,
        sample_info_path=dataset_name,
        log=False,
        lazzy_mode=True,
    )

    # dataset = Parallel(n_jobs=-1)(delayed(format_mimic_data)(data) for data in tqdm(raw_dataset) if data)
    dataset = [format_mimic_data(data, tokenizer, script_args.completion_only_loss) for data in tqdm(raw_dataset) if data]
    print(f"Loading Evaluation Dataset Finish, dataset_num={len(dataset)}")

    # data_index = list(range(len(raw_dataset)))
    dataset = Dataset.from_list(dataset)
    total_size = len(dataset)
    return dataset, total_size

def split_dataset(dataset: Dataset, test_size=0.001):
    train_val_split = dataset.train_test_split(test_size=test_size)
    return train_val_split

@dataclass
class CustomDataCollator(DataCollatorMixin):
    pad_token_id: int
    padding_side: str = "right"
    completion_only_loss: bool = True
    return_tensors: str = "pt"

    def torch_call(self, examples: list[Union[list[int], Any, dict[str, Any]]]) -> dict[str, Any]:
        # Convert to tensor
        input_ids = [torch.tensor(example["input_ids"]) for example in examples]
        attention_mask = [torch.ones_like(input_ids) for input_ids in input_ids]
        labels = [torch.tensor(example["input_ids"]) for example in examples] # if "labels" not in example else torch.tensor(example["labels"])
        if self.completion_only_loss and "completion_mask" in examples[0]:
            completion_mask = [torch.tensor(example["completion_mask"]) for example in examples]

        # Pad
        output = {}
        output["input_ids"] = pad(input_ids, padding_value=self.pad_token_id, padding_side=self.padding_side)
        output["attention_mask"] = pad(attention_mask, padding_value=0, padding_side=self.padding_side)
        output["labels"] = pad(labels, padding_value=-100, padding_side=self.padding_side)
        if self.completion_only_loss and "completion_mask" in examples[0]:
            completion_mask = pad(completion_mask, padding_value=0, padding_side=self.padding_side)
            output["labels"][completion_mask == 0] = -100  # mask everything that is not in the completion

        return output

def print_trainable_parameters(mdoel):
    for param_name, param in model.named_parameters():
        if param.requires_grad:
            print(f"{param_name}")

if __name__ == "__main__":
    parser = TrlParser((SFTScriptArguments, SFTConfig, ModelConfig))
    script_args, training_args, model_config = parser.parse_args_and_config()
    model_config.lora_target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "up_proj", "gate_proj", "down_proj"]
    model_config.model_name_or_path = LOCAL_MODEL_PATHS[model_config.model_name_or_path] if model_config.model_name_or_path in LOCAL_MODEL_PATHS else model_config.model_name_or_path
    
    ## only completion loss can only work when disable liger kernel for the reason of trl source code
    script_args.completion_only_loss = training_args.completion_only_loss
    script_args.max_seq_length = training_args.max_seq_length
    if training_args.completion_only_loss:
        training_args.dataset_kwargs = {"skip_prepare_dataset": True}
        training_args.use_liger_kernel = False
        
    ################
    # Model init kwargs & Tokenizer
    ################
    config = AutoConfig.from_pretrained(model_config.model_name_or_path)
    if getattr(config, "quantization_config", None) is not None:
        config.quantization_config["use_exllama"] = False
    model = AutoModelForCausalLM.from_pretrained(model_config.model_name_or_path, attn_implementation="sdpa", config=config) # attn_implementation="flash_attention_2"

    tokenizer = AutoTokenizer.from_pretrained(
        model_config.model_name_or_path, trust_remote_code=model_config.trust_remote_code, use_fast=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if model.config.pad_token_id is None:
        model.config.pad_token_id = tokenizer.pad_token_id
    tokenizer.padding_side = "left"

    ################
    # Dataset & DataLoader
    ################
    obtain_dataset_func = obtain_dataset_lazzy if script_args.load_dataset_mode == "lazzy" else obtain_dataset
    training_dataset, data_size = obtain_dataset_func(script_args, script_args.dataset_name, tokenizer, training_args.dataset_num_proc)
    if script_args.load_dataset_mode == "lazzy":
        training_args.split_batches = True
        batch_size = training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps
        training_args.max_steps = int(training_args.num_train_epochs * data_size // batch_size)

    eval_dataset, _ = obtain_dataset(script_args, script_args.eval_dataset_name, tokenizer) if script_args.eval_dataset_name else (None, None)
    data_collector = CustomDataCollator(tokenizer.pad_token_id, padding_side='left', completion_only_loss=script_args.completion_only_loss)

    ################
    # Training
    ################
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=training_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collector,
        processing_class=tokenizer,
        peft_config=get_peft_config(model_config),
    )
    if list(pathlib.Path(training_args.output_dir).glob("checkpoint-*")):
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()

    # Save and push to hub
    trainer.save_model(training_args.output_dir)
    if training_args.push_to_hub:
        trainer.push_to_hub(dataset_name=script_args.dataset_name)