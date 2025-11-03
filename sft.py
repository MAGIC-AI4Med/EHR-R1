import os
import re
import json
import torch

import random
import pathlib
import pandas as pd
from tqdm import tqdm
from datasets import load_dataset
from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM
from dataclasses import dataclass, field
from datasets import Dataset, IterableDataset
from trl import (
    ScriptArguments,
    ModelConfig,
    SFTConfig,
    SFTTrainer,
    TrlParser,
    get_peft_config,
)

from mimiciv_dataset.multidataset import MultipleDataset
from mimiciv_dataset.mimiciv import MIMICIV
from mimiciv_dataset.ehrshot_dataset import EHRSHOTDataset
from models.base_model import LOCAL_MODEL_PATHS
from collections import Counter
from joblib import Parallel, delayed
import csv

try:
    local_rank = int(os.environ["LOCAL_RANK"])
except:
    local_rank = 0

IGNORE_INDEX = 0

@dataclass
class SFTScriptArguments(ScriptArguments):
    root_dir: str = field(metadata={"help": "dataset root"}, default="./datas")
    load_dataset_mode: str = field(metadata={"help": "mode for load dataset, choose from [lazzy, direct]"}, default="direct")
    max_seq_length: int = field(metadata={"help": "max sequence length"}, default=-1)

def format_mimic_data(data, tokenizer, completion_only_loss, max_seq_length=-1):
        # raw_dataset: class MIMICiV
        input_prompt = "{input}\n{instruction}"
        clean_data = {}
        if "messages" not in data:
            messages = [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": input_prompt.format(input=data["input"], instruction=data["instruction"])},
                {"role": "assistant", "content": data["output"]},
            ]   
        else:
            messages = data["messages"]
            
        if completion_only_loss:
            assert len(messages) <= 3
            total_inputs = tokenizer.apply_chat_template(messages, tokenize=False)
            outputs = [message["content"] for message in messages if message["role"] == "assistant"][-1]
            outputs_begin_index = total_inputs.find(outputs)
            inputs = total_inputs[:outputs_begin_index]

            clean_data = tokenizer(total_inputs)
            inputs_token_ids = tokenizer(inputs)["input_ids"]
            inputs_token_len = len(inputs_token_ids)

            clean_data["completion_mask"] = [IGNORE_INDEX] * inputs_token_len + clean_data["input_ids"][inputs_token_len:]
        else:
            clean_data = {}
            # clean_data["messages"] = messages
            total_inputs = tokenizer.apply_chat_template(messages, tokenize=False)
            clean_data = tokenizer(total_inputs, return_tensors="pt")
            if max_seq_length > 0:
                clean_data = {k:v[0, -max_seq_length:] for k,v in clean_data.items()}

        return clean_data


def obtain_dataset_lazzy(script_args, tokenizer, num_workers=8):

    def gen_mimic_data(shards, tokenizer, completion_only_loss, max_seq_length):
        # raw_dataset: class MIMICiV

        for dataset in shards:
            for data in dataset:
                clean_data = format_mimic_data(data, tokenizer, completion_only_loss, max_seq_length)
                yield clean_data 

    if args.dataset_name.endswith("csv"):
        sample_info_df = pd.read_csv(args.dataset_name)
        sample_info = sample_info_df.to_dict(orient='records')
    
    elif args.dataset_name.endswith("jsonl"):
        with open(args.dataset_name, "r") as f:
            sample_info = [json.loads(line) for line in f.readlines()]
    
    else:
        raise NotImplementedError

    # df = pd.read_csv(script_args.dataset_name)
    # sample_info = df.to_dict(orient='records')
    total_size = len(sample_info)
    print(f"Loading {total_size} training sample...")

    dataset_list = [
        MIMICIV(
        root_dir=script_args.root_dir,
        sample_info=sample_info,
        log=False,
        lazzy_mode=True,
    )]

    shards = dataset_list
    total_size = len(shards[0])
    dataset = IterableDataset.from_generator(gen_mimic_data, gen_kwargs={"shards": shards, "tokenizer": tokenizer, "completion_only_loss": script_args.completion_only_loss, "max_seq_length": script_args.max_seq_length})
    print(f"Loading Dataset Finish, dataset.num_shards={dataset.n_shards}")

    return dataset, total_size

def obtain_dataset(script_args, tokenizer, num_workers):

    if "ehrshot" in script_args.dataset_name.lower():
        raw_dataset = EHRSHOTDataset(
            sample_info_path=script_args.dataset_name,
            log=False,
            lazzy_mode=True,
        )
    else:
        raw_dataset = MIMICIV(
            root_dir=script_args.root_dir,
            sample_info_path=script_args.dataset_name,
            log=False,
            lazzy_mode=True,
        )

    dataset = [format_mimic_data(data, tokenizer, script_args.completion_only_loss) for data in tqdm(raw_dataset) if data]

    # data_index = list(range(len(raw_dataset)))
    dataset = Dataset.from_list(dataset)
    total_size = len(dataset)
    return dataset, total_size

def split_dataset(dataset: Dataset, test_size=0.001):
    train_val_split = dataset.train_test_split(test_size=test_size)
    return train_val_split

if __name__ == "__main__":
    parser = TrlParser((SFTScriptArguments, SFTConfig, ModelConfig))
    script_args, training_args, model_config = parser.parse_args_and_config()
    model_config.lora_target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "up_proj", "gate_proj", "down_proj"]
    model_config.model_name_or_path = LOCAL_MODEL_PATHS[model_config.model_name_or_path] if model_config.model_name_or_path in LOCAL_MODEL_PATHS else model_config.model_name_or_path

    ## only completion loss can only work when disable liger kernel for the reason of trl source code
    script_args.completion_only_loss = training_args.completion_only_loss
    training_args.dataset_kwargs = {"skip_prepare_dataset": True}
    if training_args.completion_only_loss:
        training_args.use_liger_kernel = False

    ################
    # Model init kwargs & Tokenizer
    ################
    # config = AutoConfig.from_pretrained(model_config.model_name_or_path, trust_remote_code=True)
    # if getattr(config, "quantization_config", None) is not None:
    #     config.quantization_config["use_exllama"] = False
    model = AutoModelForCausalLM.from_pretrained(model_config.model_name_or_path, trust_remote_code=True, attn_implementation="sdpa")
    tokenizer = AutoTokenizer.from_pretrained(
        model_config.model_name_or_path, trust_remote_code=model_config.trust_remote_code, use_fast=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if model.config.pad_token_id is None:
        model.config.pad_token_id = tokenizer.pad_token_id

    ################
    # Dataset
    ################
    obtain_dataset_func = obtain_dataset_lazzy if script_args.load_dataset_mode == "lazzy" else obtain_dataset
    dataset, data_size = obtain_dataset_func(script_args, tokenizer, training_args.dataset_num_proc)
    # dataset = split_dataset(dataset)
    if script_args.load_dataset_mode == "lazzy":
        training_args.split_batches = True
        batch_size = training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps
        training_args.max_steps = int(training_args.num_train_epochs * data_size // batch_size)
        
    ################
    # Training
    ################
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        eval_dataset=None,
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