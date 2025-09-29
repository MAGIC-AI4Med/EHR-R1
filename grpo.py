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
from joblib import Parallel, delayed

from datasets import load_dataset
from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM
from transformers.data.data_collator import DataCollatorMixin
from dataclasses import dataclass, field
from datasets import Dataset, IterableDataset
from trl import (
    ScriptArguments,
    ModelConfig,
    GRPOConfig,
    GRPOTrainer,
    TrlParser,
    get_peft_config
)
from trl.trainer.utils import pad

from mimiciv_dataset.mimiciv import MIMICIV
from mimiciv_dataset.multidataset import MultipleDataset
from models.base_model import LOCAL_MODEL_PATHS
from eval.score_func import calculate_em, calculate_rouge, calculate_acc

try:
    local_rank = int(os.environ["LOCAL_RANK"])
except:
    local_rank = 0

IGNORE_INDEX = 0

@dataclass
class GRPOScriptArguments(ScriptArguments):
    root_dir: str = field(metadata={"help": "dataset root"}, default="/mnt3/longquan.lys/projects/EHRL/mimiciv")
    load_dataset_mode: str = field(metadata={"help": "mode for load dataset, choose from [lazzy, direct]"}, default="direct")
    eval_dataset_name: str = field(metadata={"help": "dataset name for evaluation"}, default="")
    add_consultation_data: bool = field(metadata={"help": "whether load consultation data"}, default=False)
    curriculum: bool = field(metadata={"help": "mix multiple dataset with curriculum learning, only work when dataset_num=2."}, default=False)
    length_curriculum: bool = field(metadata={"help": "mix multiple dataset with curriculum learning, only work when dataset_num=2."}, default=False)

def format_mimic_data(data, tokenizer):
        # raw_dataset: class MIMICiV
        input_prompt = "{input}\n{instruction}\n/think" if data["task_info"]["task_type"] == "decision_making" else "{input}\n{instruction}\n/no_think"
        # input_prompt = think_input_prompt
        # output_prompt = "<think>\n\n</think>\n{prediction}"
        input_text = input_prompt.format(input=data["input"], instruction=data["instruction"])

        clean_data = {}
        prompt_message = [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": input_text},
            ]

        clean_data["prompt"] = tokenizer.apply_chat_template(prompt_message, add_generation_prompt=True, tokenize=False, enable_thinking=True)

        clean_data["metric"] = data["task_info"]["metric"]
        clean_data["task"] = data["task_info"]["task"]
        clean_data["label"] = json.dumps(data["task_info"]["label"])
        return clean_data

def obtain_dataset_lazzy(script_args, dataset_name, tokenizer):

    def gen_mimic_data(shards, tokenizer):
        for dataset in shards:
            for data in dataset:
                sample = format_mimic_data(data, tokenizer)
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

    dataset = IterableDataset.from_generator(gen_mimic_data, gen_kwargs={"shards": shards, "tokenizer":tokenizer})
    print(f"Loading Dataset Finish, dataset.num_shards={dataset.n_shards}")
    return dataset, total_size

def obtain_dataset(script_args, dataset_name, tokenizer):
    raw_dataset = MIMICIV(
        root_dir=script_args.root_dir,
        sample_info_path=dataset_name,
        log=False,
        lazzy_mode=False,
    )

    # dataset = Parallel(n_jobs=-1, backend="multiprocessing")(delayed(format_mimic_data)(data, tokenizer) for data in tqdm(raw_dataset) if data)
    dataset = [format_mimic_data(data, tokenizer) for data in tqdm(raw_dataset) if data]
    print(f"Loading Evaluation Dataset Finish, dataset_num={len(dataset)}")

    # data_index = list(range(len(raw_dataset)))
    dataset = Dataset.from_list(dataset)
    total_size = len(dataset)
    return dataset, total_size

def split_dataset(dataset: Dataset, test_size=0.001):
    train_val_split = dataset.train_test_split(test_size=test_size)
    return train_val_split

def print_trainable_parameters(mdoel):
    for param_name, param in model.named_parameters():
        if param.requires_grad:
            print(f"{param_name}")

def mimiciv_verify_reward(completions, **kwargs):
    score_list = []
    for completion, label, metric, completion_ids in zip(completions, kwargs["label"], kwargs["metric"], kwargs["completion_ids"]):
        if "</think>" not in completion:
            score_list.append(0)
            continue

        prediction = completion.rsplit("</think>", 1)[-1]
        label = json.loads(label)
        if metric in ["em", "f1", "recall", "precision"]:
            prediction = prediction.split("\n")
            prediction = [p.strip() for p in prediction if p.strip()]
            score = calculate_em(label, prediction)["f1"]

        elif metric in ["acc", "aucroc"]:
            score = calculate_acc(label, str(prediction))["acc"]
        
        else:
            raise NotImplementedError

        score_list.append(score)

    return score_list

def mimiciv_format_reward(completions, **kwargs):
    score_list = []
    for completion, prompt in zip(completions, kwargs["prompts"]):
        
        # think format
        if "</think>" not in completion or "<think>" not in completion:
            score_list.append(-1)
            continue

        reasoning, prediction = completion.rsplit("</think>", 1)

        # reasoning format
        if "## Extraction" in reasoning and "## Reasoning" in reasoning and "## Final Results" in reasoning:
            pass
        else:
            score_list.append(-1)
            continue 
        
        # prediction repeat reward
        prediction = prediction.split("\n")
        prediction = [p.strip() for p in prediction if p.strip()]
        if len(list(set(prediction))) != len(prediction):
            score_list.append(-1)
            continue

        score_list.append(1)

    return score_list


if __name__ == "__main__":
    parser = TrlParser((GRPOScriptArguments, GRPOConfig, ModelConfig))
    script_args, training_args, model_config = parser.parse_args_and_config()
    model_config.lora_target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "up_proj", "gate_proj", "down_proj"]
    model_config.model_name_or_path = LOCAL_MODEL_PATHS[model_config.model_name_or_path] if model_config.model_name_or_path in LOCAL_MODEL_PATHS else model_config.model_name_or_path
        
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

    ################
    # Dataset & DataLoader
    ################
    obtain_dataset_func = obtain_dataset_lazzy if script_args.load_dataset_mode == "lazzy" else obtain_dataset
    training_dataset, data_size = obtain_dataset_func(script_args, script_args.dataset_name, tokenizer)
    training_args.generation_kwargs = {"temperature": 0.7, "top_p": 0.95, "top_k": 20}

    eval_dataset, _ = obtain_dataset(script_args, script_args.eval_dataset_name, tokenizer) if script_args.eval_dataset_name else (None, None)

    ################
    # Training
    ################
    trainer = GRPOTrainer(
        model=model,
        args=training_args,
        reward_funcs=[mimiciv_verify_reward],
        train_dataset=training_dataset,
        eval_dataset=eval_dataset,
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