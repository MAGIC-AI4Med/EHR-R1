import os
import re
import sys
import time
import json
import argparse
import random
import copy
import pandas as pd
from tqdm import tqdm
from collections import Counter
from joblib import Parallel, delayed

from models import get_model
from mimiciv_dataset.mimiciv import MIMICIV
from mimiciv_dataset.mimiciv_cdm_dataset import MIMICIVCDM
from mimiciv_dataset.ehrshot_dataset import EHRSHOTDataset
from eval.score_func import calculate_em, calculate_rouge, calculate_acc, calculate_cdm


def parse_args():
    parser = argparse.ArgumentParser(prog="EHR Foundation Model Evaluation")

    # data args
    parser.add_argument("--dataset_name", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument("--resume", action="store_true", default=False)
    # dataset args
    parser.add_argument("--root_dir", type=str, default="./datas")
    parser.add_argument("--lazzy_mode", action="store_true", default=False)
    parser.add_argument("--chunk_num", type=int, default=1)
    parser.add_argument("--chunk_idx", type=int, default=0)
    parser.add_argument("--cdm_candidate", action="store_true", default=False)

    # model args
    parser.add_argument("--model_name_or_path", type=str, required=True)
    parser.add_argument("--use_vllm", action="store_true", default=False)
    parser.add_argument("--url", type=str, default=None)
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.8)
    parser.add_argument("--max_seq_len", type=int, default=32000)

    # inference args
    parser.add_argument("--prompt", action="store_true", default=False)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--sample_num", type=int, default=1)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--top_k", type=int, default=20)
    parser.add_argument("--max_new_tokens", type=int, default=4096)
    parser.add_argument("--think_prompt", action="store_true", default=False)
    parser.add_argument("--direct_answer", action="store_true", default=False)
    parser.add_argument("--oracle_reasoning", action="store_true", default=False)

    # log args
    parser.add_argument(
        "--cache_file", default="cache.jsonl", help="name of the cache file"
    )
    parser.add_argument(
        "--result_file", default="result.json", help="name of the results file"
    )
    args = parser.parse_args()

    if "qwen3" in args.model_name_or_path:
        args.think_prompt = True

    assert args.chunk_idx < args.chunk_num
    if args.chunk_num == 1:
        args.cache_file = os.path.join(args.output_path, args.cache_file)
        args.result_file = os.path.join(args.output_path, args.result_file)
    else:
        args.cache_file = os.path.join(args.output_path, f"cache_{args.chunk_num}_{args.chunk_idx}.jsonl")
        args.result_file = os.path.join(args.output_path, f"result_{args.chunk_num}_{args.chunk_idx}.json")

    if not os.path.exists(os.path.join(args.output_path)):
        os.makedirs(os.path.join(args.output_path), exist_ok=True)

    print("====Input Arguments====")
    print(json.dumps(vars(args), indent=4, sort_keys=False))

    return args

DECISION_MAKING_PROMPT = """Note that you should directly output the answer without any other information. If there are several items in the prediction, please separate them by `\\n`. For all predicted items, please use the item name instead of the item code. Do not output the code like ICD10 or ICD9."""

RISK_PREDICTION_PROMPT = """Note that you should directly output the answer without any other information. You can only choose one answer from the Candidate List."""

CANDIDATE_PROMPT = """You should choose the item from the candidate list below. Candidate List: {candidates}."""

def direct_answer_infer(args, model, examples):
    inputs = [example["prompt"] for example in examples]
    
    assert len(inputs) == 1
    
    if examples[0]["task_info"]["task_type"] == "risk_prediction":
        logit_bias_words = ["yes", "no"]
        max_new_tokens = 1

    else:
        logit_bias_words = None
        max_new_tokens = 128
    
    infer_args = {
        "n": args.sample_num,
        "max_tokens": max_new_tokens,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "top_k": args.top_k
    }

    outputs = model(
        inputs,
        infer_args=infer_args,
        logit_bias_words=logit_bias_words,
        enable_thinking=False
    )
    outputs = [[sample for sample in output] for output in outputs] # flatten
    # trajectories = [f"The answer is {prediction}" for prediction in predictions]
    return outputs

def reasoning_infer(args, model, examples):
    inputs = [example["prompt"] for example in examples]
    
    if examples[0]["task_info"]["task_type"] == "risk_prediction":
        logit_bias_words = ["yes", "no"]
        max_new_tokens = 1
        enable_thinking = False

    else:
        logit_bias_words = None
        max_new_tokens = args.max_new_tokens
        enable_thinking = True
    
    infer_args = {
        "n": args.sample_num,
        "max_tokens": max_new_tokens,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "top_k": args.top_k
    }

    outputs = model(
        inputs,
        infer_args=infer_args,
        logit_bias_words=logit_bias_words,
        enable_thinking=enable_thinking
    )

    def split_reasoning_trajectory(model_name, output):
        try:
            if "reasoning" in output:
                return output

            elif "qwen" in model_name or "m2" in model_name:
                reasoning, trajectory = output["trajectory"].split("</think>")
                reasoning += "</think>"
                trajectory = trajectory.strip()
                output["trajectory"] = trajectory
                output["reasoning"] = reasoning
            elif "gpt_oss" in model_name or "gpt-oss" in model_name:
                trajectory = re.findall(r"<\|channel\|>final<\|message\|>(.*?)<\|return\|>", output["trajectory"], re.DOTALL)[0]
                reasoning = re.findall(r"<\|channel\|>analysis<\|message\|>(.*?)<\|end\|>", output["trajectory"], re.DOTALL)[0]
                trajectory = trajectory.strip()
                output["trajectory"] = trajectory
                output["reasoning"] = reasoning
            else:
                raise NotImplementedError
                
            return output

        except:
            return output
    outputs = [[split_reasoning_trajectory(args.model_name_or_path, sample) for sample in output] for output in outputs] # flatten
    # trajectories = [f"The answer is {prediction}" for prediction in predictions]
    return outputs


def format_mimic_data(args, data):
    input = data["input"]
    instruction = data["instruction"]
    candidates = data["candidates"]

    if data["task_info"]["task"] in ["lab_hyperkalemia"]:
        pass

    if args.prompt:
        PROMPT = DECISION_MAKING_PROMPT if data["task_info"]["task_type"] == "decision_making" else RISK_PREDICTION_PROMPT
        if candidates:
            input_prompt = "\n".join([input, instruction, CANDIDATE_PROMPT.format(candidates=candidates), PROMPT])
        else:
            input_prompt = "\n".join([input, instruction, PROMPT])
    else:
        input_prompt = input + "\n" + instruction
    
    if args.think_prompt:
        if args.direct_answer:
            input_prompt += "\n/no_think"
        else:
            input_prompt += "\n/think"

    if args.oracle_reasoning and data["task_info"].get("reasoning", None):
        try:
            reasoning = eval(data["task_info"]["reasoning"])["choices"][0]["message"]["content"]
            reasoning_wo_reasoning = reasoning.rsplit("Final Results", 1)[0]
            clean_data = {
                "prompt": [
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": input_prompt},
                    {"role": "assistant", "content": f"<think>{reasoning_wo_reasoning}"}
                ]
            }
            data["task_info"]["reasoning"] = reasoning_wo_reasoning
        except:
            clean_data = {
                "prompt": [
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": input_prompt},
                ]
            }
        
    else:
        clean_data = {
            "prompt": [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": input_prompt},
            ]
        }

    clean_data["task_info"] = copy.deepcopy(data["task_info"])
    clean_data["task_info"]["candidates"] = candidates
    return clean_data

def get_data_chunk(sample_infos, chunk_num, chunk_idx):
    chunk_size = len(sample_infos) // chunk_num + 1
    return sample_infos[chunk_idx*chunk_size:(chunk_idx+1)*chunk_size]

def obtain_dataset(args):
    sample_info_df = pd.read_csv(args.dataset_name)
    sample_infos = sample_info_df.to_dict(orient='records')
    sample_infos = get_data_chunk(sample_infos, args.chunk_num, args.chunk_idx)
    
    if "cdm" in args.dataset_name:
        dataset_class = MIMICIVCDM
        raw_dataset = dataset_class(
            sample_info=sample_infos,
            shuffle=False,
            lazzy_mode=args.lazzy_mode,
            candidate=args.cdm_candidate
        )
    
    # elif "ehrshot" in args.dataset_name:
    #     dataset_
    elif "ehrshot" in args.dataset_name:
        dataset_class = EHRSHOTDataset
        raw_dataset = dataset_class(
            sample_info=sample_infos,
            shuffle=False,
            lazzy_mode=args.lazzy_mode
        )

    else:
        dataset_class = MIMICIV
        raw_dataset = dataset_class(
            sample_info=sample_infos,
            shuffle=False,
            lazzy_mode=args.lazzy_mode
        )

    dataset = []
    for new_data in tqdm(raw_dataset):
        dataset.append(format_mimic_data(args, new_data))

    task_log = Counter([data["task_info"]["task"] for data in dataset])
    print("The task type distribution in the test data:")
    print(task_log)

    if args.resume and os.path.exists(args.cache_file):
        with open(args.cache_file, "r") as f:
            cache_datas = [json.loads(line) for line in f.readlines()]
        dataset = dataset[len(cache_datas):]
    else:
        cache_datas = []

    return dataset, cache_datas

def get_score(outputs, task_info):
    if task_info["metric"] in ["em", "f1", "recall", "precision"]:
        if "\\n" in outputs["trajectory"]:
            prediction = outputs["trajectory"].split("\\n")
        else:
            prediction = outputs["trajectory"].split("\n")

        prediction = [p.strip() for p in prediction]
        score = calculate_em(task_info["label"], prediction)

    elif task_info["metric"] == "rouge":
        score = calculate_rouge(task_info["label"], str(outputs["trajectory"]))

    elif task_info["metric"] in ["acc", "aucroc"]:
        score = calculate_acc(task_info["label"], str(outputs["trajectory"]))
    
    elif task_info["metric"] == "cdm_diagnosis":
        prediction = outputs["trajectory"] # + outputs.get("reasoning", "")
        # prediction = outputs["trajectory"]
        cdm_score = calculate_cdm(task_info["category"], prediction)

        prediction = outputs["trajectory"].split("\n")
        prediction = [p.strip() for p in prediction]
        score = calculate_em(task_info["label"], prediction)

        score.update(cdm_score)
    else:
        raise NotImplementedError
    
    return score
    
def score_func(example_logs):
    for log in example_logs:
        if isinstance(log["outputs"], dict):
            log["score"] = get_score(log["outputs"], log["task_info"])

        elif isinstance(log["outputs"], list):
            score_list = []
            for outputs in log["outputs"]:
                score_list.append(get_score(outputs, log["task_info"]))
            
            score_dict = {key: [score[key] for score in score_list] for key in score_list[0]}
            log["score"] = score_dict
        
        else:
            raise NotImplementedError

    return example_logs

def log_cache(example_logs, cache_file):
    with open(cache_file, "w") as f:
        for log in example_logs:
            f.write(json.dumps(log, ensure_ascii=False, separators=(',', ': ')) + "\n")

def update_cache(example_logs, cache_file):
    with open(cache_file, "a") as f:
        for log in example_logs:
            f.write(json.dumps(log, ensure_ascii=False, separators=(',', ': ')) + "\n")

def update_result(args, cache_datas):
    result = {'count': len(cache_datas), 'args': vars(args)}
    with open(args.result_file, "w") as f:
        json.dump(result, f, indent=4, ensure_ascii=False)

if __name__ == "__main__":
    args = parse_args()



    # if len(dataset) > 0:
    dataset, cache_datas = obtain_dataset(args)
    # rescore cache for score func update
    cache_datas = score_func(cache_datas)

    if len(dataset) > 0:
        model = get_model(args.model_name_or_path, args.use_vllm, args.gpu_memory_utilization, max_seq_len=args.max_seq_len, url=args.url)

    # eval begin
    for batch_id in tqdm(
        range(len(dataset)//args.batch + 1), total=len(dataset)//args.batch + 1, desc=f"Inference with k={args.sample_num}"
    ): 
        examples = dataset[args.batch * (batch_id): args.batch * (batch_id + 1)]
        if not examples:
            break

        start_time = time.time()
        if args.direct_answer:
            trajectories = direct_answer_infer(args, model, examples)
        else:
            trajectories = reasoning_infer(args, model, examples)
            
        end_time = time.time()
        # record the evaluation process
        example_logs = [
            {
                "task_info": example["task_info"],
                "outputs": trajectories[example_id]
            }
            for example_id, example in enumerate(examples)
        ]

        # log results
        example_logs = score_func(example_logs)
        update_cache(example_logs, args.cache_file)
        cache_datas += example_logs
        update_result(args, cache_datas)
    
    log_cache(cache_datas, args.cache_file)
    update_result(args, cache_datas)

    
        

    

        
