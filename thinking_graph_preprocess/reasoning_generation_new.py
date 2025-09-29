import os
import sys
import time
import json
import copy
import select
import random
import datetime
import argparse
import pandas as pd
import threading
from tqdm import tqdm
from joblib import Parallel, delayed
from concurrent.futures import ThreadPoolExecutor, wait, TimeoutError

from models.gpt4o import get_gpt4_result
from eval.utils import get_default_config, ask_chatgpt, read_prompts, ask_chatgpt_async_send, ask_chatgpt_async_fetch, read_prompts_pair
from mimiciv_dataset.mimiciv import MIMICIV

PROMPT_W_KNOWLEDGE = """
========================================
# Patient EHR Context #
{context}

========================================
# Retrieved Medical Knowledge #
{medical_knowledge}

========================================
# Ground Truth #
{ground_truth}

========================================
# Task #
{task}
========================================

# Data Description
- # Patient EHR Context #: Contains all medical events and content from the patient's hospitalization journey.
- # Retrieved Medical Knowledge #: Contains Patient EHR Context elements and their relationships potentially relevant to the Ground Truth entity.
- # Ground Truth #: Contains the correct answers for this task; your predictions must exactly match these.
- # Task #: Contains the specific task description; you need to complete the task based on the information in # Patient EHR Context #.
  
# Instructions
Please provide a logically rigorous medical reasoning process so that the # Ground Truth # can be derived from the content in # Patient EHR Context # and # Task #.

# Requirements
The reasoning process should include three stages: Extraction, Reasoning, and Final Results.

## Extraction
- In this stage, extract and identify each piece of "key information" from the # Patient EHR Context # according to the provided # Retrieved Medical Knowledge #.
- Don't pay attention to information that you think is not helpful for the reasoning. 
- Each step in the Extraction stage should follow the format below, **You need to specify the event name and time for each extracted information**:
    **Event Name [Event Time]**: list the information extracted from the event and analyze the potential relationship between the key information and the ground truth.

## Reasoning
- Analyze the relationship between the context information and the item in # Ground Truth # in a very specific and professional manner, providing detailed reasoning steps.
- Your analysis should include the item in #Ground Truth# as much as possible. Items that cannot be inferred from the context can be omitted.
- Do not use the word "maybe", "possible" or "though" in the generated reasoning. You should do your best to find all the supporting information you can to ensure the correctness of your reasoning.
- The reasoning process should be concise and rigorous, and each step should explain the specific medical knowledge involved, making the reasoning process more credible.
- All reasoning must be based on the context information to infer the items in the ground truth and no reverse inference can be performed.

## Final Results
- Provide the final result for the task. **Note that the final result should only contain the items contained in # Gound Truth # that have been correctly inferred in ## Reasoning stage.**
- Each item in the ## Final Results should be contained in the # Gound Truth # with extactly same string.

# Important Notes!!!
- **For each piece of # Retrieved Medical Knowledge # that is relevant to completing the # Task #, locate the exact position of its first item within the # Patient EHR Context # and explicitly annotate it during the Extraction phase to ensure a more thorough analysis.**
- **During the ## Reasoning stage, remember to analyze very carefully how each item in #Ground Truth# is inferred**
- **Most importantly, integrate references to # Ground Truth # and # Retrieved Medical Knowledge # in an implicit manner. At any point in the reasoning process, do not use phrases such as “according to the medical knowledge above”, "as shown in ground truth" or any wording that reveals you are aware of the underlying medical knowledge or the ground truth.**

# Output Format
## Extraction
[YOUR OUTPUT]

## Reasoning
[YOUR OUTPUT]

## Final Results
[YOUR OUTPUT]
"""

PROMPT_WO_KNOWLEDGE = """
========================================
# Patient EHR Context #
{context}

========================================
# Ground Truth #
{ground_truth}

========================================
# Task #
{task}
========================================

# Data Description
- # Patient EHR Context #: Contains all medical events and content from the patient's hospitalization journey.
- # Ground Truth #: Contains the correct answers for this task; your predictions must exactly match these.
- # Task #: Contains the specific task description; you need to complete the task based on the information in # Patient EHR Context #.
  
# Instructions
Please provide a logically rigorous medical reasoning process so that the # Ground Truth # can be derived from the content in # Patient EHR Context # and # Task #.

# Requirements
The reasoning process should include three stages: Extraction, Reasoning, and Final Results.

## Extraction
- Don't pay attention to information that you think is not helpful for the reasoning. 
- Each step in the Extraction stage should follow the format below, **You need to specify the event name and time for each extracted information**:
    **Event Name [Event Time]**: list the information extracted from the event and analyze the potential relationship between the key information and the ground truth.

## Reasoning
- Analyze the relationship between the context information and the item in # Ground Truth # in a very specific and professional manner, providing detailed reasoning steps.
- Your analysis should include the item in #Ground Truth# as much as possible. Items that cannot be inferred from the context can be omitted.
- Do not use the word "maybe", "possible" or "though" in the generated reasoning. You should do your best to find all the supporting information you can to ensure the correctness of your reasoning.
- The reasoning process should be concise and rigorous, and each step should explain the specific medical knowledge involved, making the reasoning process more credible.
- All reasoning must be based on the context information to infer the items in the ground truth and no reverse inference can be performed.

## Final Results
- Provide the final result for the task. **Note that the final result should only contain the items contained in # Gound Truth # that have been correctly inferred in ## Reasoning stage.**
- Each item in the ## Final Results should be contained in the # Gound Truth # with extactly same string.

# Important Notes!!!
- **During the ## Reasoning stage, remember to analyze very carefully how each item in #Ground Truth# is inferred**
- **Most importantly, integrate references to # Ground Truth # in an implicit manner. At any point in the reasoning process, do not use phrases such as “according to ground truth above”, "as shown in ground truth" or any wording that reveals you are aware of the ground truth.**

# Output Format
## Extraction
[YOUR OUTPUT]

## Reasoning
[YOUR OUTPUT]

## Final Results
[YOUR OUTPUT]
"""

PROMPT_WO_FORMAT = """
========================================
# Patient EHR Context #
{context}

========================================
# Ground Truth #
{ground_truth}

========================================
# Task #
{task}
========================================

# Data Description
- # Patient EHR Context #: Contains all medical events and content from the patient's hospitalization journey.
- # Ground Truth #: Contains the correct answers for this task; your predictions must exactly match these.
- # Task #: Contains the specific task description; you need to complete the task based on the information in # Patient EHR Context #.

# Instructions
Please provide a step-by-step reasoning process that leads to the correct prediction for # Task # based on the # Patient EHR Context # and answer the ground truth in the format of `Therefore, the answer is {{ground truth}}`.

# Important Notes!!!
- **Most importantly, integrate references to # Ground Truth # in an implicit manner. At any point in the reasoning process, do not use phrases such as “according to the ground truth above”, "as shown in ground truth" or any wording that reveals you are aware of the ground truth.**
"""

def request_model(msg, model, temp ="0.0"):    
    param = get_default_config(model=model) 
    param["queryConditions"]["messages"][0]["content"] = msg
    param["queryConditions"]["temperature"] = temp
    ask_chatgpt_async_send(param)
    time.sleep(120)
    return ask_chatgpt_async_fetch(param) # 如果请求失败会直接 throw


def request_with_try(args, data, model, cancel_flag):
    retry_count = 0
    result = None
    prompt = None
    time.sleep(random.randint(0, 120)) # 防止所有人一起请求，因为后面 sleep 了 120 秒，所以让大家均匀的散布在 120s 内开始执行
    while retry_count < 5:
        try:
            if cancel_flag.is_set():
                # task cancelled
                return None
            # prompt = get_prompt(data)
            # prompt = f"""Please directly output `{str(data["idx"])}`"""
            if args.without_knowledge:
                prompt = PROMPT_WO_KNOWLEDGE.format(context=data["input"], ground_truth=data["task_info"]["target"], task=data["instruction"])
            
            elif args.without_format:
                prompt = PROMPT_WO_FORMAT.format(context=data["input"], ground_truth=data["task_info"]["target"], task=data["instruction"])
            
            else:
                graph = json.loads(data["task_info"]["graph"])["umls_graph"]
                graph_str = "\n".join([str(evidence) for evidence in graph])
                prompt = PROMPT_W_KNOWLEDGE.format(context=data["input"], medical_knowledge=graph_str, ground_truth=data["task_info"]["target"], task=data["instruction"])  

            param = get_default_config(model=model) 
            param["queryConditions"]["messages"][0]["content"] = prompt
            param["queryConditions"]["temperature"] = str(args.temperature)
            param["queryConditions"]["n"] = args.n
            ask_chatgpt_async_send(param)
            time.sleep(120)
            result = ask_chatgpt_async_fetch(param) # 如果请求失败会直接 throw

            

            break
        except Exception as e:
            print(f"Error: {e}")
            if retry_count == 5 - 1:
                print(
                    f"MAX_RETRIES reached for prompt: {prompt[:20]}..., \nerror {e}, \nresult:{result}")
            # 当前接口分钟级更新流量限制，连续请求容易导致大量报错，加入随机数防止线程都挤到一起。
            time.sleep(random.random() * 15 + 3)
            retry_count += 1

    # print(f'---Prompt---\n{prompt}\n---Response---\n{result}')
    return data, prompt, result


def obtain_dataset(args):
    if os.path.isdir(args.data_index_dir):
        sample_infos = []
        for file_name in os.listdir(args.data_index_dir):
            if file_name.endswith(".csv"):
                data_index_df = pd.read_csv(os.path.join(args.data_index_dir, file_name))
                if not args.without_knowledge or args.filter_nograph:
                    data_index_df = data_index_df.dropna(subset=['graph'])
                sample_infos += data_index_df.to_dict(orient='records')

    elif os.path.isfile(args.data_index_dir):
        data_index_df = pd.read_csv(args.data_index_dir)
        if not args.without_knowledge and not args.without_format:
            data_index_df = data_index_df.dropna(subset=['graph'])
        sample_infos = data_index_df.to_dict(orient='records')
    
    else:
        raise NotImplementedError
    
    for idx, sample_info in enumerate(sample_infos):
        sample_info["idx"] = idx

    cache_datas = []
    cache_idx = []
    if os.path.exists(args.output_path):
        cache_df = pd.read_csv(args.output_path)
        cache_datas = cache_df.to_dict(orient='records')
        cache_idx = [data["idx"] for data in cache_datas]
    
    sample_infos = [sample_info for sample_info in sample_infos if sample_info["idx"] not in cache_idx]
    random.seed(42)
    random.shuffle(sample_infos)

    raw_dataset = MIMICIV(
          sample_info=sample_infos,
          shuffle=False,
          lazzy_mode=True
    )
    print(f"Get {len(raw_dataset)} to be processed and Loading {len(cache_datas)} cache data...")
    return raw_dataset, sample_infos, cache_datas


def save_results(args, futures, cache_datas):
    save_datas = []
    save_datas += cache_datas
    for future in tqdm(futures):
        if not future.done() or future.result() is None:
            continue
        
        data, prompt, result = future.result()
        sample_info = copy.deepcopy(data["task_info"])
        sample_info["reasoning"] = result
        save_datas.append(sample_info)
    
    save_df = pd.DataFrame(save_datas)
    save_df.to_csv(args.output_path, index=False)


def parse_args():
    parser = argparse.ArgumentParser(prog="EHR Data Filter and Selection")

    # basic args
    parser.add_argument("--data_index_dir", type=str, default="/dnn_training_sys/users/longquan.lys/datas/evidence_datas/extracted_concepts_per_sample/train_all_tasks_balance_3000_weight_-1case_based_on_coexist_concepts_umls_lift_minsupport0_threshold50_topk20")
    parser.add_argument("--output_path", type=str, default="/dnn_training_sys/users/longquan.lys/datas/evidence_datas/evidence_reasoning")
    parser.add_argument("--without_knowledge", action="store_true", default=False)
    parser.add_argument("--without_format", action="store_true", default=False)
    parser.add_argument("--filter_nograph", action="store_true", default=False)
    parser.add_argument("--temperature", type=float, default=0.5)
    parser.add_argument("--n", type=int, default=3)
    parser.add_argument("--model", type=str, default="gpt-4o")
    parser.add_argument("--num_worker", type=int, default=200)
    parser.add_argument('--max_timeout_seconds', type=int, default=90000000, help='最长总等待时间，超过了程序会存储已经成功的，自动结束')
    args = parser.parse_args()

    print(f"Without Knowledge Enahnce: {args.without_knowledge}")

    if os.path.isfile(args.data_index_dir):
        data_index_name = args.data_index_dir.rsplit("/", 1)[-1].rsplit(".", 1)[0]
    elif os.path.isdir(args.data_index_dir):
        data_index_name = args.data_index_dir.rsplit("/", 1)[-1]
    else:
        raise NotImplementedError
    

    if args.without_knowledge and args.filter_nograph:
        args.output_path = os.path.join(args.output_path, data_index_name, f"{args.model}-wo-knowledge-filter-nograph-new.csv")
    
    elif args.without_knowledge and not args.filter_nograph:
        args.output_path = os.path.join(args.output_path, data_index_name, f"{args.model}-wo-knowledge-new.csv")
    
    elif args.without_format:
        args.output_path = os.path.join(args.output_path, data_index_name, f"{args.model}-wo-format-new.csv")
    
    else:
        args.output_path = os.path.join(args.output_path, data_index_name, args.model + "-new.csv")
        
    if not os.path.exists(os.path.dirname(args.output_path)):
        os.makedirs(os.path.dirname(args.output_path))

    return args


if __name__ == "__main__":
    args = parse_args()
    dataset, sample_infos, cache_datas = obtain_dataset(args)
    
    # [request_with_try(data, args.model, threading.Event()) for idx, data in enumerate(dataset)]

    # 创建线程池
    with ThreadPoolExecutor(max_workers=args.num_worker) as executor:
        start_time = time.time()
        print(f"Start at: {datetime.datetime.now().strftime('%H:%M:%S')}")

        # 使用线程池并行调用函数
        futures = [executor.submit(request_with_try, args, copy.deepcopy(data), args.model, threading.Event()) for data in tqdm(dataset)]

        total_waited_second = 0
        TIMEOUT_SECONDS_PER_WAIT = 5
        LOG_INTERVAL = 60  # need to be a multiple of TIMEOUT_SECONDS_PER_WAIT
        SAVE_INTERVAL = 1800

        print(f"Begin start...")
        try:
            while total_waited_second < args.max_timeout_seconds:
                any_input, _, _ = select.select([sys.stdin], [], [], 0)

                if any_input:
                    user_input = sys.stdin.readline().strip()
                    print(
                        f"User input:{user_input}. Enter 'quit' or 'exit' to terminate the program and write to output file.\nIt takes up to {TIMEOUT_SECONDS_PER_WAIT}s to terminate.")
                    if user_input in ["quit", "exit"]:
                        raise TimeoutError

                # Wait for all futures to complete with a maximum timeout of 10 seconds
                _, _ = wait(futures, timeout=TIMEOUT_SECONDS_PER_WAIT)
                total_waited_second += TIMEOUT_SECONDS_PER_WAIT

                if total_waited_second % SAVE_INTERVAL == 0:
                    finished_count = sum(future.done() for future in futures)
                    if finished_count > 0:
                        print(f"Saving {finished_count} data....")
                        save_results(args, futures, cache_datas)
                        print(f"Saved {finished_count} data success")

                if total_waited_second % LOG_INTERVAL == 0:
                    finished_count = sum(future.done() for future in futures)

                    if finished_count == len(futures):
                        break

                    # save_results(futures)
                    print(f"Time: {datetime.datetime.now().strftime('%H:%M:%S')}, Number of finished process: {finished_count}. Enter 'quit' or 'exit' to terminate the program and write to output file.")
        except TimeoutError:
            print("Timeout occurred for at least one future.")
        
        
        save_results(args, futures, cache_datas)
        os.kill(os.getpid(), 9)