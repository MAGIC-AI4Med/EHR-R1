from operator import index
import os 
import json
from datetime import datetime
from torch.utils.data import Dataset, DataLoader
from functools import *
import pyarrow.parquet as pq
import pandas as pd
import random
import csv
from tqdm import tqdm
from dataloader.relation_define import EVENT_TYPE, EVENT_ITEMS_INDEX, EVENT_NAME
from joblib import Parallel, delayed
from mimiciv_dataset.mimiciv import MIMICIV
from dataloader.input_format import safe_read
import argparse
from joblib import Parallel, delayed


def read_parquet(parquet_dir):
    table = pq.read_table(parquet_dir)
    df = table.to_pandas()
    # df['items'] = df['items'].apply(lambda x: json.loads(x.replace("\'", "\"").replace("nan", "null")))
    json_string = df.to_json(orient="records", lines=False)
    data_list = json.loads(json_string)

    for data in data_list:
        data["items"] = json.loads(data["items"]) 
        
    return data_list

def parse_args():
    parser = argparse.ArgumentParser(prog="EHR Data Filter and Selection")

    # basic args
    parser.add_argument("--data_index_dir", type=str, required=True)
    parser.add_argument("--subject_id_path", type=str, required=True)
    parser.add_argument("--data_config", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument("--out_data_index_dir", type=str, default=None)
    parser.add_argument("--in_label_data_index_dir", type=str, default=None)
    parser.add_argument("--balance_sample", action="store_true", default=False)
    parser.add_argument("--balance_force", action="store_true", default=False)
    parser.add_argument("--context_include_admission", action="store_true", default=False)
    parser.add_argument("--force_task_num", type=float, default=None)
    parser.add_argument("--traj_len_min", type=int, default=10)
    parser.add_argument("--traj_len_max", type=int, default=100)

    args = parser.parse_args()

    if not os.path.exists(os.path.dirname(args.output_path)):
        os.makedirs(os.path.dirname(args.output_path))

    if args.data_config:
        with open(args.data_config, "r") as f:
            args.data_config = json.load(f)

    return args

def label_len(val):
    val = eval(val)
    if isinstance(val, str):
        return 1
    elif isinstance(val, list):
        return len(val)
    else:
        raise NotImplementedError

if __name__ == "__main__":
    DATASET = MIMICIV(sample_info=["10000032"], lazzy_mode=True)

    args = parse_args()

    subject_id_df = pd.read_csv(args.subject_id_path)
    subject_id = subject_id_df["subject_id"].to_list()

    if args.out_data_index_dir is not None:
        out_data_index_df = pd.read_csv(args.out_data_index_dir)
        out_data_index_df = out_data_index_df.rename(columns={'period_end': 'context_end', "period_begin": "context_begin"})
        out_data_index_df = out_data_index_df.drop(out_data_index_df.columns[0], axis=1)
    else:
        out_data_index_df = None
    
    if args.in_label_data_index_dir is not None:
        in_label_data_index_df = pd.read_csv(args.in_label_data_index_dir)
        task_in_label = {}
        for task in in_label_data_index_df["task"].unique():
            task_in_label[task] = in_label_data_index_df[in_label_data_index_df["task"] == task]["target"].unique().tolist()
    else:
        task_in_label = None
    
    sample_data_df = None
    data_num_log = {}
    for task_index_file_name in tqdm(os.listdir(args.data_index_dir)):
        task_name = task_index_file_name.split(".")[0]

        if task_name in args.data_config:
            task_sample_num = args.data_config[task_name]
        else:
            print(f"{task_name} not in config!")
            continue
        
        if args.force_task_num is not None:
            if args.force_task_num >= 1:
                task_sample_num = int(args.force_task_num)
            else:
                task_sample_num = int(args.data_config[task_name] * args.force_task_num)

        task_index_df = pd.read_csv(os.path.join(args.data_index_dir, task_index_file_name))
        task_index_df = task_index_df[task_index_df["subject_id"].isin(subject_id)]

        ### TASK FILTER MACHENISM
        # filter history event > 10
        task_index_df = task_index_df[(task_index_df["context_end"] - task_index_df["context_begin"]) > args.traj_len_min]
        task_index_df = task_index_df[(task_index_df["context_end"] - task_index_df["context_begin"]) < args.traj_len_max]

        # filter the case with that 24 hours can cover the effective context
        if 'admissions_id' in task_index_df.columns and task_name in ["procedures_icd", "procedures_ccs", "diagnoses_icd", "diagnoses_ccs"]:
            task_index_df = task_index_df[task_index_df['admissions_id'].isna() | (task_index_df['admissions_id'] > task_index_df['context_begin'])]
        
        if out_data_index_df is not None:
            # task_index_df = task_index_df.drop(task_index_df[task_index_df.isin(out_data_index_df)].index, axis=0)
            task_out_data_index_df = out_data_index_df[out_data_index_df["task"] == task_name]
            task_out_data_index_df["sample_id"] = task_out_data_index_df.apply(lambda row: str(row['subject_id']) + str(row['context_end']), axis=1)
            task_index_df["sample_id"] = task_out_data_index_df.apply(lambda row: str(row['subject_id']) + str(row['context_end']), axis=1)
            task_index_df = task_index_df[~task_index_df["sample_id"].isin(task_out_data_index_df["sample_id"])]

        if task_in_label is not None and task_name in task_in_label:
            task_index_df = task_index_df[task_index_df["target"].isin(task_in_label[task_name])]

        if task_name == "chartevents":
            task_index_df["target_len"] = task_index_df["target"].apply(label_len)
            task_index_df = task_index_df[task_index_df['target_len'] < 10]

        ### TASK FILTER MACHENISM

        if task_sample_num < task_index_df.shape[0] and task_sample_num > 0:
            if args.balance_force and DATASET.task_info[task_name]["task_type"] == "risk_prediction":
                target_list = task_index_df["target"].unique()
                sample_df = pd.DataFrame()
                for target in target_list:
                    target_df = task_index_df[task_index_df["target"] == target]
                    target_sample_num = task_sample_num // len(target_list)
                    if target_sample_num < target_df.shape[0]:
                        target_sampled_df = target_df.sample(n=target_sample_num, replace=False)
                    else:
                        print(f"[Warning] There are only {target_df.shape[0]} sample with target={target}...")
                        target_sampled_df = target_df
                    sample_df = pd.concat([sample_df, target_sampled_df])

            elif args.balance_sample or DATASET.task_info[task_name]["task_type"] == "risk_prediction" or task_name in ["chartevents", "emar", "next_event"]:
                sample_df = task_index_df.sample(n=task_sample_num, weights='target_weight', replace=False) # 采样2行
            else:
                sample_df = task_index_df.sample(n=task_sample_num, replace=False)
        else:
            sample_df = task_index_df
        
        sample_data_df = pd.concat([sample_data_df, sample_df]) if sample_data_df is not None else sample_df
        data_num_log[task_name] = min(task_sample_num, task_index_df.shape[0]) if task_sample_num > 0 else task_index_df.shape[0]
    
    sample_data_df["period_begin"] = sample_data_df["context_begin"]
    sample_data_df["period_end"] = sample_data_df["context_end"]

    sample_data_df = sample_data_df[["subject_id", "hadm_id", "task", "event", "period_begin", "period_end", "admissions_id", "last_discharge_id", "target"]]
    sample_data_df.to_csv(args.output_path, index=False)

    print(data_num_log)
    total_num = sum(list(data_num_log.values()))
    print(f"Get {total_num} sample in total!")