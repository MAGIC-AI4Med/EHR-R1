import os
import json
import pickle
import argparse
import pandas as pd
import pyarrow.parquet as pq
from datetime import datetime
from tqdm import tqdm

from collections import defaultdict
from joblib import Parallel, delayed
from scipy.stats import fisher_exact
from mimiciv_dataset.mimiciv import MIMICIV


def safe_read(json_element):
    if isinstance(json_element, float) or isinstance(json_element, int):
        json_element = str(json_element)
    
    if isinstance(json_element, list):
        return json_element
    
    if isinstance(json_element, str):
        if json_element == 'NaN' or json_element == "nan" or json_element == "None":
            return None
    
    if pd.isna(json_element):
        json_element = ""
    
    if json_element is None:
        return None
    
    return json_element

def read_parquet(parquet_dir):
    table = pq.read_table(parquet_dir)
    df = table.to_pandas()
    df["hadm_id"] = df['hadm_id']
    json_string = df.to_json(orient="records", lines=False)
    data_list = json.loads(json_string)

    for data in data_list:
        data["items"] = json.loads(data["items"]) 
        
    return data_list

def get_all_event_concepts(event_concept):
    all_event_concept = []
    for concept_type in event_concept:
        if isinstance(event_concept[concept_type], list):
            all_event_concept += event_concept[concept_type]
        else:
            continue
    
    all_event_concept = list(set(all_event_concept))
    return all_event_concept

def get_task_wise_coexist_metric(args, DATASET, task_name, task_df):

    subject_id = None
    event_concept_list = []
    task_wise_coexit_metric = {"task_num": task_df.shape[0], "concept_num": {}, "coexist_metric": {}}

    for index, row in tqdm(task_df.iterrows(), total=task_df.shape[0]):
        if subject_id != row["subject_id"] or subject_id is None:
            subject_id = row["subject_id"]
            with open(os.path.join(args.event_concept_path, f"{subject_id}.pkl"), "rb") as f:
                event_concept_list = pickle.load(f)
        
        assert DATASET.task_info[task_name].get("event", task_name) == event_concept_list[row["context_end"]]["file_name"]

        # task_key = DATASET.task_info[task_name]["target_key"]
        target_concept = list(set(event_concept_list[row["context_end"]]["concepts"].get(task_name, [])))
        # if target_concept == []:
        #     print("Missing Target Key concept!")
        #     continue

        history_concept = [get_all_event_concepts(event_concept["concepts"]) for event_concept in event_concept_list[row["context_begin"]:row["context_end"]]]
        if safe_read(row.get("last_discharge_id", None)):
            history_concept += [get_all_event_concepts(event_concept_list[int(row["last_discharge_id"])]["concepts"])]
        
        if safe_read(row.get("admissions_id", None)):
            history_concept += [get_all_event_concepts(event_concept_list[int(row["admissions_id"])]["concepts"])]
        history_concept = [c for concept in history_concept for c in concept]
        history_concept = list(set(history_concept))
        
        # add concept_num
        for concept in list(set(target_concept + history_concept)):
            if concept not in task_wise_coexit_metric["concept_num"]:
                task_wise_coexit_metric["concept_num"][concept] = 0
            task_wise_coexit_metric["concept_num"][concept] += 1
        
        for concept in target_concept:
            for h_concept in history_concept:
                if concept == history_concept:
                    continue

                if concept not in task_wise_coexit_metric["coexist_metric"]:
                    task_wise_coexit_metric["coexist_metric"][concept] = {}
                if h_concept not in task_wise_coexit_metric["coexist_metric"][concept]:
                    task_wise_coexit_metric["coexist_metric"][concept][h_concept] = 0
                task_wise_coexit_metric["coexist_metric"][concept][h_concept] += 1
            
                assert task_wise_coexit_metric["coexist_metric"][concept][h_concept] <= task_wise_coexit_metric["concept_num"][h_concept]
        
    with open(os.path.join(args.output_path, f"{task_name}.pkl"), "wb") as f:
        pickle.dump(task_wise_coexit_metric, f)

def parse_args():
    parser = argparse.ArgumentParser(prog="EHR Data Filter and Selection")

    # basic args
    parser.add_argument("--task_index_dir", type=str, default="./datas/task_index/all")
    parser.add_argument("--subject_id_path", type=str, default="./datas/patient_data/train.csv")
    parser.add_argument("--event_concept_path", type=str, default="./datas/evidence_datas/event_concepts")
    parser.add_argument("--output_path", type=str, default="./datas/evidence_datas/task_wise_coexist_concepts")
    parser.add_argument("--resume", type=bool, default=True)
    # parser.add_argument("--topk", type=int, default=10000)

    args = parser.parse_args()

    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)

    return args

if __name__ == '__main__':
    args = parse_args()
    subject_ids = pd.read_csv(args.subject_id_path)["subject_id"].tolist()
    # subject_ids = subject_ids[:1000]
    
    DATASET = MIMICIV(sample_info=["10000032"], lazzy_mode=True)
    print("Begin extract coexists concepts from patient ehr...", flush=True)
    concept_coexistence_list = []

    for task_file in os.listdir(args.task_index_dir):
        task_name = task_file.split(".")[0]
        if DATASET.task_info[task_name]["task_type"] == "risk_prediction" or task_name == "next_event":
            continue

        if os.path.exists(os.path.join(args.output_path, f"{task_name}.pkl")):
            continue
        
        task_df = pd.read_csv(os.path.join(args.task_index_dir, task_file))
        task_df = task_df[task_df['subject_id'].astype(int).isin(subject_ids)]
        ranked_task_df = task_df.groupby('subject_id').apply(lambda x: x)

        get_task_wise_coexist_metric(args, DATASET, task_name, ranked_task_df)


