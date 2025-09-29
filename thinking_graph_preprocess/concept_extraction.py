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
from mimiciv_dataset.input_format import MIMICIVStringConvertor as Convertor

from quickumls import QuickUMLS
umls_matcher_list = [QuickUMLS(os.environ["QUICK_UMLS"])] * 10

# EVENT_CONCEPT_KEY = {
#     'omr': ['result_name'],
#     'pharmacy': ['medication'],
#     'prescriptions': ['drug', 'ATC Type'],
#     'poe': ['order_type'],
#     'procedures_icd': ['procedures', 'CCS Type'],
#     'services': ['curr_service'],
#     'labevents': ['item_name'],
#     'microbiologyevents': ['test_name'],
#     'admissions': ['admission_type'],
#     'transfers': ['eventtype', 'careunit'],
#     'emar': ['medication'],
#     'diagnoses_icd': ['diagnoses', 'CCS Type'],
#     'radiology': ['exam_name'],
#     'medrecon': ['name', 'ATC Type'],
#     'pyxis': ['name'],
#     'vitalsign': ['temperature'],
#     'diagnosis': ['icd_title', 'CCS Type'],
#     'chartevents': ['item_name'],
#     'ingredientevents': ['item_name'],
#     'datetimeevents': ['item_name'],
#     'procedureevents': ['item_name'],
#     'inputevents': ['item_name'],
#     'outputevents': ['item_name']
#  }

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
  
def time_gap_hour(time1, time2):
    if not time1 or not time2:
        # if a event have not time, it can be treated as the event far away from each other.
        return 1e9 

    format = '%Y-%m-%d %H:%M:%S'
    try:
        datetime1 = datetime.strptime(time1, format)
    except:
        datetime1 = datetime.strptime(f"{time1} 00:00:00", format)
    
    try:
        datetime2 = datetime.strptime(time2, format)
    except:
        datetime2 = datetime.strptime(f"{time2} 00:00:00", format)

    delta = datetime2 - datetime1
    return delta.total_seconds() / 3600

def get_sample_info(args, subject_id):
    task_df = pd.DataFrame()
    event_item = set(DATASET.convertor.event_info.keys())
    task_item = set(DATASET.task_info.keys())

    for event_name in list(event_item & task_item):
        if event_name == "patients":
            continue

        event_df = pd.read_csv(os.path.join(args.data_index_dir, f"{event_name}.csv"))
        subject_event_df = event_df[event_df["subject_id"] == subject_id]
        task_df = pd.concat([task_df, subject_event_df])

    return task_df

def get_context_begin(trajectory_id, temp_context_bgein, patient_trajectory_list, context_hours=24):
    if patient_trajectory_list[trajectory_id]["file_name"] == "diagnoses_icd":
        patient_trajectory_list[trajectory_id]["starttime"] = patient_trajectory_list[trajectory_id - 1]["starttime"]

    hadm_id = patient_trajectory_list[trajectory_id]["hadm_id"]
    current_event_time = patient_trajectory_list[trajectory_id]["starttime"]

    for context_begin in range(temp_context_bgein, trajectory_id):
        if time_gap_hour(current_event_time, patient_trajectory_list[context_begin]["starttime"]) < context_hours and hadm_id == patient_trajectory_list[context_begin]["hadm_id"]:
            return context_begin
    
    return None

def extract_event_concept(args, EVENT_TASK, thread_id, subject_id):

    if os.path.exists(os.path.join(args.output_path, "event_concepts", f'{subject_id}.pkl')) and args.resume:
        return

    umls_matcher = umls_matcher_list[thread_id % len(umls_matcher_list)]
    try:
        trajectory = read_parquet(os.path.join(args.patient_ehr_path, str(subject_id) + ".parquet"))
        # task_infos = pd.read_csv(os.path.join(args.data_index_dir, str(subject_id) + ".csv"), index_col=False)
    except:
        print(f"""Loading {os.path.join(args.patient_ehr_path, str(subject_id) + ".parquet")} failed!""")
        return

    event_concept_list = []

    for traj_id, event_item in enumerate(trajectory):
        event_name = event_item["file_name"]

        if traj_id == 0 or event_name not in EVENT_TASK:
            event_concept_list.append({"file_name": event_name, "concepts": {}})
            continue
        
        concept_key_info = EVENT_TASK[event_name]
        key_wise_event_concepts = DATASET.convertor.item_info_process(event_item, concept_key_info)

        if event_name == "prescriptions":
            pass

        ## add next_event and radiology and discharge post-process
        if event_name == "admissions":
            if "admission_info" in event_item["items"][0]:
                event_text = "\n\n".join([f"{key}: {value}" for key, value in event_item["items"][0]["admission_info"].items()])
                text_concepts = umls_matcher.match(event_text, best_match=True, ignore_syntax=False)  
                if text_concepts:
                    text_concepts = [info["term"] for info in text_concepts[0]]
                    key_wise_event_concepts["text"] = text_concepts
        if event_name == "radiology":
            event_text = "\n\n".join([item["text"] for item in event_item["items"]])
            text_concepts = umls_matcher.match(event_text, best_match=True, ignore_syntax=False)  
            if text_concepts:
                text_concepts = [info["term"] for info in text_concepts[0]]
                key_wise_event_concepts["text"] = text_concepts
        if event_name == "discharge":
            text_concepts = umls_matcher.match(event_item["items"][0]["text"], best_match=True, ignore_syntax=False)  
            if text_concepts:
                text_concepts = [info["term"] for info in text_concepts[0]]
                key_wise_event_concepts["text"] = text_concepts
        
        event_concept_list.append({"file_name": event_name, "concepts": key_wise_event_concepts})

    with open(os.path.join(args.output_path, "event_concepts", f'{subject_id}.pkl'), 'wb') as f:
        pickle.dump(event_concept_list, f)

def parse_args():
    parser = argparse.ArgumentParser(prog="EHR Data Filter and Selection")

    # basic args
    parser.add_argument("--data_index_dir", type=str, default="./datas/task_index/patient")
    parser.add_argument("--subject_id_path", type=str, default="./datas/patient_data/train.csv")
    parser.add_argument("--patient_ehr_path", type=str, default="./dataspatients_ehr")
    parser.add_argument("--output_path", type=str, default="./datas/evidence_datas")
    parser.add_argument("--item_set_path", type=str, default="./datas/cache/item_set")
    parser.add_argument("--resume", type=bool, default=False)
    # parser.add_argument("--topk", type=int, default=10000)

    args = parser.parse_args()

    if not os.path.exists(os.path.dirname(args.output_path)):
        os.makedirs(os.path.dirname(args.output_path))

    return args

if __name__ == '__main__':
    args = parse_args()
    subject_ids = pd.read_csv(args.subject_id_path)["subject_id"].tolist()
    # subject_ids = subject_ids[:1000]
    
    DATASET = MIMICIV(sample_info=["10000032"], lazzy_mode=True)
    EVENT_TASK = {}
    for task_name, task_info in DATASET.task_info.items():
        if task_info["task_type"] == "risk_prediction" or task_name == "next_event":
            continue

        event_name = task_info.get("event", task_name)
        if event_name not in EVENT_TASK:
            EVENT_TASK[event_name] = {}
        EVENT_TASK[event_name][task_name] = task_info["target_key"]

    print("Begin extract coexists concepts from patient ehr...", flush=True)
    concept_coexistence_list = []
    Parallel(n_jobs=-1, backend="multiprocessing")(delayed(extract_event_concept)(args, EVENT_TASK, thread_id, subject_id) for thread_id, subject_id in tqdm(enumerate(subject_ids), total=len(subject_ids)))


