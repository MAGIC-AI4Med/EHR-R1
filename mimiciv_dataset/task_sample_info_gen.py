import os 
import json
from datetime import datetime
from functools import *
import pyarrow.parquet as pq
import pandas as pd
import random
import csv
from tqdm import tqdm
from joblib import Parallel, delayed
from mimiciv_dataset.mimiciv import MIMICIV
from mimiciv_dataset.input_format import safe_read
import argparse

# initialize dataset
DATASET = MIMICIV(sample_info=["10000032"], lazzy_mode=True)

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

def whether_before(time1, time2):
    assert time1 and time2

    format = '%Y-%m-%d %H:%M:%S'
    try:
        datetime1 = datetime.strptime(time1, format)
    except:
        datetime1 = datetime.strptime(f"{time1} 00:00:00", format)
    
    try:
        datetime2 = datetime.strptime(time2, format)
    except:
        datetime2 = datetime.strptime(f"{time2} 00:00:00", format)
    
    if datetime1 < datetime2:
        return True
    else:
        return False

def edstays_task_info_extraction(task_list, trajectory_id, patient_trajectory_list):
    # target info
    task_target = {}

    hadm_id = patient_trajectory_list[trajectory_id]["hadm_id"]
    ed_outtime = patient_trajectory_list[trajectory_id]["items"][0]["outtime"]

    # format task input
    context_begin_id = trajectory_id
    context_end_id = None
    admissions_event = None
    icustays_event = None
    next_ed_event = None
    same_hadm_last_time = None
    for i in range(trajectory_id+1, len(patient_trajectory_list)):
        if patient_trajectory_list[i]["file_name"] == "admissions" and hadm_id == safe_read(patient_trajectory_list[i]["hadm_id"]) and not admissions_event:
            admissions_event = patient_trajectory_list[i]
        
        if patient_trajectory_list[i]["file_name"] == "icustays" and hadm_id == safe_read(patient_trajectory_list[i]["hadm_id"]) and not icustays_event:
            icustays_event = patient_trajectory_list[i]
        
        if patient_trajectory_list[i]["file_name"] == "edstays" and not next_ed_event:
            next_ed_event = patient_trajectory_list[i]
        
        if patient_trajectory_list[i]["starttime"] is not None and whether_before(patient_trajectory_list[i]["starttime"], ed_outtime):
            context_end_id = i
        
        if hadm_id == safe_read(patient_trajectory_list[i]["hadm_id"]) and patient_trajectory_list[i]["starttime"] is not None:
            same_hadm_last_time = patient_trajectory_list[i]["starttime"]
    
    if context_end_id is None:
        return []

    ## ED_Hospitalization
    if admissions_event:
        task_target["ED_Hospitalization"] = "yes"
    else:
        task_target["ED_Hospitalization"] = "no"
    ## ED_Inpatient_Mortality
    patient_dod_date = safe_read(patient_trajectory_list[0]["items"][0]["dod"])
    if not patient_dod_date:
        task_target["ED_Inpatient_Mortality"] = "no"
    elif not same_hadm_last_time or whether_before(same_hadm_last_time, patient_dod_date):
        task_target["ED_Inpatient_Mortality"] = "no"
    else:
        task_target["ED_Inpatient_Mortality"] = "yes"
        
    
    ## ED_ICU_Tranfer_12hour
    if icustays_event:
        if time_gap_hour(ed_outtime, icustays_event["starttime"]) < 12:
            task_target["ED_ICU_Tranfer_12hour"] = "yes"
        else:
            task_target["ED_ICU_Tranfer_12hour"] = "no"
    
    else:
        task_target["ED_ICU_Tranfer_12hour"] = "no"
    
    ## ED_Reattendance_3day
    if next_ed_event:
        if time_gap_hour(ed_outtime, next_ed_event["starttime"]) <72:
            task_target["ED_Reattendance_3day"] = "yes"
        else:
            task_target["ED_Reattendance_3day"] = "no"
    
    else:
        task_target["ED_Reattendance_3day"] = "no"
    
    ## ED_Critical_Outcomes
    if "ED_ICU_Tranfer_12hour" in task_target or "ED_Inpatient_Mortality" in task_target:
        if task_target.get("ED_ICU_Tranfer_12hour", "no") == "yes" or task_target.get("ED_Inpatient_Mortality", "no") == "yes":
            task_target["ED_Critical_Outcomes"] = "yes"
        else:
            task_target["ED_Critical_Outcomes"] = "no"
    else:
        task_target["ED_Critical_Outcomes"] = "no"
    
    # make sure all the task all get info
    # assert set(task_list) == set(list(task_target.keys())), f"""task_list and task_target are not equal. task_list: {set(task_list)}. task_target: {set(list(task_target.keys()))}"""

    task_info_list = []
    for task_name, output in task_target.items():
        task_info_list.append({
            "subject_id": str(patient_trajectory_list[0]["items"][0]["subject_id"]),
            "hadm_id": str(patient_trajectory_list[trajectory_id]["hadm_id"]),
            "task": task_name,
            "event": patient_trajectory_list[trajectory_id]["file_name"], 
            "context_begin": context_begin_id,
            "context_end": context_end_id+1, 
            "target": output
        })
    return task_info_list

def admissions_task_info_extraction(task_list, trajectory_id, patient_trajectory_list):
    # target info
    task_target = {}

    admissions_time = patient_trajectory_list[trajectory_id]["items"][0]["admittime"]
    discharge_time = patient_trajectory_list[trajectory_id]["items"][0]["dischtime"]

    # format task input
    context_begin_id = trajectory_id
    context_end_id = get_after_context_event(trajectory_id, patient_trajectory_list)
    if context_end_id is None:
        return []

    ## Length of Stay, only work when discharge after 24 hour.
    if whether_before(patient_trajectory_list[context_end_id]["starttime"], discharge_time):
        if time_gap_hour(admissions_time, discharge_time) > 3 * 24:
            task_target["LengthOfStay_3day"] = "yes"
        else:
            task_target["LengthOfStay_3day"] = "no"
        
        if time_gap_hour(admissions_time, discharge_time) > 7 * 24:
            task_target["LengthOfStay_7day"] = "yes"
        else:
            task_target["LengthOfStay_7day"] = "no"
    
    ## Inpatient_Mortality
    patient_dod_date = safe_read(patient_trajectory_list[0]["items"][0]["dod"])
    if not patient_dod_date:
            task_target["Inpatient_Mortality"] = "no"
    else:
        if whether_before(discharge_time, patient_dod_date):
            task_target["Inpatient_Mortality"] = "no"
        else:
            task_target["Inpatient_Mortality"] = "yes"
    
    # make sure all the task all get info
    # assert set(task_list) == set(list(task_target.keys())), f"""task_list and task_target are not equal. task_list: {set(task_list)}. task_target: {set(list(task_target.keys()))}"""

    task_info_list = []
    for task_name, output in task_target.items():
        task_info_list.append({
            "subject_id": str(patient_trajectory_list[0]["items"][0]["subject_id"]),
            "hadm_id": str(patient_trajectory_list[trajectory_id]["hadm_id"]),
            "task": task_name,
            "event": patient_trajectory_list[trajectory_id]["file_name"], 
            "context_begin": context_begin_id,
            "context_end": context_end_id+1, 
            "target": output
        })
    return task_info_list

def icustays_task_info_extraction(task_list, trajectory_id, patient_trajectory_list):
    task_target = {}

    hadm_id = patient_trajectory_list[trajectory_id]["hadm_id"]
    icu_intime = patient_trajectory_list[trajectory_id]["items"][0]["intime"]
    icu_outtime = patient_trajectory_list[trajectory_id]["items"][0]["outtime"]

    # format task input
    context_window = 24
    context_begin_id = trajectory_id
    context_end_id = None
    next_icustays_event = None
    for i in range(trajectory_id+1, len(patient_trajectory_list)):
        
        if patient_trajectory_list[i]["file_name"] == "icustays" and hadm_id == safe_read(patient_trajectory_list[i]["hadm_id"]) and not next_icustays_event:
            next_icustays_event = patient_trajectory_list[i]
        
        if time_gap_hour(icu_intime, patient_trajectory_list[i]["starttime"]) < context_window and hadm_id == safe_read(patient_trajectory_list[i]["hadm_id"]):
            context_end_id = i
    
    if context_end_id is None:
        return []
    
    context_end_time = patient_trajectory_list[context_end_id]["starttime"]
    
    ## ICU Mortality
    patient_dod_date = safe_read(patient_trajectory_list[0]["items"][0]["dod"])
    if not patient_dod_date:
        task_target["ICU_Mortality_1day"] = "no"
        task_target["ICU_Mortality_2day"] = "no"
        task_target["ICU_Mortality_3day"] = "no"
        task_target["ICU_Mortality_7day"] = "no"
        task_target["ICU_Mortality_14day"] = "no"
    
    else:
        for day_num in [1, 2, 3, 7, 14]:
            if time_gap_hour(context_end_time, patient_dod_date) < day_num * 24:
                task_target[f"ICU_Mortality_{day_num}day"] = "yes"
            else:
                task_target[f"ICU_Mortality_{day_num}day"] = "no"
    
    # ICU_Stay
    if time_gap_hour(icu_intime, icu_outtime) < 7 * 24:
        task_target["ICU_Stay_7day"] = "no"
    else:
        task_target["ICU_Stay_7day"] = "yes"

    if time_gap_hour(icu_intime, icu_outtime) < 14 * 24:
        task_target["ICU_Stay_14day"] = "no"
    else:
        task_target["ICU_Stay_14day"] = "yes"
    
    # ICU_Readmission
    if next_icustays_event:
        task_target["ICU_Readmission"] = "yes"
    else:
        task_target["ICU_Readmission"] = "no"
    
    # make sure all the task all get info
    # assert set(task_list) == set(list(task_target.keys())), f"""task_list and task_target are not equal. task_list: {set(task_list)}. task_target: {set(list(task_target.keys()))}"""
    
    task_info_list = []
    for task_name, output in task_target.items():
        task_info_list.append({
            "subject_id": str(patient_trajectory_list[0]["items"][0]["subject_id"]),
            "hadm_id": str(patient_trajectory_list[trajectory_id]["hadm_id"]),
            "task": task_name,
            "event": patient_trajectory_list[trajectory_id]["file_name"], 
            "context_begin": context_begin_id,
            "context_end": context_end_id+1, 
            "target": output
        })
    return task_info_list

def discharge_task_info_extraction(task_list, trajectory_id, patient_trajectory_list):
    task_target = {}

    hadm_id = patient_trajectory_list[trajectory_id]["hadm_id"]
    discharge_time = patient_trajectory_list[trajectory_id]["starttime"]

    context_end_id = trajectory_id
    next_admissions_event = None
    for i in range(trajectory_id+1, len(patient_trajectory_list)):
        if patient_trajectory_list[i]["file_name"] == "admissions" and not next_admissions_event:
            next_admissions_event = patient_trajectory_list[i]
            break
    
    context_begin_id = get_previous_context_event(trajectory_id, patient_trajectory_list)

    if context_begin_id is None:
        return []
        
    if next_admissions_event:
        if time_gap_hour(discharge_time, next_admissions_event["starttime"]) < 24 * 30:
            task_target["Readmission_30day"] = "yes"
        else:
            task_target["Readmission_30day"] = "no"
    
        if time_gap_hour(discharge_time, next_admissions_event["starttime"]) < 24 * 60:
            task_target["Readmission_60day"] = "yes"
        else:
            task_target["Readmission_60day"] = "no"
    else:
        task_target["Readmission_30day"] = "no"
        task_target["Readmission_60day"] = "no"
    
    # make sure all the task all get info
    # assert set(task_list) == set(list(task_target.keys())), f"""task_list and task_target are not equal. task_list: {set(task_list)}. task_target: {set(list(task_target.keys()))}"""
    
    task_info_list = []
    for task_name, output in task_target.items():
        task_info_list.append({
            "subject_id": str(patient_trajectory_list[0]["items"][0]["subject_id"]),
            "hadm_id": str(patient_trajectory_list[trajectory_id]["hadm_id"]),
            "task": task_name,
            "event": patient_trajectory_list[trajectory_id]["file_name"], 
            "context_begin": context_begin_id,
            "context_end": context_end_id+1, 
            "target": output
        })
    return task_info_list

def get_previous_context_event(trajectory_id, patient_trajectory_list, context_hours=24):
    if patient_trajectory_list[trajectory_id]["file_name"] == "diagnoses_icd":
        patient_trajectory_list[trajectory_id]["starttime"] = patient_trajectory_list[trajectory_id - 1]["starttime"]

    hadm_id = patient_trajectory_list[trajectory_id]["hadm_id"]
    current_event_time = patient_trajectory_list[trajectory_id]["starttime"]

    context_begin_id = None
    for event_id in range(trajectory_id-1, 0, -1):
        if time_gap_hour(patient_trajectory_list[event_id]["starttime"], current_event_time) < context_hours and hadm_id == patient_trajectory_list[event_id]["hadm_id"]:
            context_begin_id = event_id
    
    return context_begin_id

def get_after_context_event(trajectory_id, patient_trajectory_list, context_hours=24):
    hadm_id = patient_trajectory_list[trajectory_id]["hadm_id"]
    current_event_time = patient_trajectory_list[trajectory_id]["starttime"]

    context_end_id = None
    for event_id in range(trajectory_id+1, len(patient_trajectory_list)):
        if time_gap_hour(current_event_time, patient_trajectory_list[event_id]["starttime"]) < context_hours and hadm_id == patient_trajectory_list[event_id]["hadm_id"]:
            context_end_id = event_id
    
    return context_end_id

def get_decision_making_task(task_list, trajectory_id, patient_trajectory_list):
    task_info_list = []
    for task_name in task_list:
        if task_name in ["diagnoses_icd", "diagnoses_ccs"]:
            patient_trajectory_list[trajectory_id]["starttime"] = patient_trajectory_list[trajectory_id - 1]["starttime"]

        output = DATASET.convertor.output_process(task_name, patient_trajectory_list[trajectory_id], DATASET.task_info[task_name]["target_key"])
        if not output:
            continue

        context_begin_id = get_previous_context_event(trajectory_id, patient_trajectory_list)
        if context_begin_id is not None:
            task_info_list.append({
                "subject_id": str(patient_trajectory_list[0]["items"][0]["subject_id"]),
                "hadm_id": str(patient_trajectory_list[trajectory_id]["hadm_id"]),
                "task": task_name,
                "event": patient_trajectory_list[trajectory_id]["file_name"], 
                "context_begin": context_begin_id,
                "context_end": int(trajectory_id), 
                "target": output
            })

    return task_info_list

def get_risk_prediction_task(task_list, trajectory_id, patient_trajectory_list):
    if len(task_list) == 0:
        return []

    event_name = patient_trajectory_list[trajectory_id]["file_name"]

    if event_name == "edstays":
        task_info_list = edstays_task_info_extraction(task_list, trajectory_id, patient_trajectory_list)
    elif event_name == "admissions":
        task_info_list = admissions_task_info_extraction(task_list, trajectory_id, patient_trajectory_list)
    elif event_name == "icustays":
        task_info_list = icustays_task_info_extraction(task_list, trajectory_id, patient_trajectory_list)
    elif event_name == "discharge":
        task_info_list = discharge_task_info_extraction(task_list, trajectory_id, patient_trajectory_list)
    else:
        raise NotImplementedError(f"""risk prediction event {event_name} not in ["edstays", "admissions", "icustays", "discharge"]""")

    return task_info_list

def ehr_anslysis(args, subject_id):
    # found event task
    event_task = obtain_event_task(args, DATASET.task_info)

    # load patient trajectory
    patient_ehr = f"""{args.ehr_dir}/{subject_id}.parquet"""
    patient_trajectory_list = read_parquet(patient_ehr)

    task_info_list = []
    discharge_event_id_list = []
    admissions_event_id_list = []
    for trajectory_id, item in (enumerate(patient_trajectory_list)):
        if item["file_name"] == "admissions":
            admissions_event_id_list.append(trajectory_id)
        if item["file_name"] == "discharge":
            discharge_event_id_list.append(trajectory_id)

        # get event basic info
        event_name = item["file_name"]
        if event_name in event_task or "any" in event_task:
            event_task_info_list = []
            event_task_info_list += get_decision_making_task(event_task[event_name]["decision_making"], trajectory_id, patient_trajectory_list)
            event_task_info_list += get_risk_prediction_task(event_task[event_name]["risk_prediction"], trajectory_id, patient_trajectory_list)
            
            for task_info in event_task_info_list:
                if task_info:
                    # if have admission with same hadm id, add admissions event
                    if len(admissions_event_id_list) > 0 and task_info["hadm_id"] == patient_trajectory_list[admissions_event_id_list[-1]]["hadm_id"]:
                        task_info["admissions_id"] = admissions_event_id_list[-1]
                    # if have last discharge (with different hadm id), add discharge event
                    if len(discharge_event_id_list) > 0 and task_info["hadm_id"] != patient_trajectory_list[discharge_event_id_list[-1]]["hadm_id"]:
                        task_info["last_discharge_id"] = discharge_event_id_list[-1]
                    
                    # task_info["target"] = json.dumps(task_info["target"])
                    task_info_list.append(task_info)

    if args.group == "patient":
        df = pd.DataFrame(task_info_list)
        df.to_csv(os.path.join(args.output_path, f"{subject_id}.csv"))

    return task_info_list

def obtain_patients_id(args):
    patients_file = os.path.join(args.patient_ids_path)
    df = pd.read_csv(patients_file)
    patient_ids = df["subject_id"].tolist()
    patient_ids = [str(id) for id in patient_ids]
    
    print(f"Get {len(patient_ids)} patients from {args.patient_ids_path}...")
    df = pd.read_parquet("/mnt3/longquan.lys/projects/EHRL/mimiciv/cache/event_static.parquet")
    df_filtered = df[df['subject_id'].isin(patient_ids)]
    filter_data = df_filtered[(df_filtered['event_num'] >= args.traj_len_min) & (df_filtered['event_num'] <= args.traj_len_max)]['subject_id'].tolist()

    print(f"Retain {len(filter_data)} patients with trajectory length in [{args.traj_len_min}, {args.traj_len_max}]...")
    return filter_data

def obtain_event_task(args, task_info):
    event_task = {} # {event: [task1, task2, task3, ...]}

    if args.task is not None:
        task_info = {k:v for k,v in task_info.items() if k in args.task}

    for task in task_info:
        assert "event" in task_info[task] or task_info[task]["task_type"] == "decision_making"
        event = task_info[task]["event"] if "event" in task_info[task] else task
        task_type = task_info[task]["task_type"]
        
        if event == "any":
            for event in DATASET.convertor.event_info:
                if event != "patient":
                    if event not in event_task:
                        event_task[event] = {"decision_making": [], "risk_prediction": []}
                    event_task[event][task_type].append(task)
        else:
            if event not in event_task:
                event_task[event] = {"decision_making": [], "risk_prediction": []}
            event_task[event][task_type].append(task)
    
    # print("Get event_task info: ")
    # print(event_task)
    return event_task

def parse_args():

    def str_list(value):
        return value.split(",")

    parser = argparse.ArgumentParser(prog="EHR Data Filter and Selection")

    # basic args
    parser.add_argument("--patient_ids_path", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument("--ehr_dir", type=str, default="./datas/patients_ehr")
    parser.add_argument("--group", choices=["patient", "task"], default="task")
    parser.add_argument("--task", type=str_list, default=None)

    parser.add_argument("--traj_len_min", type=int, default=1)
    parser.add_argument("--traj_len_max", type=int, default=1200)

    args = parser.parse_args()

    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)

    return args
    
def get_sample_weight(sample, task_target_info):
    if isinstance(sample["target"], list):
        avg_freq = sum([task_target_info[sample["task"]][target] for target in sample["target"]]) / len(sample["target"])
    elif sample["task"] != "radiology":
        avg_freq = task_target_info[sample["task"]][sample["target"]]
    else:
        avg_freq = 1

    return 1 / avg_freq

if __name__ == "__main__":

    args = parse_args()

    # get patient id
    patients_id = obtain_patients_id(args)

    # load task_info
    # all_task_info = []
    # for subject_id in tqdm(patients_id):
    #    all_task_info.append(ehr_anslysis(args, subject_id))
    if args.group == "patient":
        patients_id =[subject_id for subject_id in tqdm(patients_id) if not os.path.exists(os.path.join(args.output_path, f"{subject_id}.csv"))]
        Parallel(n_jobs=-1, backend='multiprocessing')(delayed(ehr_anslysis)(args, subject_id) for subject_id in tqdm(patients_id))
    
    elif args.group == "task":
        all_task_info = Parallel(n_jobs=-1, backend='multiprocessing')(delayed(ehr_anslysis)(args, subject_id) for subject_id in tqdm(patients_id))

        # task_split
        print("Recognize data into different task...")
        task_sample_info = {}
        task_target_info = {}
        for patient_task_info in all_task_info:
            for task_info in patient_task_info:
                if task_info["task"] not in task_sample_info:
                    task_sample_info[task_info["task"]] = []
                if task_info["task"] not in task_target_info:
                    task_target_info[task_info["task"]] = {}

                # log target info
                if isinstance(task_info["target"], list):
                    for target in task_info["target"]:
                        if target not in task_target_info[task_info["task"]]:
                            task_target_info[task_info["task"]][target] = 0
                        task_target_info[task_info["task"]][target] += 1
                elif task_info["task"] != "radiology":
                    target = task_info["target"]
                    if target not in task_target_info[task_info["task"]]:
                        task_target_info[task_info["task"]][target] = 0
                    task_target_info[task_info["task"]][target] += 1

                # log sample info
                task_sample_info[task_info["task"]].append(task_info)
        
        print("Begin get the sampling weight according to the target frequency...")
        for event, sample_list in task_sample_info.items():
            for sample in sample_list:
                sample["target_weight"] = get_sample_weight(sample, task_target_info)
                sample["target"] = json.dumps(sample["target"])
            
            df = pd.DataFrame(sample_list)
            df.to_csv(os.path.join(args.output_path, f"{event}.csv"), index=False)
        
        print({k:len(v)for k,v in task_sample_info.items()})
        print(task_target_info["next_event"])
    



        


