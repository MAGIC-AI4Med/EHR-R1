
import tqdm
import os
import json
import jsonlines
import pandas as pd
from joblib import Parallel, delayed
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor
from multiprocessing.dummy import Pool as ThreadPool
from multiprocessing import Pool, Lock


subject_dir = "{YOUR_PATH_TO_SAVE_PROCESS_DATA}/patients_sorted"
patients = os.listdir(subject_dir)
save_dir = "{YOUR_PATH_TO_SAVE_PROCESS_DATA}/patients_ehr"

time_field_map = {
    # admission
    "omr": "chartdate",
    "pharmacy": "entertime",
    "poe": "ordertime",
    "procedures_icd": "chartdate",
    "prescriptions": "starttime",
    "services": "transfertime",
    "labevents": "charttime",
    "microbiologyevents": "charttime",
    "admissions": "admittime",
    "transfers": "intime",
    "emar": "charttime",
    "hcpcsevents": "chartdate",

    # note
    "discharge": "charttime",
    "radiology": "charttime",
    
    # ed
    "edstays": "intime",
    "triage": "charttime",
    "medrecon": "charttime",
    "pyxis": "charttime",
    "vitalsign": "charttime",
    "diagnosis": "charttime",

    # icu
    "icustays": "intime",
    "chartevents": "charttime",
    "ingredientevents": "starttime",
    "datetimeevents": "charttime",
    "procedureevents": "starttime",
    "inputevents": "starttime",
    "outputevents": "charttime",
}

def save_jsonl(data_list, file_name):
    
    with jsonlines.open(file_name, mode='w') as writer:
        writer.write_all(data_list)

def save_parquet(data_list, file_name):
    for data in data_list:
        data["items"] = json.dumps(data["items"])
    df = pd.json_normalize(data_list)
    df.to_parquet(file_name, index=False)

def normalize_time_formate(time_str: str):
        try:
            # print(time_str)
            datetime.strptime(time_str, '%Y-%m-%d %H:%M:%S')
            return time_str
        except ValueError:
            try:
                datetime.strptime(time_str, '%Y-%m-%d')
                return time_str + ' 00:00:00'
            except:
                return None
        
        except TypeError:
            return None
        
def process_jsonl(file_path):
    data = []
    # print(file_path)
    # lock.acquire()
    # global subject_trajectories
    # lock.release()

    subject_id = file_path.split('/')[-2]
    with open(file_path, 'r') as file:
        for line in file:
            data.append(json.loads(line))

    if len(data) == 0:
        print(f"{file_path} is empty!")
        return
    # Convert the list to DataFrame
    # df = pd.DataFrame(data)

    organized_data = []
    #current_hadm_id = None
    current_file_name = None

    for index, row in enumerate(data):
        # print(row)
        #hadm_id = row['hadm_id']
        for key in row:
            if not isinstance(row[key], str):
                row[key] = str(row[key])

        file_name = row['file_name']
        try:
            hadm_id = row['hadm_id']
            if pd.isna(hadm_id):
                hadm_id = None
        except:
            hadm_id = None
        
        assert file_name != "note"
            
        if file_name != current_file_name:
            if index != 0:
                organized_data.append(item)
            current_file_name = file_name

            if file_name == "patients":
                item = {
                    "file_name": file_name,
                    "starttime": None,
                    "endtime": None,
                    "hadm_id": str(int(float(hadm_id))) if hadm_id and hadm_id != "nan" else None,
                    "items":[row],
                }

            elif file_name in {"diagnoses_icd", "drgcodes"}:
                item = {
                    "file_name": file_name,
                    "starttime": None,
                    "endtime": None,
                    "hadm_id": str(int(float(hadm_id))) if hadm_id and hadm_id != "nan" else None,
                    "items":[row],
                }

            else:
                item = {
                    "file_name": file_name,
                    "starttime": normalize_time_formate(row[time_field_map[file_name]]),
                    "endtime": normalize_time_formate(row[time_field_map[file_name]]),
                    "hadm_id": str(int(float(hadm_id))) if hadm_id and hadm_id != "nan" else None,
                    "items":[row],
                }
        else:
            item["items"].append(row)
            if item["file_name"] in {"diagnoses_icd", "drgcodes"}:
                pass
            else:
                current_time = normalize_time_formate(row[time_field_map[file_name]])
                if current_time:
                    if item["starttime"] is None:
                        item["starttime"] = current_time
                    
                    if item["endtime"] is None:
                        item["endtime"] = current_time
                    elif datetime.strptime(current_time, '%Y-%m-%d %H:%M:%S') > datetime.strptime(item["endtime"], '%Y-%m-%d %H:%M:%S'):
                        item["endtime"] = current_time

    organized_data.append(item)
    save_parquet(data_list = organized_data, file_name= save_dir + f"/{subject_id}.parquet")
    
        
# def process_subject(subject):
#     # if os.path.exists(save_dir + f"/{subject}.parquet"):
#     #     return
#     process_jsonl()
    
if __name__ == "__main__":

    # filtered_patients = [subject for subject in tqdm.tqdm(patients)]
    Parallel(n_jobs=-1, backend='multiprocessing')(delayed(process_jsonl)(subject_dir + '/' + subject + "/combined.jsonl") for subject in tqdm.tqdm(patients))
    # patients = patients[57800:]
    # for subject in tqdm.tqdm(patients):
    #     process_jsonl(subject_dir + '/' + subject + "/combined.jsonl")


    # subject_trajectories = {}
    # for result in results:
    #     subject_trajectories[result[0]["items"][0]["subject_id"]] = json.dumps(result)

    # save_parquet(subject_trajectories, file_name=save_dir+".parquet")
