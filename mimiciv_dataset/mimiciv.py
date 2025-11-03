import os 
import json
from datetime import datetime
from torch.utils.data import Dataset, DataLoader
from functools import *
import pandas as pd
import pyarrow.parquet as pq
import random
import csv
from joblib import Parallel, delayed
from mimiciv_dataset.input_format import MIMICIVStringConvertor, safe_read

from tqdm import tqdm

def read_parquet(parquet_dir):
    # table = pq.read_table(parquet_dir)
    # df = table.to_pandas()
    df = pd.read_parquet(parquet_dir)
    # df['items'] = df['items'].apply(lambda x: json.loads(x.replace("\'", "\"").replace("nan", "null")))
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
    datetime1 = datetime.strptime(time1, format)
    datetime2 = datetime.strptime(time2, format)
    delta = datetime2 - datetime1
    return delta.total_seconds() / 3600

class MIMICIV(Dataset):
    def __init__(
        self,
        root_dir="./datas",
        sample_info_path=None,
        sample_info=None,
        lazzy_mode=False,
        log=True,
        shuffle=True,
    ):  
        ## load data
        random.seed(42)
        self.origin_data_dir = os.path.join(root_dir, "index_mapping") # original data dir, with hosp, ed, icu folders
        self.ehr_dir = os.path.join(root_dir, "patients_ehr") # path to preprocess patient parquet file
        self.cache_dir = os.path.join(root_dir, "cache") # path to store cache file

        self.sample_cache_dir = os.path.join(self.cache_dir, "sample")

        self.sample_info_path = sample_info_path  # data.csv file
        self.sample_info = sample_info  # directly subject id, usually use in testing
        assert self.sample_info_path or self.sample_info

        self.lazzy_mode = lazzy_mode # load data on the fly when set to `True`, otherwise load all data to memory (require lots of memories).
        self.log = log # print log or not

        os.makedirs(self.cache_dir, exist_ok=True)
        os.makedirs(self.sample_cache_dir, exist_ok=True)

        self.convertor = MIMICIVStringConvertor(
            origin_data_dir=self.origin_data_dir,
            cache_dir=self.cache_dir,
        )

        if self.sample_info is None:
            if self.sample_info_path is None:
                self.sample_info = []
            
            elif self.smaple_info.path.endswith(".csv"):
                df = pd.read_csv(self.sample_info_path)
                self.sample_info = df.to_dict(orient = 'records')
            
            elif self.sample_info_path.endswith(".jsonl"):
                with open(self.sample_info_path, "r") as f:
                    self.sample_info = [json.loads(line) for line in f.readlines()]
            
            else:
                raise NotImplementedError

        else:
            if isinstance(self.sample_info[0], str) or isinstance(self.sample_info[0], int):
                self.sample_info = [{"subject_id": id} for id in self.sample_info]

        if shuffle:
            random.shuffle(self.sample_info) == len(self.sample_info)

        self.load_cache = False
        self.load_sample_cache()
        self.patient_trajectory_dict = self.load_trajectory()
        self.similar_item = {}
        
        if self.log:
            print(len(self.sample_info))

        self.task_info = {
            "admissions": {
                "target_key": "admission_type",
                "metric": "em",
                "bid_event": [],
                "task_type": "decision_making",
                "instruction": "Given the sequence of events that have occurred in a hospital, please give the next Admissions suggestion for the patients.",
            },
            "omr": {
                "target_key": "result_name",
                "metric": "em",
                "bid_event": [],
                "task_type": "decision_making",
                "instruction": "Given the sequence of events that have occurred in a hospital, please give the next Online Medical Record suggestion for the patients.",
            },
            "procedures_icd": {
                "target_key": "procedures",
                "metric": "recall",
                "bid_event": [],
                "task_type": "decision_making",
                "instruction": "Given the sequence of events that have occurred in a hospital, please give the next Procedures International Classification of Diseases Item suggestion for the patients.",
            },
            "procedures_ccs": {
                "target_key": "CCS Type",
                "event": "procedures_icd",
                "metric": "recall",
                "bid_event": [],
                "task_type": "decision_making",
                "instruction": "Given the sequence of events that have occurred in a hospital, please give the next Procedures Clinical Classifications Software Item suggestion for the patients.",
            },
            "diagnoses_icd": {
                "target_key": "diagnoses",
                "metric": "recall",
                "bid_event": ["discharge"],
                "task_type": "decision_making",
                "instruction": "Given the sequence of events that have occurred in a hospital, please give the next Diagnoses International Classification of Diseases Item suggestion for the patients.",
                # "instruction": "Given the sequence of events that have occurred in a hospital, please give the next Diagnoses suggestion for the patients.",
            },
            "diagnoses_ccs": {
                "target_key": "CCS Type",
                "event": "diagnoses_icd",
                "metric": "recall",
                "bid_event": ["discharge"],
                "task_type": "decision_making",
                "instruction": "Given the sequence of events that have occurred in a hospital, please give the next Diagnoses Clinical Classifications Software Item suggestion for the patients.",
            },
            "labevents": {
                "target_key": "item_name",
                "metric": "em",
                "bid_event": [],
                "task_type": "decision_making",
                "instruction": "Given the sequence of events that have occurred in a hospital, please give the next Labotary Test Events suggestion for the patients.",
            },
            "microbiologyevents": {
                "target_key": "test_name",
                "metric": "em",
                "bid_event": [],
                "task_type": "decision_making",
                "instruction": "Given the sequence of events that have occurred in a hospital, please give the next Microbiology Test Events suggestion for the patients.",
            },
            "services": {
                "target_key": "curr_service",
                "metric": "em",
                "bid_event": [],
                "task_type": "decision_making",
                "instruction": "Given the sequence of events that have occurred in a hospital, please give the next Services suggestion for the patients.",
            },
            "transfers": {
                "target_key": "eventtype",
                "metric": "em",
                "bid_event": [],
                "task_type": "decision_making",
                "instruction": "Given the sequence of events that have occurred in a hospital, please give the next Transfers suggestion for the patients.",
            },
            "emar": {
                "target_key": "medication",
                "metric": "em",
                "bid_event": [],
                "task_type": "decision_making",
                "instruction": "Given the sequence of events that have occurred in a hospital, please give the next Electronic Medicine Administration Record suggestion for the patients.",
            },
            "poe": {
                "target_key": "order_type",
                "metric": "em",
                "bid_event": [],
                "task_type": "decision_making",
                "instruction": "Given the sequence of events that have occurred in a hospital, please give the next Provider Order Entry suggestion for the patients.",
            },
            # "pharmacy": {
            #     "target_key": "medication",
            #     "metric": "em",
            #     "bid_event": [],
            #     "task_type": "decision_making",
            #     "instruction": "Given the sequence of events that have occurred in a hospital, please give the next Pharmacy suggestion for the patients.",
            # },
            "prescriptions": {
                "target_key": "drug",
                "metric": "recall",
                "bid_event": [],
                "task_type": "decision_making",
                "instruction": "Given the sequence of events that have occurred in a hospital, please give the next Prescriptions suggestion for the patients.",
            },
            "prescriptions_atc": {
                "target_key": "ATC Type",
                "event": "prescriptions",
                "metric": "recall",
                "bid_event": [],
                "task_type": "decision_making",
                "instruction": "Given the sequence of events that have occurred in a hospital, please give the next Anatomical Therapeutic Chemical Classification Prescriptions suggestion for the patients.",
            },
            # note
            "radiology": {
                "target_key": "exam_name",
                "metric": "em",
                "bid_event": [],
                "task_type": "decision_making",
                "instruction": "Given the sequence of events that have occurred in a hospital, please give the next Radiology Examinations suggestion for the patients.",
            },
            "discharge": {
                "target_key": "text",
                "metric": "rouge",
                "bid_event": [],
                "task_type": "decision_making",
                "instruction": "Given the sequence of events that have occurred in a hospital, please give the next Discharge Report suggestion for the patients.",
            },
            # ed 
            "medrecon": {
                "target_key": "name",
                "metric": "recall",
                "bid_event": [],
                "task_type": "decision_making",
                "instruction": "Given the sequence of events that have occurred in a hospital, please give the next ED Medrecon suggestion for the patients.",
            },
            "medrecon_atc": {
                "target_key": "ATC Type",
                "event": "medrecon",
                "metric": "recall",
                "bid_event": [],
                "task_type": "decision_making",
                "instruction": "Given the sequence of events that have occurred in a hospital, please give the next ED Medrecon on Anatomical Therapeutic Chemical (ATC) Classification suggestion for the patients.",
            },
            "pyxis": {
                "target_key": "name",
                "metric": "em",
                "bid_event": [],
                "task_type": "decision_making",
                "instruction": "Given the sequence of events that have occurred in a hospital, please give the next ED Pyxis suggestion for the patients.",
            },
            "diagnosis": {
                "target_key": "icd_title",
                "metric": "recall",
                "bid_event": ["discharge"],
                "task_type": "decision_making",
                "instruction": "Given the sequence of events that have occurred in a hospital, please give the next ED Diagnoses on International Classification of Diseases suggestion for the patients.",
            },
            "diagnosis_ccs": {
                "target_key": "CCS Type",
                "event": "diagnosis",
                "metric": "recall",
                "bid_event": ["discharge"],
                "task_type": "decision_making",
                "instruction": "Given the sequence of events that have occurred in a hospital, please give the next ED Diagnoses on Clinical Classifications Software Item suggestion for the patients.",
            },
            
            # icu
            "chartevents": {
                "target_key": "item_name",
                "metric": "em",
                "bid_event": [],
                "task_type": "decision_making",
                "instruction": "Given the sequence of events that have occurred in a hospital, please give the next Chart Events suggestion for the patients.",
            },
            "ingredientevents": {
                "target_key": "item_name",
                "metric": "em",
                "bid_event": [],
                "task_type": "decision_making",
                "instruction": "Given the sequence of events that have occurred in a hospital, please give the next Ingredient Events suggestion for the patients.",
            },
            "datetimeevents": {
                "target_key": "item_name",
                "metric": "em",
                "bid_event": [],
                "task_type": "decision_making",
                "instruction": "Given the sequence of events that have occurred in a hospital, please give the next Datetime Events suggestion for the patients.",
            },
            "procedureevents": {
                "target_key": "item_name",
                "metric": "em",
                "bid_event": [],
                "task_type": "decision_making",
                "instruction": "Given the sequence of events that have occurred in a hospital, please give the next Procedure Events suggestion for the patients.",
            },
            "inputevents": {
                "target_key": "item_name",
                "metric": "em",
                "bid_event": [],
                "task_type": "decision_making",
                "instruction": "Given the sequence of events that have occurred in a hospital, please give the next Input Events suggestion for the patients.",
            },
            "outputevents": {
                "target_key": "item_name",
                "metric": "em",
                "bid_event": [],
                "task_type": "decision_making",
                "instruction": "Given the sequence of events that have occurred in a hospital, please give the next Output Events suggestion for the patients.",
            },

            ## next event prediction
            "next_event": {
                "target_key": "file_name",
                "event": "any",
                "metric": "acc",
                "bid_event": [],
                "task_type": "decision_making",
                "instruction": "Given the sequence of events that have occurred in a hospital, please give the next event suggestion for the patients.",
            },

            ## risk prediction
            "ED_Hospitalization": {
                "target_key": None,
                "event": "edstays",
                "metric": "aucroc",
                "bid_event": [],
                "task_type": "risk_prediction",
                "instruction": "Given the sequence of events that have occurred in a hospital, please predict whether the patient will be hospitalized after the emergency room visit.",
            },
            "ED_Inpatient_Mortality": {
                "target_key": None,
                "event": "edstays",
                "metric": "aucroc",
                "bid_event": [],
                "task_type": "risk_prediction",
                "instruction": "Given the sequence of events that have occurred in a hospital, please predict whether the patient will die during hospitalization.",
            },
            "ED_ICU_Tranfer_12hour": {
                "target_key": None,
                "event": "edstays",
                "metric": "aucroc",
                "bid_event": [],
                "task_type": "risk_prediction",
                "instruction": "Given the sequence of events that have occurred in a hospital, please predict whether the patient will be transferred to the ICU within 12 hours after the emergency room.",
            },
            "ED_Reattendance_3day": {
                "target_key": None,
                "event": "edstays",
                "metric": "aucroc",
                "bid_event": [],
                "task_type": "risk_prediction",
                "instruction": "Given the sequence of events that have occurred in a hospital, please predict whether the patient will return to the emergency department within 72 hours after the emergency visit.",
            },
            "ED_Critical_Outcomes": {
                "target_key": None,
                "event": "edstays",
                "metric": "aucroc",
                "bid_event": [],
                "task_type": "risk_prediction",
                "instruction": "Given the sequence of events that have occurred in a hospital, please predict whether the patient will die during hospitalization or will be transferred to the ICU within 12 hours after the emergency room.",
            },
            "Readmission_30day": {
                "target_key": None,
                "event": "discharge",
                "metric": "aucroc",
                "bid_event": [],
                "task_type": "risk_prediction",
                "instruction": "Given the sequence of events that have occurred in a hospital, please predict whether the patient will be readmitted to the hospital within 30 days",
            },
            "Readmission_60day": {
                "target_key": None,
                "event": "discharge",
                "metric": "aucroc",
                "bid_event": [],
                "task_type": "risk_prediction",
                "instruction": "Given the sequence of events that have occurred in a hospital, please predict whether the patient will be readmitted to the hospital within 60 days",
            },
            "Inpatient_Mortality": {
                "target_key": None,
                "event": "admissions",
                "metric": "aucroc",
                "bid_event": [],
                "task_type": "risk_prediction",
                "instruction": "Given the sequence of events that have occurred in a hospital, please predict whether the patient will die during hospitalization.",
            },
            "LengthOfStay_3day": {
                "target_key": None,
                "event": "admissions",
                "metric": "aucroc",
                "bid_event": [],
                "task_type": "risk_prediction",
                "instruction": "Given the sequence of events that have occurred in a hospital, please predict whether the patient's hospital stay will exceed 3 days.",
            },
            "LengthOfStay_7day": {
                "target_key": None,
                "event": "admissions",
                "metric": "aucroc",
                "bid_event": [],
                "task_type": "risk_prediction",
                "instruction": "Given the sequence of events that have occurred in a hospital, please predict whether the patient's hospital stay will exceed 7 days",
            },
            "ICU_Mortality_1day": {
                "target_key": None,
                "event": "icustays",
                "metric": "aucroc",
                "bid_event": [],
                "task_type": "risk_prediction",
                "instruction": "Given the sequence of events that have occurred in a hospital, please predict whether the patient will die within 1 day.",
            },
            "ICU_Mortality_2day": {
                "target_key": None,
                "event": "icustays",
                "metric": "aucroc",
                "bid_event": [],
                "task_type": "risk_prediction",
                "instruction": "Given the sequence of events that have occurred in a hospital, please predict whether the patient will die within 2 day.",
            },
            "ICU_Mortality_3day": {
                "target_key": None,
                "event": "icustays",
                "metric": "aucroc",
                "bid_event": [],
                "task_type": "risk_prediction",
                "instruction": "Given the sequence of events that have occurred in a hospital, please predict whether the patient will die within 3 day.",
            },
            "ICU_Mortality_7day": {
                "target_key": None,
                "event": "icustays",
                "metric": "aucroc",
                "bid_event": [],
                "task_type": "risk_prediction",
                "instruction": "Given the sequence of events that have occurred in a hospital, please predict whether the patient will die within 7 day.",
            },
            "ICU_Mortality_14day": {
                "target_key": None,
                "event": "icustays",
                "metric": "aucroc",
                "bid_event": [],
                "task_type": "risk_prediction",
                "instruction": "Given the sequence of events that have occurred in a hospital, please predict whether the patient will die within 14 day.",
            },
            "ICU_Stay_7day": {
                "target_key": None,
                "event": "icustays",
                "metric": "aucroc",
                "bid_event": [],
                "task_type": "risk_prediction",
                "instruction": "Given the sequence of events that have occurred in a hospital, please predict whether the patient will stay in the ICU for more than 7 days.",
            },
            "ICU_Stay_14day": {
                "target_key": None,
                "event": "icustays",
                "metric": "aucroc",
                "bid_event": [],
                "task_type": "risk_prediction",
                "instruction": "Given the sequence of events that have occurred in a hospital, please predict whether the patient will stay in the ICU for more than 14 days.",
            },
            "ICU_Readmission": {
                "target_key": None,
                "event": "icustays",
                "metric": "aucroc",
                "bid_event": [],
                "task_type": "risk_prediction",
                "instruction": "Given the sequence of events that have occurred in a hospital, please predict whether the patient will be admitted to the ICU again during this hospitalization",
            }
        }
    
    def save_sample_cache(self, samlpe_info):
        if self.sample_cache_file:
            with open(self.sample_cache_file, "a") as f:
                f.write(json.dumps(samlpe_info, ensure_ascii=False, separators=(',', ': ')) + "\n")
    
    def load_sample_cache(self):
        # load cache
        if not self.sample_info_path or self.lazzy_mode:
            self.sample_cache_file = None
            return

        samlpe_info_name = self.sample_info_path.rsplit("/", 1)[-1].rsplit(".")[0]
        self.sample_cache_file = os.path.join(self.sample_cache_dir, f"{samlpe_info_name}.jsonl")
        if os.path.exists(self.sample_cache_file):
            try:
                with open(self.sample_cache_file, "r") as f:
                    self.sample_info_cache = [json.loads(line) for line in f.readlines()]
            except:
                self.sample_info_cache = None
        else:
            self.sample_info_cache = None

        # detect whether to load the cache data
        if self.sample_info_cache != None and len(self.sample_info_cache) == len(self.sample_info):
            # begin check the data point
            self.load_cache = True
            print("Begin check the cache data...")
            for sample, sample_cache in tqdm(zip(self.sample_info, self.sample_info_cache)):
                if sample["task"] == sample_cache["task_info"]["task"] and sample["target"] == sample_cache["task_info"]["target"]:
                    continue
                else:
                    print("Loading the cache data failed, reloading from the data index file...")
                    os.remove(self.sample_cache_file)
                    self.load_cache = False
                    break

        if not self.load_cache:
            ## clean the cache file
            with open(self.sample_cache_file, "w") as f:
                pass
            

    def load_trajectory(self):
        if self.lazzy_mode or self.load_cache:
            return None

        print("Loding Patient Trajectory...")
        patient_ids = list(set([sample["subject_id"] for sample in self.sample_info]))
        # patient_trajectory_list = Parallel(n_jobs=-1)(delayed(read_parquet)(f"""{self.ehr_dir}/{subject_id}.parquet""") for subject_id in tqdm(patient_ids))
        patient_trajectory_list = [read_parquet(f"""{self.ehr_dir}/{subject_id}.parquet""") for subject_id in tqdm(patient_ids)]
        patient_trajectory_dict = {trajectory[0]["items"][0]["subject_id"]:trajectory for trajectory in patient_trajectory_list}
        print("Loding Patient Trajectory Finish!")
        return patient_trajectory_dict

    def __getitem__(self, idx):
        subject_id = str(self.sample_info[idx]["subject_id"])
        
        if self.load_cache:
            sample = self.sample_info_cache[idx]
        
        elif "input" in self.sample_info[0]:
            sample = self.sample_info[idx]

        else:
            if self.lazzy_mode:
                patient_trajectory_list = read_parquet(f"""{self.ehr_dir}/{subject_id}.parquet""")
            else:
                patient_trajectory_list = self.patient_trajectory_dict[subject_id]
            sample = self.process_cases(idx, self.sample_info[idx], patient_trajectory_list)

            if not self.load_cache and not self.lazzy_mode:
                self.save_sample_cache(sample)

        return sample
    
    def process_cases(self, idx, sample_info, patient_trajectory_list):
        # get context idx
        context_begin = sample_info.get("period_begin", 0)
        context_end = sample_info.get("period_end", len(patient_trajectory_list)-1)

        # get target event
        sample_info["task"] = sample_info["task"] if "task" in sample_info else sample_info["event"]
        task_name = sample_info["task"] 
        assert task_name in self.task_info.keys(), f"""Sample with task_name={task_name} not in predefined task_info: {list(self.task_info.keys())}"""
        assert ("target" in sample_info and sample_info["target"]) or self.task_info[task_name]["target_key"], f"""Sample with task_name={task_name} miss target information"""

        # preprocess context
        trajectory_events = [item for item in patient_trajectory_list[context_begin:context_end] if item["file_name"] not in self.task_info[task_name]["bid_event"] and item["file_name"] != "admissions"]
        context_input_text_list = [self.convertor.input_process(item) for item in trajectory_events]
        
        # add last discharge note (medical history)
        last_discharge_note = ""
        if safe_read(sample_info.get("last_discharge_id", None)):
            last_discharge_id = int(sample_info["last_discharge_id"])
            last_discharge_note = self.convertor.input_process(patient_trajectory_list[last_discharge_id])
        
        # add admissions infomation
        admission_text = ""
        if safe_read(sample_info.get("admissions_id", None)) and task_name != "admissions":
            admissions_id = int(sample_info["admissions_id"])
            admission_text = self.convertor.input_process(patient_trajectory_list[admissions_id])
        
        # add patient information
        patient_text = self.convertor.input_process(patient_trajectory_list[0])

        # concate the context
        context_input_text_list = [patient_text, last_discharge_note, admission_text] + context_input_text_list
        instruction = self.task_info[task_name]["instruction"]
        
        # process output
        if "target" in sample_info:
            try:
                output = eval(sample_info["target"])
            except:
                output = sample_info["target"]
        else:
            assert self.task_info[task_name]["task_type"] == "decision_making", f"""Risk Prediction Task {self.task_info[task_name]["task_type"]} requires pre-defined answer!"""
            output = self.convertor.output_process(task_name, patient_trajectory_list[context_end], target_key=self.task_info[task_name]["target_key"])

        # without target
        if not output:
            return {}

        # sample_info["target"] = output
        candidates = self.make_candidates(task_name, output)

        sample = {
            "idx": idx,
            "instruction": instruction,
            "input": "\n\n".join(context_input_text_list),
            "output": "\n".join(output) if isinstance(output, list) else output,
            "candidates": candidates,
            "task_info": self.task_info[task_name],
        }
        sample["task_info"].update(sample_info)
        sample["task_info"].update({"label": output})
        
        return sample
    
    def make_candidates(self, task_name, output):
        if self.task_info[task_name]["task_type"] == "risk_prediction":
            return ["yes", "no"]
        
        if task_name == "next_event_prediction":
            candidate_list = list(self.convertor.event_info.keys())
            return candidate_list

        if task_name not in self.similar_item:
            candidate_file = os.path.join(self.cache_dir, "similar_item", f"{task_name}.csv")
            if not os.path.exists(candidate_file):
                return None
        
            candidate_df = pd.read_csv(candidate_file)
            self.similar_item[task_name] = {row[0]: list(row[1:]) for _, row in candidate_df.iterrows()}
        
        total_candidate = self.similar_item[task_name]
        if len(total_candidate) > 100:
            candidate_list = []
            for label in output:
                candidate_list += total_candidate.get(label, [])
            
            candidate_list = random.sample(candidate_list, min(len(candidate_list), 100))
            candidate_list = list(set(candidate_list + output))
                
        else:
            candidate_list = list(total_candidate.keys())
        
        random.shuffle(candidate_list)
        return candidate_list
    
    def __len__(self):
        return len(self.sample_info)

def get_event(data):
  return data["task_info"]["event"]

if __name__ == "__main__":
    df = pd.read_csv("/dnn_training_sys/users/longquan.lys/datas/evidence_datas/evidence_reasoning/test_human_check_task_100_unweight_minsupport5_threshold5_topk20/final_check_know.csv")
    datas = df[df["uuid"] == "42c70539-b005-4b32-9e29-aa4171e9826d"].to_dict(orient="recird")
    # 创建一个数据集
    dataset = MIMICIV(
        root_dir="/mnt3/longquan.lys/projects/EHRL/mimiciv",
        # sample_info_path="/dnn_training_sys/users/longquan.lys/datas/data_index/decision_making/train_decision_making_100_unweight.csv",
        sample_info=datas,
        lazzy_mode=True,
    )

    for data in tqdm(dataset):
        print(data['input'])
