import os 
import json
from datetime import datetime
from torch.utils.data import Dataset, DataLoader
from functools import *
import pandas as pd
import numpy as np
import pyarrow.parquet as pq
import random
import csv
import copy
from tqdm import tqdm
from collections import defaultdict
# from input_format import EHRSHOTStringConverter

ADDITIONAL_INFO = {
    "Body weight": {
        "unit": "oz",
        "ref_low": "350",
        "ref_high": "1000"
    },
    "Body height": {
        "unit": "inch",
        "ref_low": "5",
        "ref_high": "100"
    },
    "Body mass index / BMI": {
        "unit": "kg/m2",
        "ref_low": "18.5",
        "ref_high": "24.9"
    },
    "Body surface area": {
        "unit": "m2",
        "ref_low": "0.1",
        "ref_high": "10"
    },
    "Heart rate": {
        "unit": "bpm",
        "ref_low": "60",
        "ref_high": "100"
    },
    "Systolic blood pressure": {
        "unit": "mmHg",
        "ref_low": "90",
        "ref_high": "140"
    },
    "Diastolic blood pressure": {
        "unit": "mmHg",
        "ref_low": "60",
        "ref_high": "90"
    },
    "Body temperature": {
        "unit": "F",
        "ref_low": "95",
        "ref_high": "100.4"
    },
    "Respiratory rate": {
        "unit": "breaths/min",
        "ref_low": "12",
        "ref_high": "18"
    },
    "Oxygen saturation": {
        "unit": "%",
        "ref_low": "95",
        "ref_high": "100"
    },
    "Hemoglobin": {
        "unit": "g/dL",
        "ref_low": "12",
        "ref_high": "17"
    },
    "Hematocrit": {
        "unit": "%",
        "ref_low": "36",
        "ref_high": "51"
    },
    "Erythrocytes": {
        "unit": "10^6/uL",
        "ref_low": "4.2",
        "ref_high": "5.9"
    },
    "Leukocytes": {
        "unit": "10^6/uL",
        "ref_low": "4",
        "ref_high": "10"
    },
    "Platelets": {
        "unit": "10^3/uL",
        "ref_low": "150",
        "ref_high": "350"
    },
    "Sodium": {
        "unit": "mmol/L",
        "ref_low": "136",
        "ref_high": "145"
    },
    "Potassium": {
        "unit": "mmol/L",
        "ref_low": "3.5",
        "ref_high": "5.0"
    },
    "Chloride": {
        "unit": "mmol/L",
        "ref_low": "98",
        "ref_high": "106"
    },
    "Carbon dioxide, total": {
        "unit": "mmol/L",
        "ref_low": "23",
        "ref_high": "28"
    },
    "Calcium": {
        "unit": "mmol/L",
        "ref_low": "9",
        "ref_high": "10.5"
    },
    "Glucose": {
        "unit": "mmol/L",
        "ref_low": "70",
        "ref_high": "100"
    },
    "Urea nitrogen": {
        "unit": "mg/dL",
        "ref_low": "8",
        "ref_high": "20"
    },
    "Creatinine": {
        "unit": "mg/dL",
        "ref_low": "0.7",
        "ref_high": "1.3"
    },
    "Anion gap": {
        "unit": "mmol/L",
        "ref_low": "3",
        "ref_high": "11",
    }
}

AGGREGATED_MAPPING = {
    "LOINC/29463-7": "Body weight",
    "LOINC/8302-2": "Body height",
    " LOINC/39156-5": "Body mass index / BMI",
    "LOINC/8277-6": "Body surface area",
    "SNOMED/301898006": "Body surface area",

    "LOINC/8867-4": "Heart rate",
    "SNOMED/364075005": "Heart rate",
    "SNOMED/78564009": "Heart rate",
    "LOINC/8480-6": "Systolic blood pressure",
    "SNOMED/271649006": "Systolic blood pressure",
    "LOINC/8462-4": "Diastolic blood pressure",
    "SNOMED/271650006": "Diastolic blood pressure",
    "LOINC/8310-5": "Body temperature",
    "LOINC/9279-1": "Respiratory rate",
    "LOINC/LP21258-6": "Oxygen saturation",

    "LOINC/718-7": "Hemoglobin",
    "SNOMED/271026005": "Hemoglobin",
    "SNOMED/441689006": "Hemoglobin",
    "LOINC/4544-3": "Hematocrit",
    "LOINC/20570-8": "Hematocrit",
    "LOINC/48703-3": "Hematocrit",
    "SNOMED/28317006": "Hematocrit",

    "LOINC/789-8": "Erythrocytes",
    "LOINC/26453-1": "Erythrocytes",
    "LOINC/20584-9": "Leukocytes",
    "LOINC/6690-2": "Leukocytes",
    "LOINC/777-3": "Platelets",
    "SNOMED/61928009": "Platelets",
    "LOINC/2951-2": "Sodium",
    "LOINC/2947-0": "Sodium",
    "SNOMED/25197003": "Sodium",
    "LOINC/2823-3": "Potassium",
    "SNOMED/312468003": "Potassium",
    "LOINC/6298-4": "Potassium",
    "LOINC/2075-0": "Chloride",
    "SNOMED/104589004": "Chloride",
    "LOINC/2028-9": "Carbon dioxide, total",
    "LOINC/17861-6": "Calcium",
    "SNOMED/271240001": "Calcium",
    "LOINC/2345-7": "Glucose",
    "SNOMED/166900001": "Glucose",
    "LOINC/2339-0": "Glucose",
    "SNOMED/33747003": "Glucose",
    "LOINC/14749-6": "Glucose",
    "LOINC/3094-0": "Urea nitrogen",
    "SNOMED/105011006": "Urea nitrogen",
    "LOINC/2160-0": "Creatinine",
    "SNOMED/113075003": "Creatinine",
    "LOINC/33037-3": "Anion gap",
    "LOINC/41276-7": "Anion gap",
    "SNOMED/25469001": "Anion gap"
}

class EHRSHOTDataset(Dataset):
    def __init__(
        self,
        root_dir="/dnn_training_sys/users/longquan.lys/EHRSHOT/Shah_Lab_Four_EHR_Dataset/EHRSHOT/EHRSHOT_ASSETS",
        sample_info_path=None,
        sample_info=None,
        lazzy_mode=False,
        num_shot=-1,
        num_repeat=0,
        train=True,
        log=True,
        shot=-1,
        shuffle=True,
    ):  
        random.seed(42)
        self.task_info = {
            'guo_los': {
                'metric': 'aucroc',
                'task_type': 'risk_prediction',
                'instruction': f"""Given the sequence of events that have occurred in a hospital, please predict whether a patient's total length of stay during a visit to the hospital will be at least 7 days."""
            },
            'guo_readmission': {
                'metric': 'aucroc',
                'task_type': 'risk_prediction',
                'instruction': f"""Given the sequence of events that have occurred in a hospital, please predict whether a patient will be re-admitted to the hospital within 30 days after being discharged from a visit."""
            },
            'guo_icu': {
                'metric': 'aucroc',
                'task_type': 'risk_prediction',
                'instruction': f"""Given the sequence of events that have occurred in a hospital, please predict whether a patient will be transferred to the ICU during a visit to the hospital."""
            },
            'lab_anemia': {
                'metric': 'aucroc',
                'task_type': 'risk_prediction',
                'instruction': f"""Given the sequence of events that have occurred in a hospital, please predict whether an anemia lab comes back as normal (>=120 g/L)."""
            },
            'lab_hyperkalemia': {
                'metric': 'aucroc',
                'task_type': 'risk_prediction',
                'instruction': f"""Given the sequence of events that have occurred in a hospital, please predict whether a hyperkalemia lab comes back as normal(<=5.5 mmol/L). """ # mild (>5.5 and <=6mmol/L), moderate (>6 and <=7 mmol/L), or severe (>7 mmol/L).
            },
            'lab_hyponatremia': {
                'metric': 'aucroc',
                'task_type': 'risk_prediction',
                'instruction': f"""Given the sequence of events that have occurred in a hospital, please predict whether a hyponatremia lab comes back as normal (>=135 mmol/L).""" # mild (>=130 and <135 mmol/L), moderate (>=125 and <130 mmol/L), or severe (<125 mmol/L)
            },
            'lab_hypoglycemia': {
                'metric': 'aucroc',
                'task_type': 'risk_prediction',
                'instruction': f"""Given the sequence of events that have occurred in a hospital, please predict whether a hypoglycemia lab comes back as normal (>=3.9 mmol/L).""" # mild (>=3.5 and <3.9 mmol/L), moderate (>=3 and <3.5 mmol/L), or severe (<3 mmol/L).
            },
            'lab_thrombocytopenia': {
                'metric': 'aucroc',
                'task_type': 'risk_prediction',
                'instruction': f"""Given the sequence of events that have occurred in a hospital, please predict whether a thrombocytopenia lab comes back as normal (>=150 109/L).""" # mild (>=100 and <150 109/L), moderate (>=50 and <100 109/L), or severe (<50 109/L),
            },
            'new_acutemi': {
                'metric': 'aucroc',
                'task_type': 'risk_prediction',
                'instruction': f"""Given the sequence of events that have occurred in a hospital, please predict whether the patient will have her first diagnosis of an acute myocardial infarction within the next year."""
            },
            'new_celiac': {
                'metric': 'aucroc',
                'task_type': 'risk_prediction',
                'instruction': f"""Given the sequence of events that have occurred in a hospital, please predict whether the patient will have her first diagnosis of celiac disease within the next year."""
            },
            'new_hyperlipidemia': {
                'metric': 'aucroc',
                'task_type': 'risk_prediction',
                'instruction': f"""Given the sequence of events that have occurred in a hospital, please predict whether the patient will have her first diagnosis of hyperlipidemia within the next year."""
            },
            'new_hypertension': {
                'metric': 'aucroc',
                'task_type': 'risk_prediction',
                'instruction': f"""Given the sequence of events that have occurred in a hospital, please predict whether the patient will have her first diagnosis of essential hypertension within the next year."""
            },
            'new_lupus': {
                'metric': 'aucroc',
                'task_type': 'risk_prediction',
                'instruction': f"""Given the sequence of events that have occurred in a hospital, please predict whether the patient will have her first diagnosis of lupus within the next year."""
            },
            'new_pancan': {
                'metric': 'aucroc',
                'task_type': 'risk_prediction',
                'instruction': f"""Given the sequence of events that have occurred in a hospital, please predict whether the patient will have her first diagnosis of pancreatic cancer within the next year."""
            }
        }
        
        self.train = train
        self.root_dir = root_dir
        self.ehr_dir = os.path.join(root_dir, 'data', "patient_ehr")
        self.cache_dir = os.path.join(root_dir, "cache")
        
        self.sample_info_path = sample_info_path
        self.sample_info = sample_info
        assert self.sample_info or self.sample_info_path
        
        self.lazzy_mode = lazzy_mode # load data on the fly when set to `True`, otherwise load all data to memory (require lots of memories).
        self.log = log # print log or not
        
        os.makedirs(self.cache_dir, exist_ok=True)
        
        if self.sample_info is None:
            if self.sample_info_path is None:
                self.sample_info = []
            else:
                df = pd.read_csv(self.sample_info_path)
                self.sample_info = df.to_dict(orient = 'records')
   
        if train:
            self.sample_info = [sample for sample in self.sample_info if sample.get("split", "train") == 'train']
        else:
            self.sample_info = [sample for sample in self.sample_info if sample.get("split", "train") == 'val']
        
        if shuffle:
            random.shuffle(self.sample_info)
        
        self.get_code_mapping()
    
    def get_code_mapping(self):
        # code2token2description
        token_2_code_path = os.path.join(self.root_dir, 'models/clmbr/token_2_code.json')
        token_2_description_path = os.path.join(self.root_dir, 'models/clmbr/token_2_description.json')
        with open(token_2_code_path, 'r') as f:
            self.token_2_code = json.load(f)
            self.code_2_token = {v: k for k, v in self.token_2_code.items()}
        with open(token_2_description_path, 'r') as f:
            self.token_2_description = json.load(f)
            # self.description_2_token = {v: k for k, v in self.token_2_description.items()}
        self.code_2_description = {}
        for code, token in self.code_2_token.items():
            if token in self.token_2_description:
                self.code_2_description[code] = self.token_2_description[token]
        
        cpt4_code_2_description_path = os.path.join(self.root_dir, 'models/clmbr/cpt4_code.json')
        with open(cpt4_code_2_description_path, 'r') as f:
            self.cpt4_code_2_description = json.load(f)
        
        icd10pcs_code_2_description_path = os.path.join(self.root_dir, 'models/clmbr/icd10pcs.json')
        with open(icd10pcs_code_2_description_path, 'r') as f:
            self.icd10pcs_code_2_description = json.load(f)
    
    def gather_visit_info(self, group_data, event_str_list):
        if len(group_data) == 1:
            item = group_data[0]
            code = item.get('code', None)
            content = self.code_2_description.get(code, None)

            if "CPT4" in content:
                cpt4_code = content.split("/", 1)[-1]
                content = self.cpt4_code_2_description.get(cpt4_code, None)
                if content is None:
                    return event_str_list
            
            elif "ICD10PCS" in content:
                icd10pcs_code = content.split("/", 1)[-1]
                content = self.icd10pcs_code_2_description.get(icd10pcs_code, None)
                if content is None:
                    return event_str_list

            event_str_list.append(
                f"- {content}"
            )
        else:
            for item in group_data:
                code = item.get('code', None)
                content = self.code_2_description.get(code, None)
                if "CPT4" in content:
                    cpt4_code = content.split("/", 1)[-1]
                    content = self.cpt4_code_2_description.get(cpt4_code, None)
                    if content is None:
                        continue

                elif "ICD10PCS" in content:
                    icd10pcs_code = content.split("/", 1)[-1]
                    content = self.icd10pcs_code_2_description.get(icd10pcs_code, None)
                    if content is None:
                        continue

                if f"- {content}" not in event_str_list:
                    event_str_list.append(
                        f"- {content}"
                    )

        return event_str_list
    
    def gather_meansurement_info(self, group_data, event_str_list):
        info_keys = ['Item_Name', 'Valuenum', 'Valueuom', "Ref_Range_Lower", "Ref_Range_Upper", "Flag"]
        event_str_list.append(
            f"""| {" | ".join([key.title() for key in info_keys])} |"""
        )
        event_str_list.append(f"""| {" | ".join(["------"] * len(info_keys))} |""")
        event_list = []
        for item in group_data:
            row = []

            code = item.get('code', None)
            item_name = AGGREGATED_MAPPING.get(code, None)
            if item_name is None:
                item_name = self.code_2_description.get(code, None)
                if item_name is None:
                    continue
                else:
                    value = item.get('value', None)
                    if not isinstance(value, str):
                        continue
                    
                    unit = item.get('unit', None)
                    row = [item_name, str(value), str(unit), "nan", "nan", "nan"]
                    event_str_list.append(
                        f"""| {" | ".join(row)} |"""
                    )
                    event_list.append(item_name)
                    continue

            value = item.get('value', None)
            if not isinstance(value, str):
                continue

            item_info = ADDITIONAL_INFO[item_name]
            try:
                item_flag = "normal" if float(value) >= float(item_info["ref_low"]) and float(value) <= float(item_info["ref_high"]) else "abnormal"
            except:
                continue

            row = [item_name, value, item_info["unit"], item_info["ref_low"], item_info["ref_high"], item_flag]
            event_str_list.append(
                f"""| {" | ".join(row)} |"""
            )
            event_list.append(item_name)
        
        # if len(event_str_list) < 4:
        #     import pdb
        #     pdb.set_trace()

        # only keep the latest results
        latest_event_str_list = event_str_list[:3]
        for event_id, (event, event_str) in enumerate(zip(event_list, event_str_list[3:])):
            if event not in event_list[event_id+1:]:
                latest_event_str_list.append(event_str)

        event_str_list = latest_event_str_list
        return event_str_list
    
    def input_process(self, test_item):
        patient_id = test_item['patient_id']
        patient_path = os.path.join(self.ehr_dir, str(patient_id) + '.csv')
        patient_info = pd.read_csv(patient_path)
        patient_info['start'] = pd.to_datetime(patient_info['start'])
        patient_info['end'] = pd.to_datetime(patient_info['end'])
        # start_time = datetime.strptime(test_item['start'], '%Y-%m-%dT%H:%M:%S')
        # end_time = datetime.strptime(test_item['end'], '%Y-%m-%dT%H:%M:%S')
        # context_info = patient_info[(patient_info['start'] >= start_time) & (patient_info['start'] <= end_time)]
        person_info = patient_info[patient_info["omop_table"] == "person"]
        context_info = pd.concat([person_info, patient_info.iloc[test_item["period_begin"]:test_item["period_end"]+1]])   
        # group by omap_table 
        context_info = context_info.groupby('omop_table').apply(lambda x: x.to_dict(orient='records')).tolist()
        context_info = sorted(context_info, key=lambda x: x[0]['start'])

        item_str_mapping_list = []

        # for group_name, group_data in context_info:
        for group_data in context_info:
            event_str_list = []
            event_name = group_data[0]['omop_table']
            if event_name.lower() == "note":
                continue
            event_time = group_data[0]['start'].strftime('%Y-%m-%d %H:%M:%S')
            # item_str_mapping_list[event_name]
            title = f"## {event_name.title()} [{event_time}]"
            event_str_list.append(title)

            if event_name == "person":
                for item in group_data:
                    description = self.code_2_description.get(item.get("code", None), None)
                    event_str_list.append(f"- {description}")

            elif "drug" in event_name or "condition" in event_name or "procedure" in event_name:
                event_str_list = self.gather_visit_info(group_data, event_str_list)
            
            else:
                event_str_list = self.gather_meansurement_info(group_data, event_str_list)
                if len(event_str_list) < 4:
                    continue
                
            item_str_mapping_list.append("\n".join(event_str_list))

        text = "\n\n".join(item_str_mapping_list)
        return text 
            
    def __len__(self):
        return len(self.sample_info)
        
    def __getitem__(self, index):
        sample = self.sample_info[index]
        task_name = sample['task_name']

        context = self.input_process(sample)   
        task_info = {}
        task_info["task"] = task_name
        task_info["task_type"] = self.task_info[task_name]["task_type"]
        task_info["metric"] = self.task_info[task_name]["metric"]
        task_info["label"] = sample['label']

        if "lab_" in task_name:
            task_info["label"] = "yes" if task_info["label"] == "normal" else "no"
        
        output_sample = {
            "idx": index,
            "instruction": self.task_info[task_name]['instruction'],
            "input": context,
            "output": task_info["label"],
            "candidates": ["yes", "no"],
            "task_info": task_info
        } 
        return output_sample
        # num_repeat is to repeat the run exp, so that the results are robust across different data 

if __name__ == "__main__":
    dataset = EHRSHOTDataset(
        root_dir="/dnn_training_sys/users/longquan.lys/EHRSHOT/Shah_Lab_Four_EHR_Dataset/EHRSHOT/EHRSHOT_ASSETS",
        sample_info_path="/dnn_training_sys/users/longquan.lys/EHRSHOT/Shah_Lab_Four_EHR_Dataset/EHRSHOT/EHRSHOT_ASSETS/cache/index/ehrshot_test.csv",
        lazzy_mode=False,
        num_shot=1,
        num_repeat=0,
        train=True,
        log=True,
        shot=-1,
        shuffle=True
    )
    print(f"size of dataset: {len(dataset)}")
    for sample in tqdm(dataset):
        # sample = dataset[i]
        # print(f"sample {i}: {sample}")
        print(f"input: {sample['input']}")
        print(f"output: {sample['output']}")
        print(f"instruction: {sample['instruction']}")
        print(f"candidates: {sample['candidates']}")
        print(f"task_info: {sample['task_info']}")
        print("=" * 50)