from collections import defaultdict
import pickle
import pandas as pd
# from src.mimiciv.input_format import MIMICIVCDMStringConverter, safe_read
from torch.utils.data import Dataset, DataLoader
import random 
import os 
from tqdm import tqdm
from mimiciv_dataset.input_format import safe_read

class MIMICIVCDM(Dataset):
    def __init__(
        self,
        root_dir="/dnn_training_sys/users/longquan.lys/MIMICIV-CMD/MIMIC-Clinical-Decision-Making-Dataset-chaoyi",
        sample_info_path="/dnn_training_sys/users/longquan.lys/datas/mimiciv_cdm/mimiciv_cdm_test.csv",
        sample_info=None,
        lazzy_mode=False,
        log=True,
        shuffle=True,
        candidate=False
    ):  
        super().__init__()
        ## load data
        random.seed(42)
        categories = ['appendicitis', 'cholecystitis', 'diverticulitis', 'pancreatitis']
        all_data = {}
        lab_test_mapping_dir = os.path.join(root_dir, "physionet.org/files/mimic-iv-ext-cdm/1.1/lab_test_mapping.pkl")
        with open(lab_test_mapping_dir, 'rb') as f:
            self.lab_test_mapping = pickle.load(f)  

        microbiology_test_mapping_dir = os.path.join(root_dir, "physionet.org/files/mimic-iv-ext-cdm/1.1/microbiology_test_mapping.pkl")
        with open(microbiology_test_mapping_dir, 'rb') as f:
            self.microbiology_test_mapping = pickle.load(f)  
        
        if sample_info is None:
            self.list_data = pd.read_csv(sample_info_path).to_dict(orient="records")
            # for category in categories:
                # path = f'{root_dir}/{category}_hadm_info_first_diag.pkl'
                # # with open(path, 'rb') as f:
                # #     cur_data = pickle.load(f)
                # #     for key, value in cur_data.items():
                # #         value['category'] = category
                # #     all_data.update(cur_data)
                # path = f'{sample_info_path}/{category}_test.csv'
                # self.list_data += pd.read_csv(path).to_dict(orient="records")

            # self.list_data = []
            # for key, value in all_data.items():
            #     self.list_data.append({**value, "hadm_id": key})

        else:
            self.list_data = sample_info
        
        
        self.task_info = {
            'mimiciv_cdm_diagnosis': {
                "instruction": "Given the sequence of events that have occurred in a hospital, please give the next Diagnoses in International Classification of Diseases Item suggestion for the patiens.",
            },
            'mimiciv_cdm_treatment': {
                "instruction": "Given the sequence of events that have occurred in a hospital, please give the next Procedures in International Classification of Diseases Item suggestion for the patiens.",
            },
        }
        self.new_data = self._preprocess_data()

        self.lazzy_mode = lazzy_mode # load data on the fly when set to `True`, otherwise load all data to memory (require lots of memories).
        self.log = log # print log or not
        self.candidate = candidate
        self.similar_item = {}

        if shuffle:
            random.seed(42)
            random.shuffle(self.new_data)
        
        if self.log:
            print(len(self.new_data))

    
    def _preprocess_data(self):
        new_data = []
        category_treatment_candidates = defaultdict(list)
        for index in tqdm(range(len(self.list_data))):
            cur_item = self.list_data[index]
            category = cur_item['category']
             
            item_str_list = []
            item_str_list.append("## Patient Demographics")
            item_str_list.append(f"- Patient History: {cur_item['Patient History']}")
            item_str_list.append(f"- Physical Examination: {cur_item['Physical Examination']}")


            item_str_list.append("\n")
            item_str_list.append("## Labotary Test Events")
            laboratory_tests = eval(str(cur_item['Laboratory Tests']))
            reference_range_lower = eval(cur_item['Reference Range Lower'].replace("nan", "None"))
            reference_range_upper = eval(cur_item['Reference Range Upper'].replace("nan", "None"))
            # "item_name", "valuenum", "valueuom", "ref_range_lower", "ref_range_upper"
            laboratory_keys = ['Item_name', "Valuenum", "Valueuom", "Ref_range_lower", "Ref_range_upper"]
            item_str_list.append(f"""| {" | ".join(laboratory_keys)} |""")
            item_str_list.append(f"""| {" | ".join(["------"] * len(laboratory_keys))} |""")
            for key, value in laboratory_tests.items():
                item_id = key
                laboratory_test_name = self.lab_test_mapping[item_id == self.lab_test_mapping['itemid']].iloc[0]['label']
                if len(value.split(" ")) == 2:
                    laboratory_test_result, laboratory_test_unit = value.split(" ")
                else:
                    laboratory_test_result = value
                    laboratory_test_unit = None
                refer_range_lower = reference_range_lower[item_id]
                refer_range_upper = reference_range_upper[item_id]
                item_str_list.append(f"""| {laboratory_test_name} | {laboratory_test_result} | {laboratory_test_unit} | {refer_range_lower} | {refer_range_upper} |""")


            microbiology_tests = safe_read(cur_item['Microbiology'])
            if microbiology_tests:
                microbiology_tests = eval(cur_item['Microbiology'])
                item_str_list.append("\n")
                item_str_list.append("## Microbiology Test Events")
                microbiology_keys = ['Item_name', "Valuestr"]
                item_str_list.append(f"""| {" | ".join(microbiology_keys)} |""")
                item_str_list.append(f"""| {" | ".join(["------"] * len(microbiology_keys))} |""")
                for key, value in microbiology_tests.items():
                    item_id = key
                    microbiology_test_name = self.microbiology_test_mapping[item_id]
                    microbiology_test_result = value 
                    item_str_list.append(f"""| {microbiology_test_name} | {microbiology_test_result} |""")

    
            item_str_list.append("\n")
            item_str_list.append("## Radiology Examinations")
            radiation_keys = ['Exam_name', 'Text']
            item_str_list.append(f"""| {" | ".join(radiation_keys)} |""")
            item_str_list.append(f"""| {" | ".join(["------"] * len(radiation_keys))} |""")
            radiation_tests = eval(cur_item['Radiology'])
            for each_test in radiation_tests:
                cur_exam_name = each_test['Exam Name']
                cur_exam_report = each_test['Report'].strip()
                item_str_list.append(f"""| {cur_exam_name} | {cur_exam_report} |""")
            # item_str = "\n".join(item_str_list)
            
            text = "\n".join(item_str_list)
            
            sample = {
                "input": text,
                "instruction": self.task_info['mimiciv_cdm_diagnosis']['instruction'],
                "output": eval(cur_item['ICD Diagnosis']),
                "task": f"{category}_diagnosis",
                "category": category,
            }
            new_data.append(sample)
#             if 'Procedures ICD9 Title' in cur_item or 'Procedures ICD10 Title' in cur_item:
#                 if safe_read(cur_item['Procedures ICD9 Title']):
#                     output = eval(safe_read(cur_item['Procedures ICD9 Title']))
                
#                 elif safe_read(cur_item['Procedures ICD10 Title']):
#                     output = eval(safe_read(cur_item['Procedures ICD10 Title'])
# )
#                 else:
#                     output = "No treatment information available."

#                 if output != 'No treatment information available.':
#                     sample = {
#                         "input": text,
#                         "instruction": self.task_info['mimiciv_cdm_treatment']['instruction'],
#                         "output": output,
#                         "task": f"{category}_treatment",
#                         "category": category,
#                     }
#                     # category_treatment_candidates[category].extend(sample['output'])
#                     new_data.append(sample)

        return new_data
    
    def make_candidates(self, output, task_name="diagnoses_icd"):
        if task_name not in self.similar_item:
            candidate_file = os.path.join("/mnt3/longquan.lys/projects/EHRL/mimiciv/cache", "similar_item", f"{task_name}.csv.csv")
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
            
    
    def __getitem__(self, index):
        cur_item = self.new_data[index]
        sample = {
            "idx": index,
            "instruction": cur_item['instruction'],
            "input": cur_item['input'],
            "output": "\n".join(cur_item['output']) if isinstance(cur_item['output'], list) else cur_item['output'],
            "candidates": self.make_candidates(cur_item['output']) if self.candidate else [],
            "task_info": {
                'label': cur_item['output'],
                "metric": "cdm_diagnosis" if "diagnosis" in cur_item["task"] else "em",
                "task": cur_item["task"],
                "task_type": "decision_making", 
                "category": cur_item["category"]
            }
        }
        return sample

    def __len__(self):
        return len(self.new_data)

if __name__ == "__main__":
    dataset = MIMICIVCDM(
        "/dnn_training_sys/users/longquan.lys/MIMICIV-CMD/MIMIC-Clinical-Decision-Making-Dataset-chaoyi",)
    print(len(dataset))
    for i in range(10):
        sample = dataset[i]
        print(sample['input'])
        print(sample['output'])
        print(sample['candidates'])
        print(sample['task_info'])
        print("=" * 50)