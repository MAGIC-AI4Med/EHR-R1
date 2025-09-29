import os 
import pandas as pd
import json
import datetime
import jsonlines
import numpy

class MyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, datetime.datetime):
            print("MyEncoder-datetime.datetime")
            return obj.strftime("%Y-%m-%d %H:%M:%S")
        if isinstance(obj, bytes):
            return str(obj, encoding='utf-8')
        if isinstance(obj, int):
            return str(obj)
        if isinstance(obj, float):
            return str(obj)
        elif isinstance(obj, numpy.int64):
           return str(obj)
        else:
            return super(MyEncoder, self).default(obj)

subject_dict = {}       
def save_item(subject_id, source, item, root_dir):
    patient_dir = root_dir + str(subject_id)
    
    if not os.path.exists(patient_dir):
        try:
            os.mkdir(patient_dir)
        except:
            pass
    
    if os.path.exists(patient_dir+'/' +source+'.jsonl'):
        with jsonlines.open(patient_dir+'/' +source+'.jsonl',  'a') as f:
            f.write(item)
    else:
        with jsonlines.open(patient_dir+'/' +source+'.jsonl',  'w') as f:
            f.write(item)

def save_item_whole(source, item, index, root_dir):
    
    for subject_id in tqdm.tqdm(list(item.keys())):
        patient_dir = root_dir + str(subject_id)

        if not os.path.exists(patient_dir):
            try:
                os.mkdir(patient_dir)
            except:
                pass
    
        
        with open(patient_dir+'/' +source+'.jsonl',  'w') as f:
            for ii in item[subject_id]:
                ii = json.dumps(ii, cls=MyEncoder)
                f.write(ii)
                f.write("\n")

subject_dict = {}                          
def work(csv_path, source, file_name, index_csv):
    chunksize = 1000000
    chunks = pd.read_csv(csv_path, chunksize=chunksize)
    for items in tqdm.tqdm(chunks):
        if "subject_id" in list(items.keys()):
            print(file_name)
            for index in tqdm.tqdm(range(len(items))):
                sample = items.iloc[index].to_dict()
                sample['file_name'] = file_name
                subject_id = sample["subject_id"]

                try:
                    subject_dict[subject_id].append(sample)
                except:
                    subject_dict[subject_id] = [sample]
        else:
            break
    
outside_files = {
    "hosp": ["d_hcpcs.csv","d_icd_diagnoses.csv", "d_icd_procedures.csv", "d_labitems.csv", "poe_detail.csv", "emar_detail.csv","provider.csv"],
    "ed": ["discharge_detail.csv","radiology_detail.csv"],
    "icu": ["caregiver.csv","d_items.csv"],
    "note": ["discharge_detail.csv","radiology_detail.csv"]
}
        
import tqdm
root_path = "{YOUR_PATH_TO_MIMIC_IV}"
output_path = "{YOUR_PATH_TO_SAVE_PROCESS_DATA}/patients"
sources = ["hosp", "ed", "icu", "note"]

for source in sources:
    csv_list = os.listdir(root_path + '/' + source)
    index_csv = 0

    for csv_file in csv_list:
        #print(csv_file)
        if (".csv" not in csv_file) or (csv_file in outside_files[source]):
            continue
        else:
            work(csv_path = root_path + '/' + source+ '/' + csv_file, source = source, file_name = csv_file.replace(".csv",''), index_csv = index_csv)
            index_csv = index_csv+1

    save_item_whole(source=source, index=index_csv, item=subject_dict, root_dir=output_path)

# with open("/mnt/hwfile/medai/zhangxiaoman/DATA/MIMIC-IV/mimic-iv-note/2.2/note/discharge.csv",'r') as f:
#     discharge_notes = pd.read_csv(f)
# print(len(discharge_notes))

# for index in tqdm.tqdm(range(len(discharge_notes))):
#     sample = discharge_notes.iloc[index].to_dict()
#     discharge_note = {}
#     discharge_note['file_name'] = "discharge"
#     for key in list(sample.keys()):
#         discharge_note[key] = sample[key]
#     #print(discharge_note)
#     #input()
#     subject_id = discharge_note["subject_id"]
#     save_item(subject_id,source=source, item = discharge_note)
    

