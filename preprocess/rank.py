import os 
import pandas as pd
import json
from datetime import datetime, timedelta
import jsonlines
import math
import tqdm
from joblib import Parallel, delayed

ROOT_DIR = "{YOUR_PATH_TO_SAVE_PROCESS_DATA}/patients"
SUBJECTS = os.listdir(ROOT_DIR)

def sort_note(data_list):
    sorted_data_list = sorted(data_list, key=lambda x: (datetime.strptime(x['charttime'], '%Y-%m-%d %H:%M:%S'), x['note_seq']))
    return sorted_data_list

def sort_ed(data_list):
    file_name_order = {
        "edstays": 0,
        "triage": 1,
        "medrecon": 2,
        "pyxis": 3,
        "vitalsign": 4,
        "diagnosis": 5
    }
    
    def custom_sort(item):
        # 获取 stay_id
        stay_id = item['stay_id']
        # 获取 file_name 的排序值
        file_name_rank = file_name_order.get(item['file_name'], 999)  # 默认排序值999（如果不在定义的选项中）
        # 获取 charttime，只有在特定 file_name 下才考虑 charttime
        if item['file_name'] in {"medrecon", "pyxis", "vitalsign"} and 'charttime' in item:
            chart_time = datetime.strptime(item['charttime'], '%Y-%m-%d %H:%M:%S')
        elif item["file_name"] in {"edstays", "triage"}:
            chart_time = datetime.min  # 不排序的情况下给 None
        elif item['file_name'] in {"diagnosis"}:
            chart_time = datetime.max
        
        # 返回的排序键，None 在排序中被认为比任何值小
        return (stay_id, chart_time, file_name_rank)
    
    # 进行排序
    sorted_data_list = sorted(data_list, key=custom_sort)

    # auto add time for triage and diagnosis ed item. triage should follow the edstays and diagnosis should at the end of the ed.
    ed_stays_items = {data["stay_id"]:data for data in sorted_data_list if data["file_name"] == "edstays"}
    for data in sorted_data_list:
        if data["file_name"] == "triage":
            data["charttime"] = (datetime.strptime(ed_stays_items[data["stay_id"]]["intime"], '%Y-%m-%d %H:%M:%S') + timedelta(seconds=1)).strftime('%Y-%m-%d %H:%M:%S')
        
        elif data["file_name"] == "diagnosis":
            data["charttime"] = ed_stays_items[data["stay_id"]]["outtime"]
    
    return sorted_data_list

def sort_icu(data_list):
    time_field_map = {
        "chartevents": "charttime",
        "ingredientevents": "starttime",
        "datetimeevents": "charttime",
        "procedureevents": "starttime",
        "inputevents": "starttime",
        "outputevents": "charttime",
        "icustays": None  # 不参与时间排序
    }
    
    def get_time(item):
        # 确定使用的时间字段
        file_name = item['file_name']
        time_field = time_field_map.get(file_name, 'charttime')
        # 如果时间字段是 None，不参与排序
        if time_field is None:
            return None
        # 获取时间字段的值，如果字段不存在，返回 None
        time_str = item.get(time_field)
        if time_str:
            return datetime.strptime(time_str, '%Y-%m-%d %H:%M:%S')
        else:
            return None
    
    # 自定义排序函数
    def custom_sort(item):
        # 获取 hadm_id 和 stay_id
        hadm_id = item.get('hadm_id', 0)
        stay_id = item.get('stay_id', 0)
        # 是否为 icustays，用于特定排序规则
        is_icustays = 1 if item['file_name'] == "icustays" else 0
        # 获取时间字段的值，如果是 icustays，则不参与时间排序
        time_value = get_time(item)
        
        # 返回的排序键
        return (hadm_id, stay_id, -is_icustays, time_value)
    
    # 进行排序
    sorted_data_list = sorted(data_list, key=custom_sort)
    return sorted_data_list

def sort_hosp(data_list):
        # 最后我希望你对另一个datalist进行排序，还是在filename里包含如下文件名
        # "patients"，"omr"，"pharmacy"，"poe"，"procedures_icd"，"prescriptions"，"services"，"labevents"，"microbiologyevents"，"admissions"，"transfers"， "emar"，"diagnoses_icd"，"drgcodes"
        # 总的来说我希望你先按"hadm_id"排序再按时间排序，接下我将仔细的定义时间：
        # 对于patients而言 请始终保持其最顶，无须进行任何排序
        # 对于omr，chartdate key记录了他的记录时间，请注意它不具有"hadm_id"这个key，它需要按照时间插入所有的参与排序的item之间
        # 对于pharmacy然后"starttime"记录了它对应的时间
        # 对于poe然后"ordertime"记录了它对应的时间
        # 对于procedures_icd然后"chartdate"记录了它对应的时间
        # 对于prescriptions然后"starttime"记录了它对应的时间
        # 对于services然后"transfertime"记录了它对应的时间
        # 对于labevents然后"charttime"记录了它对应的时间
        # 对于labevents然后"charttime"记录了它对应的时间，请注意，对于labevents会存在hadm_id 为NaN的情况，此时它和omr一样 需要按照时间插入所有的参与排序的item之间
        # 对于microbiologyevents，"charttime"记录了它对应的时间，请注意，同样的会存在hadm_id 为NaN的情况，此时它和omr一样 需要按照时间插入所有的参与排序的item之间
        # 对于admissions，"admittime"记录了它对应的时间，请无视时间将它放在每一个相同"hadm_id"的最前面
        # 对于transfers，"intime"记录了它对应的时间
        # 对于emar，"charttime"记录了它对应的时间，请注意，同样的会存在hadm_id 为NaN的情况，此时它和omr一样 需要按照时间插入所有的参与排序的item之间
        # 对于diagnoses_icd，请按照"hadm_id" 放到相同的hadm_id最后面
        # 对于drgcodes，请按照"hadm_id" 放到相同的hadm_id最后面 比diagnoses_icd 还要后面。
        
        # 请注意，任何时候若出现了hadm_id 为NaN的情况，这条数据就需要按照时间插入所有的参与排序的item之间
    
    # 定义时间字段选择的映射
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
    
    # no_time_types = ["diagnoses_icd", "drgcodes"]
    
    def get_time(item):
        # 确定使用的时间字段
        file_name = item['file_name']
        time_field = time_field_map[file_name]
        
        
        time_str = item[time_field]  
            
        # 特殊处理 omr 和 procedures_icd 时间，只有日期，没有具体时间
        if file_name in {"omr", "procedures_icd"}:
            return datetime.strptime(time_str + " 00:00:00", '%Y-%m-%d %H:%M:%S')# 表示当天最前
        else:
            return datetime.strptime(time_str, '%Y-%m-%d %H:%M:%S')
    
    # 按时间排序的列表
    time_sorted = []
    # 没有时间但需要特殊处理的元素
    no_time_elements = []
    # 患者类型
    patients = []
    
    for item in data_list:
        file_type = item.get("file_name")
        if file_type == "patients":
            patients.append(item)
        else:
            try: 
                time = get_time(item)
                time_sorted.append(item)
            except:
                # print(item)
                # input()
                no_time_elements.append(item)
    
    time_sorted =  sorted(time_sorted, key=get_time)     
    #print(no_time_elements)
    for element in no_time_elements:
        hadm_id = element.get("hadm_id")
        element_type = element.get("file_name")
        inserted = False

        final_position = 0
        for i, item in enumerate(time_sorted):
            if item.get("hadm_id") == hadm_id:
                if element_type == "diagnoses_icd" and item.get("file_name") != "drgcodes":
                    final_position = i

                elif element_type == "drgcodes":
                    final_position = i
                
                else:
                    pass

        time_sorted.insert(final_position+1, element)
        inserted = final_position > 0

        if not inserted:
            # 如果未找到相同的 hadm_id，则放在列表末尾
            time_sorted.append(element)
    
    # 将 patients 类型的元素放在最顶部
    final_sorted_list = patients + time_sorted
    
    return final_sorted_list

def my_merge(result_data_list, stay_data_list,file_name):
    stay_dict = {}
    current_hadm_id = None
    current_stay_id = None
    former_stay_id= 0
    for entry in stay_data_list:
        current_stay_id = entry['stay_id']
        if current_stay_id != former_stay_id:
            current_hadm_id = entry["hadm_id"]
            stay_dict[current_hadm_id] = {}
            stay_dict[current_hadm_id]["file_name"] = file_name
            stay_dict[current_hadm_id]["hadm_id"] = entry["hadm_id"]
            stay_dict[current_hadm_id]["stay_id"] = current_stay_id
            stay_dict[current_hadm_id]["intime"] = entry["intime"]
            stay_dict[current_hadm_id]["outtime"] = entry["outtime"]
            stay_dict[current_hadm_id]["root_item"] = entry
            stay_dict[current_hadm_id]["sub_items"] =[]
            former_stay_id = current_stay_id
        else:
            stay_dict[current_hadm_id]["sub_items"].append(entry)

    for hadm_id in stay_dict.keys():
        result_data_list.append(stay_dict[hadm_id])
    
    return result_data_list

    
def sort_combined_data(combined_data_dict):
    result_data_list = combined_data_dict["hosp.jsonl"]
    
    try:
        note_items = []
        for _ in combined_data_dict["note.jsonl"]:
            note_items.append(_)
        result_data_list = result_data_list + note_items
    except:
        pass
    
    try:
        ed_data_list = combined_data_dict["ed.jsonl"]
        result_data_list = result_data_list + ed_data_list
        # result_data_list = my_merge(result_data_list,ed_data_list,file_name = "ed")
    except:
        pass
    
    try:
        icu_data_list = combined_data_dict["icu.jsonl"]
        result_data_list = result_data_list + icu_data_list
        # result_data_list = my_merge(result_data_list,icu_data_list,file_name = "icu")
    except:
        pass
    
    result_data_list = sort_hosp(result_data_list)
    
    return result_data_list

sort_functions = {
    "note.jsonl": sort_note,
    "ed.jsonl": sort_ed,
    "icu.jsonl": sort_icu,
    "hosp.jsonl": sort_hosp,
}

def read_jsonl(jsonl_dir):
    data_list = []
    with open(jsonl_dir, 'r') as file:
        for line in file:
            data_list.append(json.loads(line))
    return data_list

def save_jsonl(data_list, file_name):
    
    with jsonlines.open(file_name, mode='w') as writer:
        writer.write_all(data_list)
    
def rank_per_patient(subject_dir):
    combined_data_dict = {}
    jsonl_list = os.listdir(subject_dir)
    
    
    save_patient_dir = subject_dir.replace("patients","patients_sorted")
    if not os.path.exists(save_patient_dir):
        os.mkdir(save_patient_dir)
    
    for jsonl_name in jsonl_list:
        if jsonl_name == "combined.jsonl":
            continue

        # if jsonl_name != "ed.jsonl":
        #     continue

        jsonl_dir = subject_dir + '/' + jsonl_name
        
        data_list = read_jsonl(jsonl_dir)
        sort_function = sort_functions[jsonl_name]
        
        data_list =  sort_function(data_list)
        # save_jsonl(data_list, file_name = save_patient_dir + '/' + jsonl_name)
        
        combined_data_dict[jsonl_name] = data_list
    
    combined_data_list = sort_combined_data(combined_data_dict)
    save_jsonl(combined_data_list, file_name = save_patient_dir +'/'+"combined.jsonl")


def process_subject(subject):
    if not os.path.exists(ROOT_DIR + '/' + subject + '/combined.jsonl'):
        rank_per_patient(ROOT_DIR + '/' + subject)


if __name__ == "__main__":
    
    
    # 使用 parfor 的方式处理
    Parallel(n_jobs=-1)(delayed(process_subject)(subject) for subject in tqdm.tqdm(SUBJECTS))

    # # cnt = 0
    # for subject in tqdm.tqdm(SUBJECTS):
    #     rank_per_patient(ROOT_DIR + '/' + subject)
        # cnt = cnt + 1
        # if cnt >=10:
        #     break
    
    # rank_per_patient("/mnt3/longquan.lys/projects/EHRL/mimiciv/patient_test/patients/10000032")
    
    