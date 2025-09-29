import os
import json
import pandas as pd
import tqdm
from functools import partial

EVENT_NAME = {
    # admission
    "patients": "Patient Demographics",
    "omr": "Online Medical Record",
    "pharmacy": "Pharmacy",
    "poe": "Provider Order Entry",
    "procedures_icd": "Procedures on International Classification of Diseases",
    "prescriptions": "Prescriptions",
    "services": "Services",
    "labevents": "Labotary Test Events",
    "microbiologyevents": "Microbiology Test Events",
    "admissions": "Admissions",
    "transfers": "Transfers",
    "emar": "Electronic Medicine Administration Record",
    "hcpcsevents": "Hcpcs Events",
    "diagnoses_icd": "Diagnoses on International Classification of Diseases",
    "drgcodes": "Drgcodes",
    # note
    "radiology":  "Radiology Examinations",
    "discharge": "Discharge",
    # ed
    "edstays": "EDstays",
    "triage": "Triage",
    "medrecon": "Medrecon",
    "pyxis": "Pyxis",
    "vitalsign": "Vitalsign",
    "diagnosis": "ED Diagnoses on International Classification of Diseases",
    # icu
    "icustays": "ICUstays",
    "chartevents": "Chart Events",
    "ingredientevents": "Ingredient Events",
    "datetimeevents": "Datetime Events",
    "procedureevents": "Procedure Events",
    "inputevents": "Input Events",
    "outputevents": "Output Events",
}

def safe_read(json_element):
    if isinstance(json_element, float) or isinstance(json_element, int):
        json_element = str(json_element)
    
    if isinstance(json_element, list):
        return json_element
    
    if isinstance(json_element, str):
        if json_element == 'NaN' or json_element == "nan" or json_element == "None":
            return ""
    
    if pd.isna(json_element):
        json_element = ""
    
    if json_element is None:
        return ""
    
    return json_element

### load item index func
def process_icd(index_csv):
    inflect_index = {"9":{},"10":{}}
    with open(index_csv) as f:
        icd_index = pd.read_csv(f)
    for index in tqdm.tqdm(range(len(icd_index))):
        sample = icd_index.iloc[index]
        inflect_index[str(sample["icd_version"])][str(sample["icd_code"])] = sample["long_title"]
    return inflect_index

def process_prescriptions_atc(index_csv):
    df = pd.read_csv(index_csv)
    inflect_index = df.set_index('ndc_code')['atc_name'].to_dict()
    return inflect_index
    
def process_hcpcs_item(index_csv):
    inflect_index = {}
    prompt ="{short_description}"
    with open(index_csv) as f:
        index_pd = pd.read_csv(f)
    for index in tqdm.tqdm(range(len(index_pd))):
        sample = index_pd.iloc[index]
        inflect_index[str(sample["code"])] = prompt.format(short_description = sample["short_description"])
    return inflect_index
    
def process_lab_item(index_csv):
    inflect_index = {}
    #prompt ="Event_name: {label} \nFluid: {fluid} \nCategory: {category} \n"
    prompt = "{label}"
    with open(index_csv) as f:
        index_pd = pd.read_csv(f)
    for index in tqdm.tqdm(range(len(index_pd))):
        sample = index_pd.iloc[index]
        #inflect_index[str(sample["itemid"])] = prompt.format(label = sample["label"], fluid = sample["fluid"], category = sample["category"])
        inflect_index[str(sample["itemid"])] = prompt.format(label = sample["label"])
    return inflect_index
    
    
def process_icu_item(index_csv):
    """
    label,abbreviation,linksto,category,unitname,param_type,lownormalvalue,highnormalvalue
    """
    inflect_index = {}
        
    #prompt ="Event_name: {label} \nAbbreviation: {abbreviation} \nLinksto: {linksto} \nCategory: {category} \nUnitname: {unitname} \nParam_type: {param_type} \nLownormalvalue: {lownormalvalue} \nHighnormalvalue: {highnormalvalue} \n"
    prompt = "{label}"
    with open(index_csv) as f:
        index_pd = pd.read_csv(f)
    
    # index_pd = index_pd[index_pd["field_name"] == "exam_name" & index_pd["field_ordinal"] == "1"]

    for index in tqdm.tqdm(range(len(index_pd))):
        sample = index_pd.iloc[index]
        #inflect_index[str(sample["itemid"])] = prompt.format(label = sample["label"], abbreviation = sample["abbreviation"], linksto = sample["linksto"], category = sample["category"], unitname = sample["unitname"], param_type = sample["param_type"], lownormalvalue = sample["lownormalvalue"], highnormalvalue = sample["highnormalvalue"])
        inflect_index[str(sample["itemid"])] = prompt.format(label = sample["label"])
    return inflect_index

def process_radiology_item(index_csv):
    inflect_index = {}
    with open(index_csv) as f:
        index_pd = pd.read_csv(f)
    
    index_pd = index_pd[index_pd["field_name"] == "exam_name"]
    for index in tqdm.tqdm(range(len(index_pd))):
        sample = index_pd.iloc[index]
        note_id = sample["note_id"]
        
        if note_id not in inflect_index:
            inflect_index[note_id] = []
        inflect_index[note_id].append(sample["field_value"])
    
    for note_id in inflect_index:
        inflect_index[note_id] = list(set(inflect_index[note_id]))
    
    return inflect_index


### free text transfer func
def patients_item_to_free_text(item):
    item_list = item["items"]
    assert len(item_list)==1, print(item_list)
    gender = "Male" if item_list[0]["gender"] == "M" else "Female"
    age = item_list[0]["anchor_age"]
    
    prompt = """## Patient Demographics\n- Age: {age}\n- Gender: {gender}"""
    return prompt.strip().format(age=age, gender=gender)
    
def labevents_item_to_free_text(item, item_indexing):
    """
    {"labevent_id": 318, "subject_id": 10000032, "hadm_id": 22841357.0, "specimen_id": 1953111, "itemid": 51237, "order_provider_id": NaN, "charttime": "2180-06-27 05:10:00", "storetime": "2180-06-27 06:59:00", "value": "1.5", "valuenum": 1.5, "valueuom": NaN, "ref_range_lower": 0.9, "ref_range_upper": 1.1, "flag": "abnormal", "priority": "ROUTINE", "comments": NaN, "file_name": "labevents"}
    """
    item_list = item["items"]
    file_name = item["file_name"]
    start_time = item["starttime"]

    prompt = "## {event_name} [{start_time}]\n".format(event_name=EVENT_NAME[file_name], start_time=start_time)
    
    chart_item = []
    title_prompt = "| Item Name | Value | Unit | Range | Flag | Comments |\n"
    title_prompt += "| ------ | ------ | ------ | ------ | ------ | ------ |\n"
    chart_prompt = "| {item_name} | {valuenum} | {valueuom} | {ref_range} | {flag} | {comments} |"
    for item in item_list:
        item_name = item_indexing[safe_read(item["itemid"])]
        valuenum = safe_read(item["valuenum"])
        valueuom = safe_read(item["valueuom"]) 
        ref_range_lower = safe_read(item["ref_range_lower"]) 
        ref_range_upper = safe_read(item["ref_range_upper"]) 
        ref_range = f"{ref_range_lower}-{ref_range_upper}"
        flag = safe_read(item["flag"]) 
        comments = safe_read(item["comments"]) 
        flag = "normal" if flag =="" else flag
        if valuenum != "":
            chart_item.append(chart_prompt.format(item_name=item_name, valuenum=valuenum, valueuom=valueuom, ref_range=ref_range, flag=flag, comments=comments))
    
    chart_item_str = "\n".join(chart_item)
    
    prompt = prompt + title_prompt + chart_item_str
    return prompt.strip()


def omr_item_to_free_text(item):
    """
    {"subject_id": 10000032, "chartdate": "2180-06-27", "seq_num": 1, "result_name": "BMI (kg/m2)", "result_value": "19.2", "file_name": "omr"}
    """
    item_list = item["items"]
    file_name = item["file_name"]
    start_time = item["starttime"]

    prompt = "## {event_name} [{start_time}]\n".format(event_name=EVENT_NAME[file_name], start_time=start_time)
    
    chart_item = []
    chart_prompt = "- {result_name}: {result_value}"
    for item in item_list:
        result_name = safe_read(item["result_name"])
        result_value = safe_read(item["result_value"])
        chart_item.append(chart_prompt.format(result_name=result_name,result_value=result_value))
    chart_item_str = "\n".join(chart_item)
    prompt = prompt + chart_item_str
    
    return prompt.strip()

def transfers_item_to_free_text(item):
    """
    {"subject_id": 10000032, "hadm_id": 22841357.0, "transfer_id": 34703856, "eventtype": "admit", "careunit": "Transplant", "intime": "2180-06-26 21:31:00", "outtime": "2180-06-27 18:49:12", "file_name": "transfers"}
    """
    # assert len( item["items"])==1, print( item["items"])
    item_list = item["items"]
    file_name = item["file_name"]
    start_time = item["starttime"]
    prompt = "## {event_name} [{start_time}]\n".format(event_name=EVENT_NAME[file_name], start_time=start_time)

    chart_item = []
    for item in item_list:
        eventtype = safe_read(item["eventtype"])
        careunit = safe_read(item["careunit"])
        chart_item.append("- {eventtype}\n".format(eventtype=careunit) if careunit else "- {eventtype}".format(eventtype=eventtype))
    
    chart_item_str = "\n".join(chart_item)
    prompt = prompt + chart_item_str
    return prompt.strip()

def poe_item_to_free_text(item):
    """
    {"poe_id": "10000032-69", "poe_seq": 69, "subject_id": 10000032, "hadm_id": 22841357, "ordertime": "2180-06-26 19:10:32", "order_type": "Lab", "order_subtype": NaN, "transaction_type": "New", "discontinue_of_poe_id": NaN, "discontinued_by_poe_id": NaN, "order_provider_id": "P699GL", "order_status": "Inactive", "file_name": "poe"}
    """
    item_list = item["items"]
    file_name = item["file_name"]
    start_time = item["starttime"]
    prompt = "## {event_name} [{start_time}]\n".format(event_name=EVENT_NAME[file_name], start_time=start_time)
    
    chart_item = []
    chart_prompt = "- {order_type}: {order_subtype}"
    
    for item in item_list:
        order_type = safe_read(item["order_type"])
        order_subtype = safe_read(item["order_subtype"])
        chart_item.append(chart_prompt.format(order_type=order_type,order_subtype=order_subtype))
    chart_item_str = "\n".join(chart_item)
    prompt = prompt + chart_item_str
    
    return prompt.strip()

def services_item_to_free_text(item):
    """
    {"subject_id": 10000032, "hadm_id": 22841357, "transfertime": "2180-06-26 18:28:08", "prev_service": NaN, "curr_service": "MED", "file_name": "services"}
    """
    item_list = item["items"]
    file_name = item["file_name"]
    start_time = item["starttime"]
    prompt = "## {event_name} [{start_time}]\n".format(event_name=EVENT_NAME[file_name], start_time=start_time)
    
    chart_item = []
    chart_prompt = "- {curr_service}"
    
    for item in item_list:
        curr_service = item["curr_service"]
        chart_item.append(chart_prompt.format(curr_service=curr_service))
    chart_item_str = "\n".join(chart_item)
    prompt = prompt + chart_item_str
    
    return prompt.strip()

def pharmacy_item_to_free_text(item):
    """
    {"subject_id": 10000032, "hadm_id": 22595853, "pharmacy_id": 14779570, "poe_id": "10000032-22", "starttime": "2180-05-07 00:00:00", "stoptime": "2180-05-07 22:00:00", "medication": "Sodium Chloride 0.9%  Flush", "proc_type": "Unit Dose", "status": "Discontinued via patient discharge", "entertime": "2180-05-07 00:00:54", "verifiedtime": "2180-05-07 00:00:54", "route": "IV", "frequency": "Q8H", "disp_sched": "0, 8, 16", "infusion_type": NaN, "sliding_scale": NaN, "lockout_interval": NaN, "basal_rate": NaN, "one_hr_max": NaN, "doses_per_24_hrs": 3.0, "duration": NaN, "duration_interval": "Ongoing", "expiration_value": 36.0, "expiration_unit": "Hours", "expirationdate": NaN, "dispensation": "Floor Stock Item", "fill_quantity": NaN, "file_name": "pharmacy"}
    """
    item_list = item["items"]
    file_name = item["file_name"]
    start_time = item["starttime"]
    prompt = "## {event_name} [{start_time}]\n".format(event_name=EVENT_NAME[file_name], start_time=start_time)

    title_prompt = "| Medication | Proc_type |\n"
    title_prompt += "| ------ | ------ |\n"

    chart_item = []
    chart_prompt = "| {medication} | {proc_type} |"
    
    for item in item_list:
        medication = safe_read(item["medication"])
        proc_type = safe_read(item["proc_type"])
        chart_item.append(chart_prompt.format(medication=medication, proc_type=proc_type))
    chart_item_str = "\n".join(chart_item)
    prompt = prompt + title_prompt + chart_item_str

    return prompt.strip()


def prescriptions_item_to_free_text(item):
    """
    {"subject_id": 10000032, "hadm_id": 22595853, "pharmacy_id": 14779570, "poe_id": "10000032-22", "poe_seq": 22.0, "order_provider_id": "P76JEQ", "starttime": "2180-05-07 00:00:00", "stoptime": "2180-05-07 22:00:00", "drug_type": "MAIN", "drug": "Sodium Chloride 0.9%  Flush", "formulary_drug_cd": "NACLFLUSH", "gsn": NaN, "ndc": 0.0, "prod_strength": "10 mL Syringe", "form_rx": NaN, "dose_val_rx": "3", "dose_unit_rx": "mL", "form_val_disp": "0.3", "form_unit_disp": "SYR", "doses_per_24_hrs": 3.0, "route": "IV", "file_name": "prescriptions"}
    """
    item_list = item["items"]
    file_name = item["file_name"]
    start_time = item["starttime"]
    prompt = "## {event_name} [{start_time}]\n".format(event_name=EVENT_NAME[file_name], start_time=start_time)

    input_key = ["drug", "prod_strength", "dose_val_rx", "dose_unit_rx"]

    title_prompt = f"""| {" | ".join([key.title() for key in input_key])} |\n"""
    title_prompt += f"""| {" | ".join(["------"] * len(input_key))} |\n"""

    chart_item = []
    # chart_prompt = "| {drug} | {prod_strength} |"
    
    for item in item_list:
        # drug = safe_read(item["drug"])
        # proc_type = safe_read(item["proc_type"])
        chart_item.append(f"""| {" | ".join([safe_read(item[key]) for key in input_key])} |""")
        # chart_item.append(chart_prompt.format(drug=drug, proc_type=proc_type))
    chart_item_str = "\n".join(chart_item)
    prompt = prompt + title_prompt + chart_item_str

    return prompt.strip() 


def note_item_to_free_text(item, item_indexing=None):
    """
    "note_id": "10000032-RR-16", "subject_id": "10000032", "hadm_id": 22595853.0, "note_type": "RR", "note_seq": "16", "charttime": "2180-05-07 09:55:00", "storetime": "2180-05-07 11:15:00", "text": "INDICATION:  ___ HCV cirrhosis c/b ascites, hiv on ART, h/o IVDU, COPD,\nbioplar, PTSD, presented from OSH ED with worsening abd distension over past\nweek.  // SBP\n\nTECHNIQUE:  Ultrasound guided diagnostic and therapeutic paracentesis\n\nCOMPARISON:  Abdominal ultrasound ___\n\nFINDINGS: \n\nLimited grayscale ultrasound imaging of the abdomen demonstrated\nmoderateascites. A suitable target in the deepest pocket in the right lower\nquadrant was selected for paracentesis.\n\nPROCEDURE:  The procedure, risks, benefits and alternatives were discussed\nwith the patient and written informed consent was obtained.\n\nA preprocedure time-out was performed discussing the planned procedure,\nconfirming the patient's identity with 3 identifiers, and reviewing a\nchecklist per ___ protocol.\n\nUnder ultrasound guidance, an entrance site was selected and the skin was\nprepped and draped in the usual sterile fashion. 1% lidocaine was instilled\nfor local anesthesia.\n\nA 5 ___ catheter was advanced into the largest fluid pocket in the right\nlower quadrant and 1.5 L of serosanguinous fluid was removed.\n\nThe patient tolerated the procedure well without immediate complication.\nEstimated blood loss was minimal.  A sample of the fluid was sent to the lab\nas requested.\n\nDr. ___ attending radiologist, was present throughout the critical\nportions of the procedure.\n\nIMPRESSION: \n\nSuccessful uncomplicated ultrasound guided diagnostic and therapeutic\nparacentesis yielding 1.5 L of serosanguineous fluid from the right lower\nquadrant.  Sample was sent to the lab as requested.\n", "file_name": "note"}
    """
    item_list = item["items"]
    file_name = item["file_name"]
    start_time = item["starttime"]
    prompt = "## {event_name} [{start_time}]\n".format(event_name=EVENT_NAME[file_name], start_time=start_time)
    

    if file_name == "discharge":
        chart_item = []
        chart_prompt = "- Note Type: {note_type}\n- Report:\n{text}"
        
        for item in item_list:
            note_type = safe_read(item["note_type"])
            text = safe_read(item["text"])
            chart_item.append(chart_prompt.format(note_type=note_type, text= text))
        chart_item_str = "\n".join(chart_item)
        prompt = prompt + chart_item_str
    
    elif file_name == "radiology":
        chart_item = []
        chart_prompt = "- Note Type: {note_type}\n- Exam Name: {exam_name}\n- Report:\n{text}"

        for item in item_list:
            note_type = safe_read(item["note_type"])
            try:
                exam_name_set = item_indexing[item["note_id"]]
            except:
                exam_name_set = []
            exam_name = ", ".join(exam_name_set)
            text = safe_read(item["text"])
            chart_item.append(chart_prompt.format(note_type=note_type, exam_name=exam_name, text= text))
        
        chart_item_str = "\n".join(chart_item)
        prompt = prompt + chart_item_str
    
    return prompt.strip()
    
def admissions_item_to_free_text(item):
    item_list = item["items"]
    file_name = item["file_name"]
    start_time = item["starttime"]
    prompt = "## {event_name} [{start_time}]\n".format(event_name=EVENT_NAME[file_name], start_time=start_time)

    admission_location = safe_read(item_list[0]["admission_location"])
    admission_type = safe_read(item_list[0]["admission_type"])
    prompt += "- Admission Type: {admission_type}\n- Admission Location: {admission_location}".format(admission_type=admission_type, admission_location=admission_location)
    
    if "admission_info" in item_list[0]:
        for info_key, info in item_list[0]["admission_info"].items():
            prompt += f"- {info_key}: {info}\n"

    return prompt.strip()

def emar_item_to_free_text(item):
    """
    {"subject_id": 10000032, "hadm_id": 22595853.0, "emar_id": "10000032-10", "emar_seq": 10, "poe_id": "10000032-36", "pharmacy_id": 48770010.0, "enter_provider_id": NaN, "charttime": "2180-05-07 00:44:00", "medication": "Potassium Chloride", "event_txt": "Administered", "scheduletime": "2180-05-07 00:44:00", "storetime": "2180-05-07 00:44:00", "file_name": "emar"}
    """
    item_list = item["items"]
    file_name = item["file_name"]
    start_time = item["starttime"]
    prompt = "## {event_name} [{start_time}]\n".format(event_name=EVENT_NAME[file_name], start_time=start_time)
    
    chart_item = []
    chart_prompt = "- {medication}"
    
    for item in item_list:
        medication = safe_read(item["medication"])
        chart_item.append(chart_prompt.format(medication=medication))
    chart_item_str = "\n".join(chart_item)
    prompt = prompt + chart_item_str
    
    return prompt.strip()

def microbiologyevents_item_to_free_text(item):
    """
    {"microevent_id": 20, "subject_id": 10000032, "hadm_id": NaN, "micro_specimen_id": 5842819, "order_provider_id": NaN, "chartdate": "2180-06-26 00:00:00", 
    "charttime": "2180-06-26 18:30:00", "spec_itemid": 70079, "spec_type_desc": "URINE", "test_seq": 1, "storedate": "2180-06-29 00:00:00", "storetime": "2180-06-29 14:32:00", 
    "test_itemid": 90039, "test_name": "URINE CULTURE", "org_itemid": 80053.0, "org_name": "ENTEROCOCCUS SP.", "isolate_num": 1.0, "quantity": NaN, "ab_itemid": 90004.0, 
    "ab_name": "AMPICILLIN", "dilution_text": "<=2", "dilution_comparison": "<=        ", "dilution_value": 2.0, 
    "interpretation": "S", "comments": "MIXED BACTERIAL FLORA ( >= 3 COLONY TYPES), CONSISTENT WITH SKIN AND/OR GENITAL CONTAMINATION.  ", "file_name": "microbiologyevents"}
    """
    item_list = item["items"]
    file_name = item["file_name"]
    start_time = item["starttime"]
    prompt = "## {event_name} [{start_time}]\n".format(event_name=EVENT_NAME[file_name], start_time=start_time)
    
    
    chart_item = []
    title_prompt = "| Item Name | AB Name | Dilution Text | Interpretation | Comments |\n"
    title_prompt += "| ------ | ------ | ------ | ------ | ------ | ------ |\n"
    # chart_prompt = "| {item_name} | {valuenum} | {valueuom} | {ref_range} | {flag} | {comments} |"
    chart_prompt = "| {test_name} | {ab_name} | {dilution_text} | {interpretation} | {comments} |"
    for item in item_list:
        test_name = safe_read(item["test_name"])
        dilution_text = safe_read(item["dilution_text"])
        interpretation = safe_read(item["interpretation"]) 
        comments = safe_read(item["comments"]) 
        ab_name = safe_read(item["ab_name"])
        
        chart_item.append(chart_prompt.format(test_name=test_name, ab_name=ab_name, dilution_text=dilution_text, interpretation=interpretation, comments=comments, ))
    
    chart_item_str = "\n".join(chart_item)
    
    prompt = prompt + title_prompt + chart_item_str
    
    return prompt.strip()


def diagnoses_icd_item_to_free_text(item, item_indexing):
    item_list = item["items"]
    file_name = item["file_name"]
    start_time = item["starttime"]
    prompt = "## {event_name} [{start_time}]\n".format(event_name=EVENT_NAME[file_name], start_time=start_time)

    chart_item = []
    chart_prompt = "- {disease_name}"
    for item in item_list:
        icd_code = safe_read(item["icd_code"])
        icd_version = safe_read(item["icd_version"])
        disease_name = item_indexing[icd_version][icd_code]
        
        chart_item.append(chart_prompt.format(disease_name=disease_name))
    
    chart_item_str = "\n".join(chart_item)
    
    prompt = prompt + chart_item_str
    
    return prompt.strip()
    
def procedures_icd_item_to_free_text(item, item_indexing):

    item_list = item["items"]
    file_name = item["file_name"]
    start_time = item["starttime"]
    prompt = "## {event_name} [{start_time}]\n".format(event_name=EVENT_NAME[file_name], start_time=start_time)
    
    
    chart_item = []
    chart_prompt = "- {procedure_name}"
    for item in item_list:
        icd_code = safe_read(item["icd_code"])
        icd_version = safe_read(item["icd_version"])
        procedure_name = item_indexing[icd_version][icd_code]
        
        chart_item.append(chart_prompt.format(procedure_name=procedure_name))
    
    chart_item_str = "\n".join(chart_item)
    
    prompt = prompt + chart_item_str
    
    return prompt.strip()

def ed_item_to_free_text(item):
    item_lists = item["items"]
    file_name = item["file_name"]
    start_time = item["starttime"]
    prompt = "## {event_name} [{start_time}]\n".format(event_name=EVENT_NAME[file_name], start_time=start_time)

    chart_item = []
    for item in item_lists:
        file_name = item["file_name"]
        if file_name == "edstay":
            gender = "Male" if safe_read(item["gender"]) == "M" else "Female"
            race = safe_read(item["race"])
            chart_prompt = "- Gender: {gender}\n- Race: {race}".format(gender=gender, race=race)
            chart_item.append(chart_prompt)

        if file_name == "triage":
            temperature = safe_read(item["temperature"])
            heartrate = safe_read(item["heartrate"])
            resprate = safe_read(item["resprate"])
            o2sat = safe_read(item["o2sat"])
            sbp = safe_read(item["sbp"])
            dbp = safe_read(item["dbp"])
            pain = safe_read(item["pain"])
            acuity = safe_read(item["acuity"])
            chiefcomplaint = safe_read(item["chiefcomplaint"])
            
            chart_prompt = "- Temperature: {temperature}\n- Heartrate: {heartrate}\n- Resprate: {resprate}\n-O2sat: {o2sat}\n-Sbp: {sbp}\n- Dbp: {dbp}\n- Pain: {pain}\n- Acuity: {acuity}\n- Chiefcomplaint: {chiefcomplaint}".format(temperature=temperature, heartrate=heartrate, resprate=resprate, o2sat=o2sat, sbp=sbp, dbp=dbp, pain=pain, acuity=acuity, chiefcomplaint=chiefcomplaint)
            chart_item.append(chart_prompt)
        
        if file_name == "diagnosis":
            disease_name = safe_read(item["icd_title"]).lower().capitalize()
            chart_prompt = "- {disease_name}".format(disease_name = disease_name)
            chart_item.append(chart_prompt)
        
        if file_name == "medrecon":
            name = safe_read(item["name"])
            
            chart_prompt = "- {name}".format(name = name)
            chart_item.append(chart_prompt)
        
        if file_name == "pyxis":
            name = safe_read(item["name"])
            chart_prompt = "- {name}".format(name = name)
            chart_item.append(chart_prompt)
        
        if file_name == "vitalsign":
            temperature = safe_read(item["temperature"])
            heartrate = safe_read(item["heartrate"])
            resprate = safe_read(item["resprate"])
            o2sat = safe_read(item["o2sat"])
            sbp = safe_read(item["sbp"])
            dbp = safe_read(item["dbp"])
            pain = safe_read(item["pain"])
            rhythm = safe_read(item["rhythm"])
            
            chart_prompt = "- Temperature: {temperature}\n- Heartrate: {heartrate}\n- Resprate: {resprate}\n- O2sat: {o2sat}\n- Sbp: {sbp}\n- Dbp: {dbp}\n- Pain: {pain}\n- Rhythm: {rhythm}".format(temperature=temperature, heartrate=heartrate, resprate=resprate, o2sat=o2sat, sbp=sbp, dbp=dbp, pain=pain, rhythm=rhythm)
            chart_item.append(chart_prompt)

    chart_item_str = " \n".join(chart_item)
    prompt = prompt + chart_item_str
    return prompt.strip()

def chartevents_item_to_free_text(item, item_indexing):
    item_lists = item["items"]
    file_name = item["file_name"]
    start_time = item["starttime"]
    prompt = "## {event_name} [{start_time}]\n".format(event_name=EVENT_NAME[file_name], start_time=start_time)

    title_prompt = "| Item Name | Value | Unit | Warning |\n"
    title_prompt += "| ------ | ------ | ------ | ------ |\n"

    chart_item = []
    for item in item_lists:
        item_name = item_indexing[safe_read(item["itemid"])]
        valuenum = safe_read(item["valuenum"])
        valueuom = safe_read(item["valueuom"])
        warning = safe_read(item["warning"])
        chart_prompt = "| {item_name} | {valuenum} | {valueuom} | {warning}".format(item_name=item_name, valuenum=valuenum, valueuom=valueuom, warning=warning)
        chart_item.append(chart_prompt)
    
    chart_item_str = prompt + title_prompt + "\n".join(chart_item)
    return chart_item_str

def icustay_item_to_free_text(item):
    file_name = item["file_name"]
    start_time = item["starttime"]
    prompt = "## {event_name} [{start_time}]\n".format(event_name=EVENT_NAME[file_name], start_time=start_time)

    return prompt.strip()

def inputevents_item_to_free_text(item, item_indexing):
    item_lists = item["items"]
    file_name = item["file_name"]
    start_time = item["starttime"]
    prompt = "## {event_name} [{start_time}]\n".format(event_name=EVENT_NAME[file_name], start_time=start_time)

    title_prompt = "| Item Name | Amount | Unit | Rate | Unit |\n"
    title_prompt += "| ------ | ------ | ------ | ------ | ------ |\n"

    chart_item = []
    for item in item_lists:
        item_name = item_indexing[safe_read(item["itemid"])]
        amount = safe_read(item["amount"])
        amountuom = safe_read(item["amountuom"])
        rate = safe_read(item["rate"])
        rateuom = safe_read(item["rateuom"])
        
        chart_prompt = "| {item_name} | {amount} | {amountuom} | {rate} | {rateuom} |".format(item_name=item_name, amount=amount, amountuom=amountuom, rate=rate, rateuom=rateuom)
        chart_item.append(chart_prompt)
    
    chart_item_str = prompt + title_prompt + "\n".join(chart_item)
    return chart_item_str
  
def outputevents_item_to_free_text(item, item_indexing):
    item_lists = item["items"]
    file_name = item["file_name"]
    start_time = item["starttime"]
    prompt = "## {event_name} [{start_time}]\n".format(event_name=EVENT_NAME[file_name], start_time=start_time)

    title_prompt = "| Item Name | Value | Unit |\n"
    title_prompt += "| ------ | ------ | ------ |\n"

    chart_item = []
    for item in item_lists:
        item_name = item_indexing[safe_read(item["itemid"])]
        value = safe_read(item["value"])
        valueuom = safe_read(item["valueuom"])
        
        chart_prompt = "| {item_name} | {value} | {valueuom} |".format(item_name=item_name, value=value, valueuom=valueuom)
        chart_item.append(chart_prompt)
    
    chart_item_str = prompt + title_prompt + "\n".join(chart_item)
    return chart_item_str

def procedureevents_item_to_free_text(item, item_indexing):
    item_lists = item["items"]
    file_name = item["file_name"]
    start_time = item["starttime"]
    prompt = "## {event_name} [{start_time}]\n".format(event_name=EVENT_NAME[file_name], start_time=start_time)

    title_prompt = "| Item Name | Value | Unit | Location | Location Category |\n"
    title_prompt += "| ------ | ------ | ------ | ------ | ------ |\n"

    chart_item = []
    for item in item_lists:
        item_name = item_indexing[safe_read(item["itemid"])]
        value = safe_read(item["value"])
        valueuom = safe_read(item["valueuom"])
        location = safe_read(item["location"])
        locationcategory = safe_read(item["locationcategory"])
        
        chart_prompt = "| {item_name} | {value} | {valueuom} | {location} | {locationcategory} |".format(item_name=item_name, value=value, valueuom=valueuom, location=location, locationcategory=locationcategory)
        chart_item.append(chart_prompt)
    
    chart_item_str = prompt + title_prompt + "\n".join(chart_item)
    return chart_item_str


def datetimeevents_item_to_free_text(item, item_indexing):
    item_lists = item["items"]
    file_name = item["file_name"]
    start_time = item["starttime"]
    prompt = "## {event_name} [{start_time}]\n".format(event_name=EVENT_NAME[file_name], start_time=start_time)

    chart_item = []
    for item in item_lists:
        item_name = item_indexing[safe_read(item["itemid"])]
        chart_prompt = "- {item_name}".format(item_name=item_name)
        chart_item.append(chart_prompt)
    
    chart_item_str = prompt + "\n".join(chart_item)
    return chart_item_str 

class MIMICIVStringConvertor:
    def __init__(self, origin_data_dir, cache_dir):
        # basic args
        self.origin_data_dir = origin_data_dir
        self.cache_dir = cache_dir
        
        ### event_info
        ## event_type: event name
        # item_mapping: mapping function from input_name to output_name
        # info_keys: info keys show in the history
        # string_convert_func: transfer function
        self.event_info = {
            # admission
            "patients": {
                "event_type": "hosp",
                "item_mapping": None,
                "info_keys": ["anchor_age", "gender"],
                "string_convert_func": patients_item_to_free_text,
            },
            "omr": {
                "event_type": "hosp",
                "item_mapping": None,
                "info_keys": ["result_name", "result_value"],
                "string_convert_func": omr_item_to_free_text,
            },
            "pharmacy": {
                "event_type": "hosp",
                "item_mapping": None,
                "info_keys": ["medication", "proc_type", "status"],
                "string_convert_func": pharmacy_item_to_free_text,
            },
            "prescriptions": {
                "event_type": "hosp",
                "item_mapping": [
                    {
                        "input_name": "ndc",
                        "item_index_file": "d_atc_prescriptions",
                        "item_index_preprocess_func": process_prescriptions_atc,
                        "output_name": "ATC Type",
                    }
                ],
                "info_keys": ["drug", "ATC Type", "prod_strength", "dose_val_rx", "dose_unit_rx"],
                "string_convert_func": prescriptions_item_to_free_text,
            },
            "poe": {
                "event_type": "hosp",
                "item_mapping": None,
                "info_keys": ["order_type", "order_subtype"],
                "string_convert_func": poe_item_to_free_text,
            },
            "procedures_icd": {
                "event_type": "hosp",
                "item_mapping": [
                    {
                        "input_name": "icd_code",
                        "item_index_preprocess_func": process_icd,
                        "item_index_file": "d_icd_procedures",
                        "output_name": "procedures"
                    },
                    {
                        "input_name": "icd_code",
                        "item_index_preprocess_func": process_icd,
                        "item_index_file": "d_ccs_procedures",
                        "output_name": "CCS Type"
                    },
                ],
                "info_keys": ["procedures", "CCS Type"],
                "string_convert_func": procedures_icd_item_to_free_text,
            },
            "services": {
                "event_type": "hosp",
                "item_mapping": None,
                "info_keys": ["curr_service"],
                "string_convert_func": services_item_to_free_text,
            },
            "labevents": {
                "event_type": "hosp",
                "item_mapping": [
                    {
                        "input_name": "itemid",
                        "item_index_file": "d_labitems",
                        "item_index_preprocess_func": process_lab_item,
                        "output_name": "item_name",
                    }
                ],
                "info_keys": ["item_name", "valuenum", "valueuom", "ref_range_lower", "ref_range_upper", "flag", "comments"],
                "string_convert_func": labevents_item_to_free_text,
            },
            "microbiologyevents": {
                "event_type": "hosp",
                "item_mapping": None,
                "info_keys": ["test_name", "dilution_text", "interpretation", "comments", "ab_name"],
                "string_convert_func": microbiologyevents_item_to_free_text,
            },
            "admissions": {
                "event_type": "hosp",
                "item_mapping": None,
                "info_keys": ["admission_type", "admission_location", "admission_info"],
                "string_convert_func": admissions_item_to_free_text,
            },
            "transfers": {
                "event_type": "hosp",
                "item_mapping": None,
                "info_keys": ["eventtype", "careunit"],
                "string_convert_func": transfers_item_to_free_text,
            },
            "emar": {
                "event_type": "hosp",
                "item_mapping": None,
                "info_keys": ["medication", "event_txt"],
                "string_convert_func": emar_item_to_free_text,
            },
            # "hcpcsevents": {
            #     "event_type": "hosp",
            #     "item_index_preprocess_func": process_hcpcs_item,
            #     "item_index_file": "d_hcpcs",
            # "info_keys": [],    
            # "string_convert_func": None,
            # },
            "diagnoses_icd": {
                "event_type": "hosp",
                "item_mapping": [
                    {
                        "input_name": "icd_code",
                        "item_index_preprocess_func": process_icd,
                        "item_index_file": "d_icd_diagnoses",
                        "output_name": "diagnoses"
                    },
                    {
                        "input_name": "icd_code",
                        "item_index_preprocess_func": process_icd,
                        "item_index_file": "d_ccs_diagnoses",
                        "output_name": "CCS Type"
                    },
                ],
                "info_keys": ["diagnoses", "CCS Type"],
                "string_convert_func": diagnoses_icd_item_to_free_text,
            },
            # "drgcodes": {
            #     "event_type": "hosp",
            #
            #     "item_mapping": None,
            # "info_keys": [],    
            # "string_convert_func": None,
            # },
            # note
            "radiology": {
                "event_type": "note",
                "item_mapping": [
                    {
                        "input_name": "note_id",
                        "item_index_preprocess_func": process_radiology_item,
                        "item_index_file": "radiology_detail",
                        "output_name": "exam_name",
                    }
                ],
                "info_keys": ["note_type", "exam_name", "text"],
                "string_convert_func": note_item_to_free_text,
            },
            "discharge": {
                "event_type": "note",
                "item_mapping": None,
                "info_keys": ["note_type", "text"],
                "string_convert_func": note_item_to_free_text,
            },
            # ed
            "edstays": {
                "event_type": "ed",
                "item_mapping": None,
                "info_keys": ["gender", "race"],
                "string_convert_func": ed_item_to_free_text,
            },
            "triage": {
                "event_type": "ed",
                "item_mapping": None,
                "info_keys": ["temperature", "heartrate", "resprate", "o2sat", "sbp", "dbp", "pain", "acuity", "chiefcomplaint"],
                "string_convert_func": ed_item_to_free_text,
            },
            "medrecon": {
                "event_type": "ed",
                "item_mapping": [
                    {
                        "input_name": "ndc",
                        "item_index_file": "d_atc_prescriptions",
                        "item_index_preprocess_func": process_prescriptions_atc,
                        "output_name": "ATC Type",
                    }
                ],
                "info_keys": ["name", "ATC Type"],
                "string_convert_func": ed_item_to_free_text,
            },
            "pyxis": {
                "event_type": "ed",
                "item_mapping": None,
                "info_keys": ["name"],
                "string_convert_func": ed_item_to_free_text,
            },
            "vitalsign": {
                "event_type": "ed",
                "item_mapping": None,
                "info_keys": ["temperature", "heartrate", "resprate", "o2sat", "sbp", "dbp", "pain", "rhythm"],
                "string_convert_func": ed_item_to_free_text,
            },
            "diagnosis": {
                "event_type": "ed",
                "item_mapping": [
                    {
                        "input_name": "icd_code",
                        "item_index_preprocess_func": process_icd,
                        "item_index_file": "d_ccs_diagnoses",
                        "output_name": "CCS Type"
                    },
                ],
                "info_keys": ["icd_title", "CCS Type"],
                "string_convert_func": ed_item_to_free_text,
            },
            # icu
            "icustays": {
                "event_type": "icu",
                "item_mapping": None,
                "info_keys": [],
                "string_convert_func": icustay_item_to_free_text,
            },
            "chartevents": {
                "event_type": "icu",
                "item_mapping": [
                    {
                        "input_name": "itemid",
                        "item_index_preprocess_func": process_icu_item,
                        "item_index_file": "d_items",
                        "output_name": "item_name",
                    }
                ],
                "info_keys": ["item_name", "valuenum", "valueuom", "warning"],
                "string_convert_func": chartevents_item_to_free_text,
            },
            "ingredientevents": {
                "event_type": "icu",
                "item_mapping": [
                    {
                        "input_name": "itemid",
                        "item_index_preprocess_func": process_icu_item,
                        "item_index_file": "d_items",
                        "output_name": "item_name",
                    }
                ],
                "info_keys": ["item_name", "amount", "amountuom", "rate", "rateuom"],
                "string_convert_func": inputevents_item_to_free_text,
            },
            "datetimeevents": {
                "event_type": "icu",
                "item_mapping": [
                    {
                        "input_name": "itemid",
                        "item_index_preprocess_func": process_icu_item,
                        "item_index_file": "d_items",
                        "output_name": "item_name",
                    }
                ],
                "info_keys": ["item_name"],
                "string_convert_func": datetimeevents_item_to_free_text,
            },
            "procedureevents": {
                "event_type": "icu",
                "item_mapping": [
                    {
                        "input_name": "itemid",
                        "item_index_preprocess_func": process_icu_item,
                        "item_index_file": "d_items",
                        "output_name": "item_name",
                    }
                ],
                "info_keys": ["item_name", "value", "valueuom", "location", "locationcategory"],
                "string_convert_func": procedureevents_item_to_free_text,
            },
            "inputevents": {
                "event_type": "icu",
                "item_mapping": [
                    {
                        "input_name": "itemid",
                        "item_index_preprocess_func": process_icu_item,
                        "item_index_file": "d_items",
                        "output_name": "item_name",
                    }
                ],
                "info_keys": ["item_name", "amount", "amountuom", "rate", "rateuom"],
                "string_convert_func": inputevents_item_to_free_text,
            },
            "outputevents": {
                "event_type": "icu",
                "item_mapping": [
                    {
                        "input_name": "itemid",
                        "item_index_preprocess_func": process_icu_item,
                        "item_index_file": "d_items",
                        "output_name": "item_name",
                    }
                ],
                "info_keys": ["item_name", "value", "valueuom"],
                "string_convert_func": outputevents_item_to_free_text,
            },
        }

        self.get_item_index_for_event()
    
    def get_item_index_for_event_old(self):
        index_cache_dir = os.path.join(self.cache_dir, "index")
        os.makedirs(index_cache_dir, exist_ok=True)

        for event_name in self.event_info:
            if self.event_info[event_name]["item_index_file"] and self.event_info[event_name]["item_index_preprocess_func"]:
                # print(f"Preprocessing item indexing for {event_name}...")
                event_type = self.event_info[event_name]["event_type"]
                event_mapping_file = self.event_info[event_name]["item_index_file"]
                item_index_cache_file = os.path.join(index_cache_dir, event_type, f"{event_mapping_file}.json")
            
                if os.path.exists(item_index_cache_file):
                    # print(f"""Found cache file {item_index_cache_file} exist, direclt loading from cache file...""")
                    with open(item_index_cache_file, "r") as f:
                        index_datas = json.load(f)

                else:
                    index_datas = self.event_info[event_name]["item_index_preprocess_func"](os.path.join(self.origin_data_dir, event_type, f"{event_mapping_file}.csv"))
                    os.makedirs(item_index_cache_file.rsplit("/", 1)[0], exist_ok=True)
                    with open(item_index_cache_file, "w") as f:
                        json.dump(index_datas, f, indent=4, ensure_ascii=False)
                
                self.event_info[event_name]["item_index"] = index_datas
                self.event_info[event_name]["string_convert_func"] = partial(self.event_info[event_name]["string_convert_func"], item_indexing=index_datas)
            else:
                self.event_info[event_name]["item_index"] = None
    
    def get_item_index_for_event(self):
        index_cache_dir = os.path.join(self.cache_dir, "index")
        os.makedirs(index_cache_dir, exist_ok=True)

        for event_name in self.event_info:
            # print(f"Preprocessing item indexing for {event_name}...")
            if self.event_info[event_name]["item_mapping"]:
                event_type = self.event_info[event_name]["event_type"]

                for mapping in self.event_info[event_name]["item_mapping"]:
                    event_mapping_file = mapping["item_index_file"]
                    item_index_cache_file = os.path.join(index_cache_dir, event_type, f"{event_mapping_file}.json")
                
                    if os.path.exists(item_index_cache_file):
                        # print(f"""Found cache file {item_index_cache_file} exist, direclt loading from cache file...""")
                        with open(item_index_cache_file, "r") as f:
                            index_datas = json.load(f)

                    else:
                        index_datas = mapping["item_index_preprocess_func"](os.path.join(self.origin_data_dir, event_type, f"{event_mapping_file}.csv"))
                        os.makedirs(item_index_cache_file.rsplit("/", 1)[0], exist_ok=True)
                        with open(item_index_cache_file, "w") as f:
                            json.dump(index_datas, f, indent=4, ensure_ascii=False)
                    
                    mapping["item_index"] = index_datas
                # mapping["string_convert_func"] = partial(mapping["string_convert_func"], item_indexing=index_datas)

    @staticmethod
    def mapping_item(item_lists, item_mappings):
        if not item_mappings:
            return item_lists

        for item in item_lists:
            for item_mapping in item_mappings:
                if item_mapping["input_name"] == "icd_code":
                    item[item_mapping["output_name"]] = item_mapping["item_index"][item["icd_version"]].get(item[item_mapping["input_name"]], None)
                
                elif item_mapping["input_name"] == "ndc":
                    ndc_code = item[item_mapping["input_name"]].split(".", 1)[0]
                    ndc_code = ("0" * (11 - len(ndc_code))) + ndc_code
                    item[item_mapping["output_name"]] = item_mapping["item_index"].get(ndc_code, None)
                
                else:
                    item[item_mapping["output_name"]] = item_mapping["item_index"].get(item[item_mapping["input_name"]], None)
        
        return item_lists
    
    def item_info_process(self, event_item, key_info):
        event_name = event_item["file_name"]
        if event_name not in self.event_info:
            return {}
        
        item_info = {}
        for task_name, task_key_info in key_info.items():
            item_info[task_name] = self.output_process(task_name, event_item, task_key_info)
        
        return item_info

    
    def input_process(self, event_item):
        # if event_item["file_name"] in self.event_info:
        #     return self.event_info[event_item["file_name"]]["string_convert_func"](event_item)
        # else:
        #     return ""

        event_name = event_item["file_name"]
        if event_name not in self.event_info:
            return ""

        item_mapping = self.event_info[event_name]["item_mapping"]
        info_keys = self.event_info[event_name]["info_keys"]

        item_lists = self.mapping_item(event_item["items"], item_mapping)
        file_name = event_item["file_name"]
        start_time = event_item["starttime"]

        title = "## {event_name} [{start_time}]".format(event_name=EVENT_NAME[file_name], start_time=start_time)
        item_str_list = []
        if len(item_lists) == 1:
            item = item_lists[0]
            for key in info_keys:
                item_str_list.append(f"- {key.title()}: {item.get(key, None)}")
            
        else:
            item_str_list.append(f"""| {" | ".join([key.title() for key in info_keys])} |""")
            item_str_list.append(f"""| {" | ".join(["------"] * len(info_keys))} |""")

            for item in item_lists:
                item_str_list.append(f"""| {" | ".join([str(item.get(key, None)) for key in info_keys])} |""")
        
        text = title + "\n" + "\n".join(item_str_list)
        return text

    
    def output_process(self, task_name, item, target_key):
        event_name = item["file_name"]
        if event_name not in self.event_info:
            return False

        item_lists = self.mapping_item(item["items"], self.event_info[event_name]["item_mapping"])
        
        # next event prediction task
        if task_name == "next_event":
            if safe_read(item["starttime"]):
                return item[target_key]
            else:
                return False

        output = [safe_read(_[target_key]) for _ in item_lists] # load all target items
        output = [o for o in output if o] # filter None items

        # discharge is generation task, return string instead of list
        if task_name == "discharge":
            if len(output) > 1:
                return False
            else:
                return output[0]
        
        # transfers task
        elif task_name == "transfers":
            output = []
            for item in item_lists:
                eventtype = item["eventtype"]
                careunit = item["careunit"]
                output.append(careunit if careunit != 'nan' else eventtype)
        
        elif task_name == "radiology":
            output = [o for output_list in output for o in output_list]
        
        output = [o.strip() for o in output if o.strip()]
        output = list(set(output))
        if output:
            return output
        else:
            return False