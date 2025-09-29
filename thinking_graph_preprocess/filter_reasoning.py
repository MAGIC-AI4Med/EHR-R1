import pickle
import pandas as pd
import os
from tqdm import tqdm
import random

if __name__ == '__main__':
    INPUT_PATHS = [
        # "/dnn_training_sys/users/longquan.lys/datas/evidence_datas/evidence_reasoning/train_all_tasks_balance_0.2_weight_minsupport5_threshold5_topk20/gpt-4o.csv",
        # "/dnn_training_sys/users/longquan.lys/datas/evidence_datas/evidence_reasoning/train_all_tasks_balance_0.2_unweight_-1case_based_on_coexist_concepts_umls_lift_minsupport0_threshold50_topk20/gpt-4o.csv",
        # "/dnn_training_sys/users/longquan.lys/datas/evidence_datas/evidence_reasoning/train_extra_reason_tasks_40000_weight_minsupport5_threshold5_topk20/gpt-4o.csv",
        # "/dnn_training_sys/users/longquan.lys/datas/evidence_datas/evidence_reasoning/train_decision_making_0.1_unweight/gpt-4o-wo-knowledge-new.csv",
        # "/dnn_training_sys/users/longquan.lys/datas/evidence_datas/evidence_reasoning/train_decision_making_0.1_unweight_effect_context/gpt-4o-wo-knowledge-new.csv",
        "/dnn_training_sys/users/longquan.lys/datas/evidence_datas/evidence_reasoning/train_all_tasks_balance_0.2_weight_minsupport5_threshold5_topk20/gpt-4o-new.csv",
        # "/dnn_training_sys/users/longquan.lys/datas/evidence_datas/evidence_reasoning/train_extra_reason_tasks_10000_weight/gpt-4o-wo-knowledge-new.csv"
    ]
    OUTPUT_PATH = "/dnn_training_sys/users/longquan.lys/datas/evidence_datas/evidence_reasoning/all/gpt-4o-wo-knowledge-new-clean-290k.csv"
    TASK_LIST = None
    TARGET_SIZE = -1

    df = pd.DataFrame()
    for input_path in INPUT_PATHS:
        df = pd.concat([df, pd.read_csv(input_path)])

    print(df.shape[0])
    print(df["task"].value_counts())
    
    df = df[df["task"] != "discharge"]
    if TASK_LIST is not None:
        df = df[df["task"].isin(TASK_LIST)]
    datas = df.to_dict(orient="records")

    clean_datas = []
    for data in tqdm(datas):
        try:
            reasoning_list = [output["message"]["content"] for output in eval(data["reasoning"])["choices"]]
        except:
            # print("Error: ", eval(data["reasoning"]))
            continue

        clean_reasoning_list = []
        for reasoning in reasoning_list:
            prediction = reasoning.rsplit("Final Results")[-1]
            flag = True
            label_predicted_num = 0
            for label in eval(data["target"]):
                # print(label)
                if label in prediction:
                    label_predicted_num += 1
            
            label_predicted_ratio = label_predicted_num / len(eval(data["target"]))
            if label_predicted_ratio < 0.7:
                flag = False
            
            if "ground truth" in reasoning.lower():
                flag = False

            if "Extraction and Analysis" in reasoning or "Reasoning and Summary" in reasoning:
                flag = False

            if flag:
                clean_reasoning_list.append(reasoning)
      
        if len(clean_reasoning_list) > 0:
            data["reasoning"] = clean_reasoning_list
            clean_datas.append(data)

    if TARGET_SIZE < 0:
        keep_len = len(clean_datas) - (len(clean_datas) % 64)
        clean_datas = random.sample(clean_datas, keep_len)
    else:
        clean_datas = random.sample(clean_datas, TARGET_SIZE)
    clean_df = pd.DataFrame(clean_datas)

    print("Final Data Size: ", clean_df.shape[0])
    print(clean_df["task"].value_counts())
    # clean_df.to_csv(OUTPUT_PATH, index=False)
