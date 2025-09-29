import os
import sys

import pickle
import json
import time
import random
import requests
import argparse
import pandas as pd
from tqdm import tqdm
import pyarrow.parquet as pq
from joblib import Parallel, delayed

from mimiciv_dataset.mimiciv import MIMICIV
# from evidence_preprocess.pubmed_retriever import PubMedRetriever, extract_relationships
from evidence_preprocess.umls_retriever import UMLSRetriever

umls_retriever = UMLSRetriever()

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

def get_umls_graph(term_pairs):
    
    def umls_search_local(pair):
        cui1 = umls_retriever.term_to_cui(pair[0])
        cui2 = umls_retriever.term_to_cui(pair[1])
        
        if cui1 is not None and cui2 is not None:
            rels = umls_retriever.find_shortest_paths_bidirectional(pair[0], pair[1])
        else:
            rels = False

        return rels

    # tmp_t = time.time()
    umls_pair = []
    cui_faile_num = 0
    for pair in term_pairs:
        rels = umls_search_local(pair)
        if rels is not False:
            if rels is not None:
                umls_pair.append(rels)
        else:
          cui_faile_num += 1
    
    # print(f"""total_pair={len(term_pairs)}, success_pair={len(umls_pair)}, cui_faile_num={cui_faile_num}""")
    return umls_pair

def get_concept_with_coexist(target_concepts, history_concepts, coexist_concepts_prior, threshold, topk, total_pair):
    pairs = []
    concepts_cluster = {}
    for target_concept in target_concepts:
        coexist_concepts_dict = coexist_concepts_prior.get(target_concept, {})
        if not coexist_concepts_dict:
            print(f"term [{target_concept}] not in coexist_concepts_prior...")
            continue

        # coexist_concepts_list = {item:score for concept, score in coexist_concepts_dict.items() if score > threshold}
        coexist_concepts_list = {item[0]:item[1] for item in sorted(coexist_concepts_dict.items(), key=lambda x: x[1], reverse=True) if item[1] > threshold}
            
        for history_concept in history_concepts:
            if history_concept == target_concept:
                continue
            
            if history_concept in coexist_concepts_list:
                if target_concept not in concepts_cluster:
                    concepts_cluster[target_concept] = {}
                concepts_cluster[target_concept][history_concept] = coexist_concepts_list[history_concept]
    
    if not concepts_cluster:
        pass

    topk_per_target_concepts = topk
    for target_concept in concepts_cluster:
        coexist_concepts_list = {item[0]:item[1] for item in sorted(concepts_cluster[target_concept].items(), key=lambda x: x[1], reverse=True)[:topk_per_target_concepts]}
        concepts_cluster[target_concept] = coexist_concepts_list

        for coexist_concepts in coexist_concepts_list:
            pairs.append([target_concept, coexist_concepts])
            
    return concepts_cluster, pairs

def filter_graph(graph, target_concepts, history_concepts):
    filtered_graph = []
    for sample in graph:
        if (sample[0] in target_concepts and sample[-1] in history_concepts) or (sample[0] in history_concepts and sample[-1] in target_concepts):
            filtered_graph.append(sample)
    
    return filtered_graph

def get_all_event_concepts(event_concept):
    all_event_concept = []
    for concept_type in event_concept:
        if isinstance(event_concept[concept_type], list):
            all_event_concept += event_concept[concept_type]
        else:
            continue
    
    all_event_concept = list(set(all_event_concept))
    return all_event_concept

def get_sample_evidence(args, sample_info, coexist_concepts_prior):
    if DATASET.task_info[sample_info["task"]]["task_type"] != "decision_making":
    # if sample_info["event"] != "diagnoses_icd":
        return []

    subject_id = sample_info["subject_id"]
    try:
        with open(os.path.join(args.event_concept_dir, f"{subject_id}.pkl"), "rb") as f:
            event_concept_list = pickle.load(f)
    except:
        print(f"""Loading {os.path.join(args.event_concept_dir, f"{subject_id}.pkl")} failed!""")
        return []
    
    task_name = sample_info["task"]
    if task_name not in coexist_concepts_prior:
        return []

    target_concepts = event_concept_list[sample_info["period_end"]]["concepts"].get(task_name, [])

    history_concepts = [get_all_event_concepts(event_concept["concepts"]) for event_concept in event_concept_list[sample_info["period_begin"]:sample_info["period_end"]]]
    if safe_read(sample_info.get("last_discharge_id", None)):
        history_concepts += [get_all_event_concepts(event_concept_list[int(sample_info["last_discharge_id"])]["concepts"])]
    if safe_read(sample_info.get("admissions_id", None)):
        history_concepts += [get_all_event_concepts(event_concept_list[int(sample_info["admissions_id"])]["concepts"])]
    history_concepts = [c for concept in history_concepts for c in concept]

    target_concepts, history_concepts = list(set(target_concepts)), list(set(history_concepts))

    # print("Begin gathering prior relation...")
    concepts_cluster_w_score, pairs = get_concept_with_coexist(target_concepts, history_concepts, coexist_concepts_prior[task_name], args.threshold, args.topk, args.total_pair)
    concepts_cluster = {}
    for key in concepts_cluster_w_score:
        concepts_cluster[key] = list(concepts_cluster_w_score[key].keys())

    if len(pairs) == 0: 
        return []

    # print("Begin gathering umls graph...")
    umls_graph = get_umls_graph(pairs)
    # print(f"umls graph: {umls_graph}")
    if not umls_graph:
        return []

    # print("Begin gathering pubmed graph...")
    # pubmed_graph = get_pubmed_graph(concepts_cluster)
    # print(f"pubmed graph: {pubmed_graph}")

    final_graph =  {
        "umls_graph": filter_graph(umls_graph, target_concepts, history_concepts),
        # "pubmed_graph": filter_graph(pubmed_graph, target_concepts, history_concepts)
    }

    if not final_graph["umls_graph"]:
        return {}

    # print(f"final graph: {final_graph}")
    return final_graph

def load_coexist_concepts(args, task_list):
    task_wise_coexist_concepts = {}
    for task in tqdm(os.listdir(args.coexist_concept_dir)):
        if task.endswith(".pkl"):
            task_name = task.rsplit(".", 1)[0]
            if task_name not in task_list:
                continue

            file_path = os.path.join(args.coexist_concept_dir, task)

            if args.task is not None and task_name != args.task:
                continue

            with open(file_path, 'rb') as f:
                task_wise_coexist_concepts_info = pickle.load(f)
                coexists_concepts = {}
                for target_concept in task_wise_coexist_concepts_info["coexist_metric"]:
                    coexists_concepts[target_concept] = {}
                    for concept in task_wise_coexist_concepts_info["coexist_metric"][target_concept]:
                        if task_wise_coexist_concepts_info["concept_num"][concept] < args.min_support:
                            continue
                        
                        coexists_concepts[target_concept][concept] = task_wise_coexist_concepts_info["coexist_metric"][target_concept][concept] * task_wise_coexist_concepts_info["task_num"] / (task_wise_coexist_concepts_info["concept_num"][concept] * task_wise_coexist_concepts_info["concept_num"][target_concept])

                task_wise_coexist_concepts[task_name] = coexists_concepts

    return task_wise_coexist_concepts

def parse_args():
    parser = argparse.ArgumentParser(prog="EHR Data Filter and Selection")

    # basic args
    parser.add_argument("--data_index_path", type=str, required=True)
    parser.add_argument("--event_concept_dir", type=str, default="./datas/evidence_datas/event_concepts")
    parser.add_argument("--coexist_concept_dir", type=str, default="./datas/evidence_datas/task_wise_coexist_concepts")
    parser.add_argument("--output_path", type=str, default="./datas/evidence_datas/extracted_concepts_per_sample")
    parser.add_argument("--filter_nograph", action="store_true", default=False)
    parser.add_argument("--save_step", type=int, default=300)
    parser.add_argument("--topk", type=int, default=20)
    parser.add_argument("--total_pair", type=int, default=100)
    parser.add_argument("--chunk_num", type=int, default=1)
    parser.add_argument("--chunk_idx", type=int, default=0)
    parser.add_argument("--resume", action="store_true", default=False)

    parser.add_argument("--task", type=str, default=None)
    parser.add_argument("--threshold", type=int, default=5)
    parser.add_argument("--min_support", type=int, default=5)
    args = parser.parse_args()

    # coexist_concept_name = args.coexist_concept_dir.rsplit("/", 1)[-1]
    data_index_name = args.data_index_path.rsplit("/", 1)[-1].rsplit(".", 1)[0]

    args.log_path = os.path.join(args.output_path, f"{data_index_name}_minsupport{args.min_support}_threshold{args.threshold}_topk{args.topk}", f"{args.chunk_num}_{args.chunk_idx}.log")
    args.output_path = os.path.join(args.output_path, f"{data_index_name}_minsupport{args.min_support}_threshold{args.threshold}_topk{args.topk}", f"{args.chunk_num}_{args.chunk_idx}.csv")

    if not os.path.exists(os.path.dirname(args.output_path)):
        os.makedirs(os.path.dirname(args.output_path))

    return args

def process_evidence(args, sample_info, coexist_concepts_prior):
    sample_graph = get_sample_evidence(args, sample_info, coexist_concepts_prior)
    if sample_graph:
        sample_info["graph"] = json.dumps(sample_graph)
    else:
        sample_info["graph"] = None
    return sample_info

def get_data_chunk(sample_infos, chunk_num, chunk_idx):
    chunk_size = len(sample_infos) // chunk_num + 1
    return sample_infos[chunk_idx*chunk_size:(chunk_idx+1)*chunk_size]

if __name__ == "__main__":
    args = parse_args()
    if args.chunk_num > 1:
        output_f = open(args.log_path, 'w')
        sys.stdout = output_f
        sys.stderr = output_f

    DATASET = MIMICIV(sample_info=["10000032"], lazzy_mode=True)

    sample_info_df = pd.read_csv(args.data_index_path)
    sample_infos = sample_info_df.to_dict(orient='records')

    task_list = [args.task] if args.task is not None else sample_info_df["task"].unique().tolist()
    coexist_concepts_prior = load_coexist_concepts(args, task_list)

    sample_infos = [sample_info for sample_info in sample_infos if DATASET.task_info[sample_info["task"]]["task_type"] == "decision_making" and (args.task is None or sample_info["task"] == args.task)]
    random.seed(42)
    random.shuffle(sample_infos)
    sample_infos = get_data_chunk(sample_infos, args.chunk_num, args.chunk_idx)
    
    clean_datas = []
    evidence_data_num = 0
    if args.resume:
        try:
            clean_df = pd.read_csv(args.output_path)
            clean_datas = clean_df.to_dict(orient='records')
            evidence_data_num = sum([bool(safe_read(data["graph"])) for data in clean_datas])
            print(f"Loading {len(clean_datas)} cache datas with {evidence_data_num} evidence datas...")
        except:
            pass
        
    
    sample_infos = sample_infos[len(clean_datas):]
    print(f"Total processing sample num: {len(sample_infos)}")
    for idx, sample_info in tqdm(enumerate(sample_infos), total=len(sample_infos)):
        data = process_evidence(args, sample_info, coexist_concepts_prior)
        clean_datas.append(data)

        if data["graph"]:
            evidence_data_num += 1
        
        print(f"Processing {idx+1} datas. Getting {evidence_data_num} data with evidence...")
        if idx % args.save_step == 0 and idx != 0:
            clean_datas_df = pd.DataFrame(clean_datas)
            clean_datas_df.to_csv(args.output_path, index=False)

    print(f"Get final evidence data num: {evidence_data_num}")
    clean_datas_df = pd.DataFrame(clean_datas)
    clean_datas_df.to_csv(args.output_path, index=False)

    if args.chunk_num > 1:
        output_f.close()

