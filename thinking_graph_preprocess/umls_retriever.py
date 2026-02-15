import json
import os
import sqlite3
import time
import numpy as np
import os, sys
sys.path.append(os.path.abspath('./'))

import torch
from transformers import AutoTokenizer, AutoModel, AutoModelForSequenceClassification

os.environ["NLTK_DATA"] = "/dnn_training_sys/users/longquan.lys/evidence_datas/umls/nltk"
from quickumls import QuickUMLS

# device = "cuda"
# dense_model = AutoModel.from_pretrained("/dnn_training_sys/users/longquan.lys/embedding_models/MedCPT-Query-Encoder")
# dense_tokenizer = AutoTokenizer.from_pretrained("/dnn_training_sys/users/longquan.lys/embedding_models/MedCPT-Query-Encoder")
# rerank_tokenizer = AutoTokenizer.from_pretrained("/dnn_training_sys/users/longquan.lys/embedding_models/MedCPT-Cross-Encoder")
# rerank_model = AutoModelForSequenceClassification.from_pretrained("/dnn_training_sys/users/longquan.lys/embedding_models/MedCPT-Cross-Encoder", device_map=device)


@torch.no_grad()
def get_reranked_scores(query, articles, batch_size=32):
    pairs = [[query, article] for article in articles]
    all_logits = []
    for i in range(0, len(pairs), batch_size):
        batch_pairs = pairs[i:i+batch_size]
        encoded = rerank_tokenizer(
            batch_pairs,
            truncation=True,
            padding=True,
            return_tensors="pt",
            max_length=512,
        )
        encoded = {k: v.to(device) for k, v in encoded.items()}
        logits = rerank_model(**encoded).logits.squeeze(dim=1)
        all_logits.extend([i.item() for i in logits])
        del logits, encoded

    return all_logits

class UMLSRetriever:
    def __init__(self):
        db_path = '/dnn_training_sys/users/longquan.lys/evidence_datas/umls/retriever_organized_umls/umls.sqlite3'
        self.memory_conn = sqlite3.connect(':memory:', check_same_thread=False)
        # load to memory
        file_conn = sqlite3.connect(db_path)
        file_conn.backup(self.memory_conn)
        file_conn.close()
        # set to only-read mode
        self.memory_conn.execute('PRAGMA query_only = ON')
        self.memory_conn.execute('PRAGMA synchronous = OFF')
        self.memory_conn.execute('PRAGMA journal_mode = OFF')
        self.memory_conn.execute('PRAGMA temp_store = MEMORY')

        self.quickumls_search = QuickUMLS("/dnn_training_sys/users/longquan.lys/evidence_datas/umls/quickumls")
        
        self.cui_to_names = {}
        res = self.memory_conn.cursor().execute(f'SELECT * FROM MRCONSOEM').fetchall()
        for i in res:
            cui = i[0]
            name = i[2]
            if cui not in self.cui_to_names:
                self.cui_to_names[cui] = [set(), set()]
            if name.lower() not in self.cui_to_names[cui][1]:
                self.cui_to_names[cui][0].add(name)
                self.cui_to_names[cui][1].add(name.lower())
        self.cui_to_names = {k: sorted(list(v[0])) for k, v in self.cui_to_names.items()}
        

    def term_to_cui(self, term):
        # EXACT MATCH "COLLATE NOCASE" is set when creating the table!
        term = term.replace("\"", " ").strip()
        res = self.memory_conn.cursor().execute(f'SELECT * FROM MRCONSOEM WHERE STR="{term}" LIMIT 1').fetchone()
        if res is None:
            # FUZZY MATCH
            term = term.replace("'", " ").strip()
            res = self.memory_conn.cursor().execute(f'SELECT * FROM MRCONSO WHERE STR MATCH \'"{term}"\' ORDER BY rank LIMIT 1').fetchone()

        if res is not None:
            cui = res[0]
            return cui
        
        match_info = self.quickumls_search.match(term, best_match=True, ignore_syntax=False)
        if match_info:
            return match_info[0][0]['cui']

        return None
    
    def cui_to_definition(self, cui):
        res = self.memory_conn.cursor().execute(f'SELECT * FROM MRDEF WHERE CUI="{cui}"').fetchall()
        if res is not None:
            msh_def = None
            nci_def = None
            icf_def = None
            csp_def = None
            hpo_def = None
            other_def = None
            for definition in res:
                source = definition[1]
                if source == "MSH":
                    msh_def = definition[2]
                    break
                elif source == "NCI":
                    nci_def = definition[2]
                elif source == "ICF":
                    icf_def = definition[2]
                elif source == "CSP":
                    csp_def = definition[2]
                elif source == "HPO":
                    hpo_def = definition[2]
                else:
                    other_def = definition[2]
            defi = msh_def or nci_def or icf_def or csp_def or hpo_def or other_def
            return defi
        return None
    
    def term_to_relations(self, term):
        cui = self.term_to_cui(term)
        res = self.memory_conn.cursor().execute(f'SELECT CUI1,CUI2,STR1,RELA,STR2 FROM MRREL WHERE CUI1="{cui}" AND RELA!="subset includes concept" AND RELA!="concept in subset" AND RELA!="Has contraindicated drug" AND RELA!="Contraindicated with disease"').fetchall()
        if res is not None:
            res = list(set(res)) 

        relations = []
        for r in res:
            # reverse_relation = self.cuis_to_relations(r[1], cui)[0][1]
            relations.append([r[3], r[4]])

        return relations
    
    def cuis_to_relations(self, cui1, cui2):
        res = self.memory_conn.cursor().execute(f'SELECT STR1,RELA,STR2 FROM MRREL WHERE (CUI1="{cui1}" AND CUI2="{cui2}") AND RELA!="subset includes concept" AND RELA!="concept in subset" AND RELA!="Has contraindicated drug" AND RELA!="Contraindicated with disease"').fetchall()
        if res is not None:
            res = list(set(res))        
        return res
    
    def terms_to_relations(self, term1, term2):
        cui1 = self.term_to_cui(term1)
        cui2 = self.term_to_cui(term2)
        res = self.memory_conn.cursor().execute(f'SELECT STR1,RELA,STR2 FROM MRREL WHERE (CUI1="{cui1}" AND CUI2="{cui2}") AND (RELA!="subset includes concept")').fetchall()
        if res is not None:
            res = list(set(res))        
        return res

    def find_shortest_paths_bidirectional(self, start, end, max_length=10, max_paths=10, max_nodes=2000):
        from collections import deque

        if start == end:
            return [[start]]

        # Forward and backward search queues and visited dictionaries
        queue_forward = deque([[start]])
        queue_backward = deque([[end]])
        visited_forward = {start: [start]}
        visited_backward = {end: [end]}

        paths = []
        nodes_explored = 0

        while queue_forward and queue_backward and len(paths) < max_paths and nodes_explored < max_nodes:
            # Expand forward
            path_forward = queue_forward.popleft()
            last_node_forward = path_forward[-1]
            last_relation_forward = path_forward[-2] if len(path_forward) > 1 else None

            if last_node_forward in visited_backward:
                # Path found
                # path_backward =  visited_backward[last_node_forward][::-1][1:]
                full_path = path_forward + visited_backward[last_node_forward][::-1][1:]
                
                prev_relation = None
                no_reverse_relation = True
                refor_full_path = [full_path[0]]
                for i in range(2, len(full_path), 2):
                    prev_node = refor_full_path[-1]

                    relation_list = self.terms_to_relations(prev_node, full_path[i])
                    if relation_list is not None and len(relation_list) > 0:
                        relation = relation_list[0][1]
                    else:
                        relation = full_path[i-1]

                    reverse_relation_list = self.terms_to_relations(full_path[i], prev_node)
                    if reverse_relation_list is not None and len(relation_list) > 0:
                        reverse_relation = reverse_relation_list[0][1]
                    else:
                        reverse_relation = None


                    if reverse_relation != prev_relation:
                        refor_full_path += [relation, full_path[i]]
                        prev_relation = relation
                    else:
                        no_reverse_relation = False
                        break
                
                if no_reverse_relation:
                    paths.append(refor_full_path)

                if len(paths) >= max_paths:
                    break

            if len(path_forward) // 2 <= max_length:
                for relation, neighbor in self.term_to_relations(last_node_forward):
                    if neighbor not in visited_forward: # no same relation with the same node
                        visited_forward[neighbor] = path_forward + [relation, neighbor]
                        queue_forward.append(visited_forward[neighbor])
                        nodes_explored += 1

            # Expand backward
            path_backward = queue_backward.popleft()
            last_node_backward = path_backward[-1]
            last_relation_backward = path_backward[-2] if len(path_backward) > 1 else None

            if last_node_backward in visited_forward:
                # Path found
                full_path = visited_forward[last_node_backward] + path_backward[::-1][1:]
                paths.append(full_path)
                if len(paths) >= max_paths:
                    break

            if len(path_backward) // 2 <= max_length:
                for relation, neighbor in self.term_to_relations(last_node_backward):
                    if neighbor not in visited_backward: # no same relation with the same node:
                        visited_backward[neighbor] = path_backward + [relation, neighbor]
                        queue_backward.append(visited_backward[neighbor])
                        nodes_explored += 1

        if paths:
            paths = sorted(paths, key=len)
            return paths[0]
        else:
            return None

# umls_search = UMLS_Search()

def get_graph_docs(umls_search, term, query, topk=10):
    tmp_t = time.time()
    cui = umls_search.term_to_cui(term)
    if cui is not None:
        # 1. search
        definition = umls_search.cui_to_definition(cui)
        rels=umls_search.cui_to_relations(cui)
        # 2. rerank
        rel_texts = [f"{rel[0]} {rel[1]} {rel[2]}" for rel in rels]
        scores = get_reranked_scores(
            query=query,
            articles=rel_texts
        )
        zipped_score_rel = list(zip(scores, rels))
        zipped_score_rel.sort(key=lambda x: x[0], reverse=True)
        rerank_rels = [i[1] for i in zipped_score_rel[:topk]]

        relation = "; ".join([f"({rel[0]}, {rel[1]}, {rel[2]})" for rel in rerank_rels])
        para_text = f"Definition: {definition}" if definition else ""
        para_text += f"\nRelation: {relation}." if relation else ""
        print("graph_search:", time.time() - tmp_t)
        if para_text:
            return [{"title": "/".join(umls_search.cui_to_names[cui]), "para": para_text, "dataset": "umls"}]
    return []


if __name__ == "__main__":
    umls_search = UMLSRetriever()
    results = umls_search.find_shortest_paths_bidirectional('Specific Gravity', 'EW EMER.')
    print(results)
    # print(get_graph_docs(term="1-Carboxyglutamic Acid", query="what is it?", topk=10))