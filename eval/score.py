import re
import pdb
import json
import torch
import numpy as np
from collections import Counter
from transformers import AutoTokenizer, AutoModelForSequenceClassification


def extracted_postprocess(answer):
    return answer.strip(" \n*{")

def detect_answer_in_response(prediction):
    match = re.search(r'(.*)[T|t]he answer is (.*?)(\.|$)', prediction, re.DOTALL)
    return True if match else False

def extract_predicted_answer_util_end(prediction):
    match = re.search(r'(.*)[T|t]he answer is (.*?)$', prediction, re.DOTALL)

    return extracted_postprocess(match.group(2)) if match else prediction[-100:]

def extract_predicted_answer(prediction):
    match = re.search(r'(.*)[T|t]he answer is (.*?)(\.|$)', prediction, re.DOTALL)

    return extracted_postprocess(match.group(2)) if match else prediction[-100:]

def single_choice_score(prediction, eval_info):
    prediction = extract_predicted_answer(prediction)
    if prediction is None:
        return False

    answer = eval_info["answer"]
    answer_idx = eval_info["answer_idx"]

    if answer.lower() in prediction.lower() or (answer_idx + ".") in prediction:
        return True
    
    elif len(prediction) == 1 and prediction == answer_idx:
        return True
    
    else:
        return False

def entity_match_score(prediction, eval_info):
    prediction = extract_predicted_answer_util_end(prediction)
    if prediction is None:
        return False
    
    answer = eval_info["answer"]

    if answer.lower() in prediction.lower():
        return True
    
    else:
        return False

def estimate_pass_at_k(num_sample, num_correct, k):
    """Estimates pass@k of each problem and returns them in an array."""

    def estimator(n: int, c: int, k: int) -> float:
        """Calculates 1 - comb(n - c, k) / comb(n, k)."""
        if n - c < k:
            return 1.0
        return (1.0 - np.prod(1.0 - k / np.arange(n - c + 1, n + 1)))

    return estimator(int(num_sample), int(num_correct), k)

verifier_model = None
def verifier_score(predictions, eval_info):
    global verifier_model
    
    if verifier_model is None:
        verifier_model = VerifierModel()
        
    predictions = [predictions] if isinstance(predictions, str) else predictions
    predictions = [extract_predicted_answer_util_end(prediction) for prediction in predictions]
    
    labels = [eval_info["answer"]] * len(predictions)
    probabilities = verifier_model(predictions, labels)
    results = [True if probability[1] > 0.5 else False for probability in probabilities]
    return results if len(results) > 1 else results[0]

SCORE_FUNC = {
    # "math": math_score,
    # "gsm8k": gsm8k_score,
    "medqa": single_choice_score,
    "medmcqa": single_choice_score,
    "mmlu": single_choice_score,
    "pubmedqa": single_choice_score,
    "bioasq": single_choice_score,
    "medqa_5op": single_choice_score,
    "mmlu_medcare": single_choice_score,
    "medmcqa": single_choice_score,
    "medsins": entity_match_score,
    "medqa_open": entity_match_score,
    "medical_o1": verifier_score,
    "open_mcqa": verifier_score,
}

def score_task(task: dict, task_name: str = None):
    score = {}

    # find score func
    task_name = task_name if task_name is not None else task["task"]["dataset"]
    if task_name in SCORE_FUNC:
        score_func = SCORE_FUNC[task_name]

    elif any(dataset_name in task_name for dataset_name in SCORE_FUNC):
        for dataset_name in SCORE_FUNC:
            if dataset_name in task_name:
                score_func = SCORE_FUNC[dataset_name]
                break
    else:
        assert "answer_idx" in task["task"]["eval"] # default is the multiple-choice question
        score_func = single_choice_score
    
    
    # get answer acc
    assert "trajectory" in task
    if isinstance(task["trajectory"], str):
        score["acc"] = score_func(task["trajectory"], eval_info=task["task"]["eval"])
        task["prediction"] = extract_predicted_answer(task["trajectory"])
         
    elif (isinstance(task["trajectory"], list) and len(task["trajectory"]) == 1):
        score["acc"] = score_func(task["trajectory"][0], eval_info=task["task"]["eval"])
        task["prediction"] = extract_predicted_answer(task["trajectory"][0])
        
    elif isinstance(task["trajectory"], list):
        if score_func.__name__ == "verifier_score":
            # batch forward
            accuracy = score_func(task["trajectory"], eval_info=task["task"]["eval"])
        else:
            accuracy = [score_func(trajectory, eval_info=task["task"]["eval"]) for trajectory in task["trajectory"]]
            
        predictions = [extract_predicted_answer(trajectory) for trajectory in task["trajectory"]]
            
        vote_prediction = Counter(predictions).most_common()[0][0]
        vote_acc = score_func(vote_prediction, eval_info=task["task"]["eval"])
        score["acc"] = vote_acc
        score["avg_acc"] = sum(accuracy) / len(accuracy)
        score["least_acc"] = float(sum(accuracy) > 0)
        task["prediction"] = vote_prediction
        
        classified_trajectories = {"positive": [], "negative": []}
        for traj_idx, acc in enumerate(accuracy):
            if acc:
                classified_trajectories["positive"].append(traj_idx)
            else:
                classified_trajectories["negative"].append(traj_idx)
        task["classified_id"] = classified_trajectories
    
    else:
        raise NotImplementedError
    
    # get answer pass@k
    if "trajectory" in task and len(task["trajectory"]) > 1:
        sample_num = len(task["trajectory"])
        correct_num = sum(accuracy)
        for k in range(1, sample_num+1):
            score[f"pass@{k}"] = estimate_pass_at_k(sample_num, correct_num, k)
        
    return score

