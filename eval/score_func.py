from thefuzz import fuzz

def calculate_em(true_set, pred_set):
    true_set = set(true_set)
    true_set = {item.lower() for item in true_set}
    pred_set = set(pred_set)
    pred_set = {item.lower() for item in pred_set}
    
    # 计算交集部分
    intersection = len(true_set.intersection(pred_set))
    
    # Precision = TP / (TP + FP)
    precision = intersection / len(pred_set) if len(pred_set) > 0 else 0
    
    # Recall = TP / (TP + FN)
    recall = intersection / len(true_set) if len(true_set) > 0 else 0
    
    # F1 = 2 * (Precision * Recall) / (Precision + Recall)
    if precision + recall > 0:
        f1 = 2 * (precision * recall) / (precision + recall)
    else:
        f1 = 0
    
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1
    }

def calculate_acc(true_string, pred_string):
    # assert isinstance(true_string, str) and isinstance(pred_string, str)
    if not isinstance(true_string, str):
        true_string = str(true_string).lower()
    if not isinstance(pred_string, str):
        pred_string = str(pred_string).lower()

    return {"acc": 1.0} if pred_string.startswith(true_string) else {"acc": 0.0}

from rouge_score import rouge_scorer

# 计算 ROUGE 分数
def calculate_rouge(reference, hypothesis, metric="rouge1"):
    scorer = rouge_scorer.RougeScorer(
        [metric], 
        use_stemmer=True
    )
    scores = scorer.score(reference, hypothesis)

    return {
        "precision": scores[metric].precision,
        "recall": scores[metric].recall,
        "f1": scores[metric].fmeasure
    }

def calculate_cdm(true_string, pred_string):
    # assert isinstance(true_string, str) and isinstance(pred_string, str)
    true_string = true_string.lower()
    pred_string = pred_string.lower()
    # pred_set = set(pred_set)
    # pred_set = {item.lower() for item in pred_set}

    # for pred in pred_set:
    #     if fuzz.ratio(pred, true_string) > 90 or true_string in pred:
            # return {"acc": 1.0}
    
    if true_string in pred_string:
        return {"acc": 1.0}

    return {"acc": 0.0}
    

from rouge_score import rouge_scorer

if __name__=="__main__":
    # 示例文本
    reference_text = "The quick brown fox jumped over the lazy dog."
    hypothesis_text = "A fast brown fox leaps over a lazy dog."

    # 计算 ROUGE 分数
    scores = calculate_rouge(reference_text, hypothesis_text)
    print(scores)