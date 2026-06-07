import pandas as pd
import numpy as np
import re
from collections import Counter

from sklearn.metrics import accuracy_score
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer
from bert_score import score as bert_score


# =========================
# NORMALIZATION
# =========================
def normalize(text):
    if pd.isna(text):
        return ""
    text = str(text).lower().strip()
    text = re.sub(r"[^\w\s]", "", text)
    text = re.sub(r"\s+", " ", text)
    return text


# =========================
# F1 SCORE (token-level)
# =========================
def f1_score(pred, gt):
    pred_tokens = normalize(pred).split()
    gt_tokens = normalize(gt).split()

    common = Counter(pred_tokens) & Counter(gt_tokens)
    num_same = sum(common.values())

    if len(pred_tokens) == 0 and len(gt_tokens) == 0:
        return 1.0
    if num_same == 0:
        return 0.0

    precision = num_same / len(pred_tokens)
    recall = num_same / len(gt_tokens)

    return 2 * precision * recall / (precision + recall)


# =========================
# BLEU
# =========================
smooth = SmoothingFunction().method1

def bleu_score(pred, gt):
    ref = [normalize(gt).split()]
    hyp = normalize(pred).split()

    if len(hyp) == 0:
        return 0.0

    return sentence_bleu(ref, hyp, smoothing_function=smooth)


# =========================
# ROUGE-L
# =========================
rouge = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)

def rouge_l(pred, gt):
    scores = rouge.score(normalize(gt), normalize(pred))
    return scores["rougeL"].fmeasure


# =========================
# LOAD DATA
# =========================
df = pd.read_csv("/home/vinh/Documents/teacher_student/code/VQA-RAD/results/best_predictions.csv")

df = df.dropna(subset=["prediction", "answer"])


# =========================
# SPLIT DATA
# =========================
closed_df = df[df["question_type"].str.lower() == "closed"]
open_df   = df[df["question_type"].str.lower() == "open"]


# =========================
# CLOSED METRICS
# =========================
def evaluate_closed(data):
    data["pred_norm"] = data["prediction"].apply(normalize)
    data["gt_norm"] = data["answer"].apply(normalize)

    data["correct"] = data["pred_norm"] == data["gt_norm"]

    acc = data["correct"].mean()
    f1 = data.apply(lambda x: f1_score(x["prediction"], x["answer"]), axis=1).mean()

    return acc, f1


# =========================
# OPEN METRICS
# =========================
def evaluate_open(data):
    bleu = data.apply(lambda x: bleu_score(x["prediction"], x["answer"]), axis=1).mean()
    rouge_l_score = data.apply(lambda x: rouge_l(x["prediction"], x["answer"]), axis=1).mean()

    # BERTScore (semantic similarity)
    P, R, F1 = bert_score(
        data["prediction"].tolist(),
        data["answer"].tolist(),
        lang="en",
        verbose=False
    )

    return bleu, rouge_l_score, F1.mean().item()


# =========================
# RUN EVALUATION
# =========================

print("\n================ CLOSED QUESTIONS ================")
closed_acc, closed_f1 = evaluate_closed(closed_df)

print(f"Samples   : {len(closed_df)}")
print(f"Accuracy  : {closed_acc:.4f}")
print(f"F1-score  : {closed_f1:.4f}")


print("\n================ OPEN QUESTIONS ================")
bleu, rouge_l_score, bert_f1 = evaluate_open(open_df)

print(f"Samples   : {len(open_df)}")
print(f"BLEU      : {bleu:.4f}")
print(f"ROUGE-L   : {rouge_l_score:.4f}")
print(f"BERTScore : {bert_f1:.4f}")


# =========================
# OVERALL SUMMARY (optional)
# =========================
print("\n================ SUMMARY ================")
print(f"Closed Acc   : {closed_acc:.4f}")
print(f"Closed F1    : {closed_f1:.4f}")
print(f"Open BLEU    : {bleu:.4f}")
print(f"Open ROUGE-L : {rouge_l_score:.4f}")
print(f"Open BERTS   : {bert_f1:.4f}")