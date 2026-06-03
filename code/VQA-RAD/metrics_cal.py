import pandas as pd
from bert_score import score

# =====================================================
# LOAD
# =====================================================

csv_path = "/home/vinh/Documents/teacher_student/code/VQA-RAD/results/best_predictions_10eps.csv"

df = pd.read_csv(csv_path)

# =====================================================
# BERTSCORE
# =====================================================

preds = df["prediction"].fillna("").astype(str).tolist()
refs = df["answer"].fillna("").astype(str).tolist()

_, _, bert_f1 = score(
    preds,
    refs,
    lang="en",
    verbose=True
)

df["BERTScore"] = bert_f1.cpu().numpy()

# =====================================================
# SPLITS
# =====================================================

closed_df = df[df["question_type"].str.lower() == "closed"]
open_df = df[df["question_type"].str.lower() == "open"]

# =====================================================
# CLOSED QUESTIONS
# =====================================================

print("\n==============================")
print(" CLOSED QUESTIONS")
print("==============================")

print(f"Samples  : {len(closed_df)}")
print(f"Accuracy : {closed_df['EM'].mean():.4f}")

# =====================================================
# OPEN QUESTIONS
# =====================================================

print("\n==============================")
print(" OPEN QUESTIONS")
print("==============================")

print(f"Samples    : {len(open_df)}")
print(f"EM         : {open_df['EM'].mean():.4f}")
print(f"F1         : {open_df['F1'].mean():.4f}")
print(f"BLEU       : {open_df['BLEU'].mean():.4f}")
print(f"ROUGE-L    : {open_df['ROUGE-L'].mean():.4f}")
print(f"BERTScore  : {open_df['BERTScore'].mean():.4f}")

# =====================================================
# OVERALL
# =====================================================

print("\n==============================")
print(" OVERALL")
print("==============================")

print(f"Samples    : {len(df)}")
print(f"EM         : {df['EM'].mean():.4f}")
print(f"F1         : {df['F1'].mean():.4f}")
print(f"BERTScore  : {df['BERTScore'].mean():.4f}")

# =====================================================
# SAVE
# =====================================================

df.to_csv(
    "best_predictions_with_bertscore.csv",
    index=False
)

# =====================================================
# PAPER TABLE
# =====================================================

paper_table = pd.DataFrame([
    {
        "Split": "Closed",
        "Accuracy": closed_df["EM"].mean(),
        "BLEU": None,
        "ROUGE-L": None,
        "F1": None,
        "BERTScore": None
    },
    {
        "Split": "Open",
        "Accuracy": open_df["EM"].mean(),
        "BLEU": open_df["BLEU"].mean(),
        "ROUGE-L": open_df["ROUGE-L"].mean(),
        "F1": open_df["F1"].mean(),
        "BERTScore": open_df["BERTScore"].mean()
    }
])

print("\n==============================")
print(" PAPER TABLE")
print("==============================")
print(paper_table.round(4))