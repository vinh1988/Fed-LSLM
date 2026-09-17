#!/usr/bin/env python3
import os
import argparse
import pandas as pd
from bert_score import score

def main():
    parser = argparse.ArgumentParser(description="Calculate metrics and BERTScore for PathVQA predictions")
    parser.add_argument(
        "--csv_path", 
        type=str, 
        default="/home/vinh/Documents/teacher_student/code/PathVQA/results/best_predictions.csv",
        help="Path to predictions CSV file"
    )
    args = parser.parse_args()

    csv_path = args.csv_path
    if not os.path.exists(csv_path):
        print(f"Error: file not found at {csv_path}")
        return

    print(f"Loading predictions from: {csv_path}")
    df = pd.read_csv(csv_path)

    # =====================================================
    # BERTSCORE
    # =====================================================
    print("Computing BERTScore...")
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
    if len(closed_df) > 0:
        print(f"Accuracy : {closed_df['EM'].mean():.4f}")
    else:
        print("Accuracy : N/A")

    # =====================================================
    # OPEN QUESTIONS
    # =====================================================
    print("\n==============================")
    print(" OPEN QUESTIONS")
    print("==============================")
    print(f"Samples    : {len(open_df)}")
    if len(open_df) > 0:
        print(f"EM         : {open_df['EM'].mean():.4f}")
        print(f"F1         : {open_df['F1'].mean():.4f}")
        print(f"BLEU       : {open_df['BLEU'].mean():.4f}")
        print(f"ROUGE-L    : {open_df['ROUGE-L'].mean():.4f}")
        print(f"BERTScore  : {open_df['BERTScore'].mean():.4f}")
    else:
        print("Metrics    : N/A")

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
    output_dir = os.path.dirname(csv_path)
    output_path = os.path.join(output_dir, "best_predictions_with_bertscore.csv")
    df.to_csv(output_path, index=False)
    print(f"\nSaved predictions with BERTScore to: {output_path}")

    # =====================================================
    # PAPER TABLE
    # =====================================================
    closed_acc = closed_df["EM"].mean() if len(closed_df) > 0 else None
    open_acc = open_df["EM"].mean() if len(open_df) > 0 else None
    open_bleu = open_df["BLEU"].mean() if len(open_df) > 0 else None
    open_rouge = open_df["ROUGE-L"].mean() if len(open_df) > 0 else None
    open_f1 = open_df["F1"].mean() if len(open_df) > 0 else None
    open_bert = open_df["BERTScore"].mean() if len(open_df) > 0 else None

    paper_table = pd.DataFrame([
        {
            "Split": "Closed",
            "Accuracy": closed_acc,
            "BLEU": None,
            "ROUGE-L": None,
            "F1": None,
            "BERTScore": None
        },
        {
            "Split": "Open",
            "Accuracy": open_acc,
            "BLEU": open_bleu,
            "ROUGE-L": open_rouge,
            "F1": open_f1,
            "BERTScore": open_bert
        }
    ])

    print("\n==============================")
    print(" PAPER TABLE")
    print("==============================")
    print(paper_table.round(4))

if __name__ == "__main__":
    main()
