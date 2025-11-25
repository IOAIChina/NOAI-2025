import pandas as pd
import zipfile
import json
from sklearn.metrics import f1_score

SUBMISSION_ZIP = "submission.zip"
GT_PATH = "ground_truth_labels.csv"


def load_predictions_from_zip(zip_path):
    with zipfile.ZipFile(zip_path, "r") as z:
        with z.open("submissionA.csv") as f:
            pred_A = pd.read_csv(f, header=None)[0].tolist()
        with z.open("submissionB.csv") as f:
            pred_B = pd.read_csv(f, header=None)[0].tolist()
    return pred_A, pred_B


def load_ground_truth(gt_path):
    df = pd.read_csv(gt_path)
    labels_A = df[df["subset"] == "A"]["label"].tolist()
    labels_B = df[df["subset"] == "B"]["label"].tolist()
    return labels_A, labels_B


def evaluate(preds, labels):
    if len(preds) != len(labels):
        print(f"Length mismatch: preds={len(preds)}, labels={len(labels)}")
        return 0.0
    return f1_score(labels, preds, average="macro")


def save_score_json(f1_a, f1_b):
    result = {
        "status": True,
        "score": {"public_a": round(f1_a, 4), "private_b": round(f1_b, 4)},
        "msg": "F1 scoring completed.",
    }
    with open("score.json", "w") as f:
        json.dump(result, f, indent=2)
    print("Score written to score.json")


if __name__ == "__main__":
    preds_a, preds_b = load_predictions_from_zip(SUBMISSION_ZIP)
    labels_a, labels_b = load_ground_truth(GT_PATH)

    f1_a = evaluate(preds_a, labels_a)
    f1_b = evaluate(preds_b, labels_b)

    save_score_json(f1_a, f1_b)
