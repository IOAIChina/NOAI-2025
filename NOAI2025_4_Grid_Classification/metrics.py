import pandas as pd
import zipfile
import json
from sklearn.metrics import accuracy_score  # <— changed
from collections import defaultdict
import matplotlib.pyplot as plt
from PIL import Image
import os

SUBMISSION_ZIP = 'submission.zip'
GT_PATH = 'ground_truth_labels.csv'

from collections import defaultdict

def evaluate_per_label_accuracy(preds, labels):
    label_correct = defaultdict(int)
    label_total = defaultdict(int)

    for p, t in zip(preds, labels):
        label_total[t] += 1
        if p == t:
            label_correct[t] += 1

    per_label_acc = {}
    for label in label_total:
        acc = label_correct[label] / label_total[label]
        per_label_acc[label] = round(acc, 4)
    return per_label_acc

def load_predictions_from_zip(zip_path):
    with zipfile.ZipFile(zip_path, 'r') as z:
        with z.open('submissionA.csv') as f:
            pred_A = pd.read_csv(f, header=None)[0].tolist()
        with z.open('submissionB.csv') as f:
            pred_B = pd.read_csv(f, header=None)[0].tolist()
    return pred_A, pred_B

def load_ground_truth(gt_path):
    df = pd.read_csv(gt_path)
    labels_A = df[df['subset'] == 'A']['label'].tolist()
    labels_B = df[df['subset'] == 'B']['label'].tolist()
    return labels_A, labels_B

def load_ground_path(gt_path):
    df = pd.read_csv(gt_path)
    labels_A = df[df['subset'] == 'A']['img_name'].tolist()
    labels_B = df[df['subset'] == 'B']['img_name'].tolist()
    return labels_A, labels_B


def evaluate_accuracy(preds, labels):
    if len(preds) != len(labels):
        print(f"Length mismatch: preds={len(preds)}, labels={len(labels)}")
        return 0.0
    return accuracy_score(labels, preds)  # <— use accuracy

def save_score_json(acc_a, acc_b):
    score = {
        "status": True,
        "score": {
            "public_a": round(acc_a, 4),
            "private_b": round(acc_b, 4)
        },
        "msg": "Accuracy scoring completed."  # <— updated message
    }
    with open("score.json", "w") as f:
        json.dump(score, f, indent=2)
    print("Score written to score.json")

def visualize_misclassified(data_dir, ids, preds, labels, subset_name, max_display=10):
    """Show up to max_display images where pred!=label."""
    mis_idx = [i for i,(p,t) in enumerate(zip(preds, labels)) if p!=t][-max_display:]
    if not mis_idx:
        print(f"No misclassifications in subset {subset_name}")
        return

    cols = min(len(mis_idx), 5)
    rows = (len(mis_idx)+cols-1)//cols
    plt.figure(figsize=(cols*3, rows*3))
    for plot_i, idx in enumerate(mis_idx):
        img_id = ids[idx]
        # try .jpg then .png
        for ext in ('.jpg','.png'):
            path = os.path.join(data_dir, f"{img_id}{ext}")
            if os.path.exists(path):
                break
        img = Image.open(path).convert("RGB")

        ax = plt.subplot(rows, cols, plot_i+1)
        ax.imshow(img)
        ax.set_title(f"T={labels[idx]}  P={preds[idx]}")
        ax.axis('off')
    plt.suptitle(f"Misclassified in subset {subset_name}")
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    preds_a, preds_b = load_predictions_from_zip(SUBMISSION_ZIP)
    labels_a, labels_b = load_ground_truth(GT_PATH)
    path_a, path_b = load_ground_path(GT_PATH)

    acc_a = evaluate_accuracy(preds_a, labels_a)
    acc_b = evaluate_accuracy(preds_b, labels_b)

    # print(f"Overall Accuracy A: {acc_a:.4f}")
    # print(f"Overall Accuracy B: {acc_b:.4f}")

    # label_acc_a = evaluate_per_label_accuracy(preds_a, labels_a)
    # label_acc_b = evaluate_per_label_accuracy(preds_b, labels_b)

    # from sklearn.metrics import accuracy_score, classification_report

    # train_acc = accuracy_score(preds_a, labels_a)
    # print(f"\nVal Accuracy: {train_acc:.4f}")
    # print("Val Classification Report:")
    # print(classification_report(preds_a, labels_a, digits=4))

    # train_acc = accuracy_score(preds_b, labels_b)
    # print(f"\nTest Accuracy: {train_acc:.4f}")
    # print("Test Classification Report:")
    # print(classification_report(preds_b, labels_b, digits=4))


    # print("\nPer-label Accuracy A:")
    # for label, acc in label_acc_a.items():
    #     print(f"  Label {label}: {acc:.4f}")

    # print("\nPer-label Accuracy B:")
    # for label, acc in label_acc_b.items():
    #     print(f"  Label {label}: {acc:.4f}")

    # VAL_DIR = "./data_download/val"
    # TEST_DIR = "./data_download/test"
    # visualize_misclassified(VAL_DIR, path_a, preds_a, labels_a, 'A', max_display=30)
    # visualize_misclassified(TEST_DIR, path_b, preds_b, labels_b, 'B', max_display=30)

    save_score_json(acc_a, acc_b)
