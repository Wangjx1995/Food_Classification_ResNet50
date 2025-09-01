import numpy as np
import torch
import os
from tqdm import tqdm
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
def evaluate_and_report(model, loader, class_names, device, out_dir):

    os.makedirs(out_dir, exist_ok=True)
    model.eval(); torch.set_grad_enabled(False)

    all_preds, all_targets = [], []
    for images, targets in tqdm(loader, desc="Eval", leave=False):
        images  = images.to(device, non_blocking=True).float()
        targets = targets.to(device, non_blocking=True)
        logits  = model(images)
        preds   = logits.argmax(1)
        all_preds.extend(preds.detach().cpu().tolist())
        all_targets.extend(targets.detach().cpu().tolist())

    all_preds   = np.asarray(all_preds, dtype=int)
    all_targets = np.asarray(all_targets, dtype=int)

    if all_targets.size == 0:
        raise RuntimeError("No val_set")


    num_classes = len(class_names)
    labels_all  = list(range(num_classes))


    cm = confusion_matrix(all_targets, all_preds, labels=labels_all)
    cm_df = pd.DataFrame(cm,
                         index=[f"true_{c}" for c in class_names],
                         columns=[f"pred_{c}" for c in class_names])
    cm_df.to_csv(os.path.join(out_dir, "confusion_matrix.csv"), encoding="utf-8-sig")


    row_sum = cm.sum(axis=1, keepdims=True)
    cm_norm = np.divide(cm, np.clip(row_sum, 1, None))
    cm_norm_df = pd.DataFrame(cm_norm,
                              index=[f"true_{c}" for c in class_names],
                              columns=[f"pred_{c}" for c in class_names])
    cm_norm_df.to_csv(os.path.join(out_dir, "confusion_matrix_normalized.csv"), encoding="utf-8-sig")


    plt.figure(figsize=(8, 6), dpi=150)
    im = plt.imshow(cm, interpolation="nearest")
    plt.title("Confusion Matrix")
    plt.colorbar(im, fraction=0.046, pad=0.04)
    ticks = np.arange(num_classes)
    plt.xticks(ticks, class_names, rotation=45, ha="right")
    plt.yticks(ticks, class_names)
    plt.xlabel("Predicted label"); plt.ylabel("True label")
    thresh = cm.max() / 2.0 if cm.max() > 0 else 0.5
    for i in range(num_classes):
        for j in range(num_classes):
            plt.text(j, i, str(cm[i, j]),
                     ha="center", va="center",
                     color="white" if cm[i, j] > thresh else "black", fontsize=9)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "confusion_matrix.png"), bbox_inches="tight")
    plt.close()


    report = classification_report(
        all_targets, all_preds,
        labels=labels_all,
        target_names=class_names,
        digits=4,
        zero_division=0,
        output_dict=True,
    )

    report_text = classification_report(
        all_targets, all_preds,
        labels=labels_all,
        target_names=class_names,
        digits=4,
        zero_division=0,
        output_dict=False,
    )
    with open(os.path.join(out_dir, "classification_report.txt"), "w", encoding="utf-8") as f:
        f.write(report_text)


    report_df = pd.DataFrame(report).transpose()
    report_df.to_csv(os.path.join(out_dir, "classification_report.csv"), encoding="utf-8-sig")


    per_class_recall    = np.divide(np.diag(cm), np.clip(cm.sum(axis=1), 1, None))  # TP / (TP+FN)
    per_class_precision = np.divide(np.diag(cm), np.clip(cm.sum(axis=0), 1, None))  # TP / (TP+FP)
    pr_df = pd.DataFrame({
        "class": class_names,
        "precision": per_class_precision,
        "recall": per_class_recall,
    })
    pr_df.to_csv(os.path.join(out_dir, "per_class_precision_recall.csv"), index=False, encoding="utf-8-sig")

    print("\n===== Classification Report =====\n")
    print(report_text)
    print(f"\nSave in;{os.path.abspath(out_dir)}")

def _parse_classification_report_to_frame(report_text):
    lines = [l.strip() for l in report_text.strip().splitlines() if l.strip()]
    header_idx = None
    for i, l in enumerate(lines):
        if l.startswith("precision") or l.startswith("class") or "precision" in l:
            header_idx = i
            break
    if header_idx is None:
        return pd.DataFrame()

    data_lines = lines[header_idx+1:]
    rows = []
    for l in data_lines:
        parts = l.split()
        if len(parts) < 2:
            continue
        name = parts[0]
        if name == "accuracy":
            try:
                acc = float(parts[-2])
                support = int(parts[-1])
            except:
                acc, support = np.nan, np.nan
            rows.append([name, acc, np.nan, np.nan, support])
        else:
            try:
                precision = float(parts[-4])
                recall    = float(parts[-3])
                f1        = float(parts[-2])
                support   = int(parts[-1])
            except:
                precision = recall = f1 = np.nan
                try:
                    support = int(parts[-1])
                except:
                    support = np.nan
            if len(parts) > 5:
                name = " ".join(parts[:-4])
            rows.append([name, precision, recall, f1, support])

    return pd.DataFrame(rows, columns=["name", "precision", "recall", "f1", "support"])