import os
import argparse
import warnings

import numpy as np
import pandas as pd
from PIL import Image

import torch
import pyiqa
import cpbd

from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve, precision_score, recall_score, f1_score, accuracy_score, balanced_accuracy_score, confusion_matrix


ISSUE_COLUMNS = ["BLR", "BRT", "DRK", "OBS", "FRM", "ROT", "OTH"]


def load_annotations(csv_path):
    df = pd.read_csv(csv_path)
    df.columns = [c.strip() for c in df.columns]

    required = ["IMG", "REJECT", "SPLIT", *ISSUE_COLUMNS]
    missing = [c for c in required if c not in df.columns]

    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df["SPLIT"] = df["SPLIT"].astype(str).str.strip().str.lower().replace({"validation": "val", "valid": "val"})
    df["IMG"] = df["IMG"].astype(str).apply(os.path.basename)

    vote_columns = ["REJECT", *ISSUE_COLUMNS]

    for col in vote_columns:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    aggregated = df.groupby(["IMG", "SPLIT"], as_index=False, sort=False)[vote_columns].sum()

    aggregated["unrecognizable"] = (aggregated["REJECT"] >= 2).astype(int)

    for col in ISSUE_COLUMNS:
        aggregated[f"{col}_label"] = (aggregated[col] >= 2).astype(int)

    issue_label_columns = [f"{c}_label" for c in ISSUE_COLUMNS]
    aggregated["any_issue"] = aggregated[issue_label_columns].max(axis=1).astype(int)

    return aggregated


def build_split_mapping(annotations, image_dir, split):
    split = split.lower()
    split_df = annotations[annotations["SPLIT"] == split].copy()
    local_files = set(f for f in os.listdir(image_dir) if f.lower().endswith(".jpg"))

    mapping = {}

    if split == "test":
        for _, row in split_df.iterrows():
            csv_name = row["IMG"]
            csv_index = int(csv_name.split("_")[-1].replace(".jpg", ""))
            local_index = csv_index - 20000
            local_name = f"VizWiz_test_{local_index:08d}.jpg"

            if local_name in local_files:
                mapping[csv_name] = local_name
            else:
                warnings.warn(f"Missing test image: {csv_name} -> {local_name}")

    elif split == "val":
        split_df = split_df[split_df["IMG"].str.startswith("VizWiz_val_")].copy()

        for _, row in split_df.iterrows():
            csv_name = row["IMG"]
            csv_index = int(csv_name.split("_")[-1].replace(".jpg", ""))
            local_index = csv_index - 28000
            local_name = f"VizWiz_val_{local_index:08d}.jpg"

            if local_name in local_files:
                mapping[csv_name] = local_name
            else:
                warnings.warn(f"Missing validation image: {csv_name} -> {local_name}")

    else:
        raise ValueError(f"Unsupported split: {split}")

    print(f"\n[{split}] successfully mapped {len(mapping)} images")

    return mapping


def create_metrics(device):
    print("\nInitializing NIQE and PIQE...")
    niqe = pyiqa.create_metric("niqe", device=device)
    piqe = pyiqa.create_metric("piqe", device=device)

    return niqe, piqe


def compute_cpbd_badness(image_path):
    image = Image.open(image_path).convert("L")
    image = np.asarray(image)
    cpbd_quality = float(cpbd.compute(image))

    return 1.0 - cpbd_quality


def compute_pyiqa_score(metric, image_path):
    with torch.no_grad():
        score = metric(image_path)

    return float(score.item())


def compute_scores(annotations, image_dir, split, mapping, niqe, piqe):
    split_df = annotations[annotations["SPLIT"] == split].copy()

    if split == "val":
        split_df = split_df[split_df["IMG"].str.startswith("VizWiz_val_")].copy()

    print("\n" + "=" * 70)
    print(f"Computing scores for {split}")
    print("=" * 70)
    print(f"Images to process: {len(split_df)}")

    rows = []
    missing_count = 0
    failed_count = 0

    for count, (_, row) in enumerate(split_df.iterrows(), start=1):
        csv_image_name = row["IMG"]

        if csv_image_name not in mapping:
            warnings.warn(f"No mapping found for {csv_image_name}")
            missing_count += 1
            continue

        local_image_name = mapping[csv_image_name]
        image_path = os.path.join(image_dir, local_image_name)

        if not os.path.isfile(image_path):
            warnings.warn(f"Image does not exist: {image_path}")
            missing_count += 1
            continue

        try:
            cpbd_score = compute_cpbd_badness(image_path)
            niqe_score = compute_pyiqa_score(niqe, image_path)
            piqe_score = compute_pyiqa_score(piqe, image_path)
        except Exception as e:
            warnings.warn(f"Metric computation failed for {local_image_name}: {e}")
            failed_count += 1
            continue

        rows.append({
            "CSV_IMG": csv_image_name,
            "IMG": local_image_name,
            "unrecognizable": int(row["unrecognizable"]),
            "any_issue": int(row["any_issue"]),
            "CPBD": cpbd_score,
            "NIQE": niqe_score,
            "PIQE": piqe_score,
        })

        if count % 100 == 0:
            print(f"Processed {count} / {len(split_df)}")

    result = pd.DataFrame(rows, columns=["CSV_IMG", "IMG", "unrecognizable", "any_issue", "CPBD", "NIQE", "PIQE"])

    print(f"\nSuccessfully processed: {len(result)} / {len(split_df)}")
    print(f"Missing mappings/files: {missing_count}")
    print(f"Metric failures: {failed_count}")

    if len(result) == 0:
        raise RuntimeError(f"No images successfully processed for split {split}")

    return result


def choose_threshold(y_true, scores):
    y_true = np.asarray(y_true)
    scores = np.asarray(scores)

    fpr, tpr, thresholds = roc_curve(y_true, scores)
    balanced_accuracy = (tpr + (1.0 - fpr)) / 2.0

    finite = np.isfinite(thresholds)
    thresholds = thresholds[finite]
    balanced_accuracy = balanced_accuracy[finite]

    best_index = int(np.argmax(balanced_accuracy))

    return float(thresholds[best_index]), float(balanced_accuracy[best_index])


def evaluate(y_true, scores, threshold):
    y_true = np.asarray(y_true)
    scores = np.asarray(scores)
    y_pred = (scores >= threshold).astype(int)

    ap = average_precision_score(y_true, scores)
    roc_auc = roc_auc_score(y_true, scores)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    accuracy = accuracy_score(y_true, y_pred)
    balanced_acc = balanced_accuracy_score(y_true, y_pred)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    return {
        "Pos": int(np.sum(y_true)),
        "Neg": int(len(y_true) - np.sum(y_true)),
        "Threshold": threshold,
        "AP": ap,
        "ROC-AUC": roc_auc,
        "Precision": precision,
        "Recall": recall,
        "F1": f1,
        "Accuracy": accuracy,
        "Balanced Acc.": balanced_acc,
        "TP": int(tp),
        "FP": int(fp),
        "TN": int(tn),
        "FN": int(fn),
    }


def run_experiment(val_scores, test_scores):
    metrics = ["CPBD", "NIQE", "PIQE"]
    targets = ["unrecognizable", "any_issue"]
    all_results = []

    for target in targets:
        print("\n" + "=" * 70)
        print(f"TARGET: {target}")
        print("=" * 70)

        y_val = val_scores[target].to_numpy()
        y_test = test_scores[target].to_numpy()

        print(f"Validation positives: {int(y_val.sum())} / {len(y_val)}")
        print(f"Test positives: {int(y_test.sum())} / {len(y_test)}")

        for metric_name in metrics:
            val_metric_scores = val_scores[metric_name].to_numpy()
            test_metric_scores = test_scores[metric_name].to_numpy()

            threshold, val_bal_acc = choose_threshold(y_val, val_metric_scores)
            result = evaluate(y_test, test_metric_scores, threshold)

            result = {
                "Method": metric_name,
                "Target": target,
                "Val Balanced Acc.": val_bal_acc,
                **result,
            }

            all_results.append(result)

            print(f"\n{metric_name}")
            print("-" * 50)

            for key, value in result.items():
                if isinstance(value, float):
                    print(f"{key:20s}: {value:.4f}")
                else:
                    print(f"{key:20s}: {value}")

    return pd.DataFrame(all_results)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--val_dir", required=True, help="Path to VizWiz validation images")
    parser.add_argument("--test_dir", required=True, help="Path to VizWiz test images")
    parser.add_argument("--csv", required=True, help="Path to VizWiz_quality_issues_train_val_test.csv")
    parser.add_argument("--output", default="nr_iqa_results.csv", help="Output CSV containing final results")

    args = parser.parse_args()

    print("Loading annotations...")
    annotations = load_annotations(args.csv)

    print("\nUnique image counts after worker aggregation:")
    print(annotations.groupby("SPLIT").size())

    val_mapping = build_split_mapping(annotations, args.val_dir, "val")
    test_mapping = build_split_mapping(annotations, args.test_dir, "test")

    device = torch.device("cpu")
    niqe, piqe = create_metrics(device)

    val_scores = compute_scores(annotations, args.val_dir, "val", val_mapping, niqe, piqe)
    test_scores = compute_scores(annotations, args.test_dir, "test", test_mapping, niqe, piqe)

    val_scores.to_csv("nr_iqa_val_scores.csv", index=False)
    test_scores.to_csv("nr_iqa_test_scores.csv", index=False)

    results = run_experiment(val_scores, test_scores)

    print("\n" + "=" * 100)
    print("FINAL RESULTS")
    print("=" * 100)

    print(results.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    results.to_csv(args.output, index=False)

    print(f"\nSaved final results to: {args.output}")
    print("Saved raw validation scores to: nr_iqa_val_scores.csv")
    print("Saved raw test scores to: nr_iqa_test_scores.csv")


if __name__ == "__main__":
    main()