#!/usr/bin/env python3
"""Purely hand-crafted OpenCV quality baseline for VizWiz images."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from statistics import mean, median

import cv2
import numpy as np

from brisque_quality import load_annotations
from sklearn.metrics import (accuracy_score, average_precision_score, balanced_accuracy_score,
                             brier_score_loss, confusion_matrix, f1_score, log_loss,
                             matthews_corrcoef, precision_score, recall_score, roc_auc_score)


FLAWS = ("FRM", "BLR", "BRT", "DRK", "OBS", "ROT", "OTH", "NON")
ISSUES = ("BLR", "BRT", "DRK", "OBS", "FRM", "ROT", "OTH")


def clamp(value: float) -> float:
    return float(min(1.0, max(0.0, value)))


def resize_for_analysis(image: np.ndarray, max_side: int = 640) -> np.ndarray:
    height, width = image.shape[:2]
    scale = min(1.0, max_side / max(height, width))
    if scale == 1.0:
        return image
    return cv2.resize(image, (round(width * scale), round(height * scale)),
                      interpolation=cv2.INTER_AREA)


def rotation_score(edges: np.ndarray) -> tuple[float, int]:
    min_length = max(20, min(edges.shape) // 6)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=50,
                            minLineLength=min_length, maxLineGap=12)
    if lines is None:
        return 0.0, 0
    deviations = []
    weights = []
    for x1, y1, x2, y2 in lines[:, 0]:
        dx, dy = float(x2 - x1), float(y2 - y1)
        angle = abs(math.degrees(math.atan2(dy, dx))) % 90
        deviation = min(angle, 90 - angle)
        length = math.hypot(dx, dy)
        deviations.append(deviation)
        weights.append(length)
    weighted_deviation = float(np.average(deviations, weights=weights))
    # Ignore normal perspective errors below 4 degrees; saturate at 25 degrees.
    return clamp((weighted_deviation - 4.0) / 21.0), len(deviations)


def obstruction_score(gray: np.ndarray) -> tuple[float, float]:
    # Find large connected regions made of low-variance 32x32 blocks.
    block = 32
    rows = max(1, math.ceil(gray.shape[0] / block))
    cols = max(1, math.ceil(gray.shape[1] / block))
    mask = np.zeros((rows, cols), dtype=np.uint8)
    for row in range(rows):
        for col in range(cols):
            tile = gray[row * block:(row + 1) * block, col * block:(col + 1) * block]
            if tile.size and float(tile.std()) < 8.0:
                mask[row, col] = 1
    low_texture_fraction = float(mask.mean())
    count, _, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    largest_fraction = 0.0 if count <= 1 else float(stats[1:, cv2.CC_STAT_AREA].max() / mask.size)
    score = clamp(0.45 * low_texture_fraction + 0.90 * largest_fraction)
    return score, low_texture_fraction


def extract_scores(image: np.ndarray) -> dict[str, float]:
    image = resize_for_analysis(image)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray_float = gray.astype(np.float32)
    height, width = gray.shape

    mean_luma = float(gray_float.mean())
    contrast = float(gray_float.std())
    dark_fraction = float(np.mean(gray <= 25))
    bright_fraction = float(np.mean(gray >= 230))

    laplacian_variance = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    tenengrad = float(np.mean(gx * gx + gy * gy))
    blur = clamp(0.65 * (1.0 - laplacian_variance / 500.0)
                 + 0.35 * (1.0 - tenengrad / 5000.0))

    edges = cv2.Canny(gray, 60, 160)
    band = max(2, round(min(height, width) * 0.08))
    border_mask = np.zeros_like(edges, dtype=bool)
    border_mask[:band] = border_mask[-band:] = True
    border_mask[:, :band] = border_mask[:, -band:] = True
    border_density = float(np.mean(edges[border_mask] > 0))
    inner_density = float(np.mean(edges[~border_mask] > 0)) if np.any(~border_mask) else 0.0
    framing = clamp((border_density / (inner_density + 0.01) - 0.8) / 2.5)

    dark = clamp(0.60 * dark_fraction + 0.40 * (1.0 - mean_luma / 115.0))
    bright = clamp(0.60 * bright_fraction + 0.40 * ((mean_luma - 140.0) / 115.0))
    obstruction, low_texture_fraction = obstruction_score(gray)
    rotation, line_count = rotation_score(edges)

    smooth = cv2.GaussianBlur(gray, (5, 5), 0)
    noise_residual = float(np.mean(np.abs(gray_float - smooth.astype(np.float32))))
    channel_means = image.reshape(-1, 3).mean(axis=0)
    color_cast = float((channel_means.max() - channel_means.min()) / 255.0)
    low_contrast = clamp((35.0 - contrast) / 35.0)
    other = clamp(0.35 * noise_residual / 18.0 + 0.35 * color_cast / 0.35
                  + 0.30 * low_contrast)

    issue_scores = [framing, blur, bright, dark, obstruction, rotation, other]
    # A soft maximum: one severe issue is enough to make an image poor.
    composite = clamp(0.65 * max(issue_scores) + 0.35 * mean(issue_scores))
    non = clamp(1.0 - composite)

    return {
        "score_FRM": framing, "score_BLR": blur, "score_BRT": bright,
        "score_DRK": dark, "score_OBS": obstruction, "score_ROT": rotation,
        "score_OTH": other, "score_NON": non, "score_composite": composite,
        "mean_luma": mean_luma, "contrast": contrast,
        "dark_pixel_fraction": dark_fraction, "bright_pixel_fraction": bright_fraction,
        "laplacian_variance": laplacian_variance, "tenengrad": tenengrad,
        "border_edge_density": border_density, "inner_edge_density": inner_density,
        "low_texture_fraction": low_texture_fraction, "noise_residual": noise_residual,
        "color_cast": color_cast, "detected_lines": line_count,
    }


def classification_metrics(labels: list[int], scores: list[float]) -> dict:
    """Match the Qwen evaluator: score > .5, plus ranking and calibration metrics."""
    y = np.asarray(labels, dtype=int)
    p = np.asarray(scores, dtype=float)
    predictions = (p > 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, predictions, labels=[0, 1]).ravel()
    bin_ids = np.minimum((p * 10).astype(int), 9)
    ece = sum(np.mean(bin_ids == bucket) * abs(p[bin_ids == bucket].mean() - y[bin_ids == bucket].mean())
              for bucket in range(10) if (bin_ids == bucket).any())
    return {
        "n": int(len(y)), "positives": int(y.sum()), "prevalence": float(y.mean()),
        "AP": float(average_precision_score(y, p)) if y.sum() else None,
        "AUROC": float(roc_auc_score(y, p)) if len(np.unique(y)) == 2 else None,
        "precision": float(precision_score(y, predictions, zero_division=0)),
        "recall": float(recall_score(y, predictions, zero_division=0)),
        "F1": float(f1_score(y, predictions, zero_division=0)),
        "accuracy": float(accuracy_score(y, predictions)),
        "balanced_accuracy": float(balanced_accuracy_score(y, predictions)),
        "MCC": float(matthews_corrcoef(y, predictions)),
        "Brier": float(brier_score_loss(y, p)),
        "log_loss": float(log_loss(y, np.clip(p, 1e-7, 1 - 1e-7), labels=[0, 1])),
        "ECE_10bins": float(ece), "threshold": 0.5, "positive_rule": "score > 0.5; ties -> 0",
        "TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp),
    }


def load_target_labels(path: Path) -> list[dict]:
    """Read the compact test.csv, or retain support for the original crowd CSV."""
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        columns = set(reader.fieldnames or [])
        if {"image_id", *ISSUES}.issubset(columns):
            rows = []
            for row in reader:
                labels = {issue: int(row[issue]) for issue in ISSUES}
                rows.append({"image": row["image_id"], "labels": labels,
                             "any_issue": int(any(labels.values()))})
            return rows
    annotations = load_annotations(path, "test")
    rows = []
    for ann in annotations:
        labels = {issue: int(ann["flaws"][issue] >= 2) for issue in ISSUES}
        rows.append({"image": ann["image"], "labels": labels,
                     "any_issue": int(any(labels.values()))})
    return rows


def top_k_coverage(rows: list[dict]) -> tuple[list[dict], dict]:
    """Per-image minimum k needed to retrieve every positive ground-truth issue."""
    per_image = []
    for row in rows:
        # Python's stable sort makes ties deterministic in the documented ISSUES order.
        ranked = sorted(ISSUES, key=lambda issue: -float(row[f"score_{issue}"]))
        positives = [issue for issue in ISSUES if int(row[f"label_{issue}"]) == 1]
        ranks = {issue: index + 1 for index, issue in enumerate(ranked)}
        minimum_k = max((ranks[issue] for issue in positives), default=0)
        per_image.append({"image": row["image"], "gt_positive_count": len(positives),
                          "gt_positive_labels": ",".join(positives),
                          "ranked_labels": ",".join(ranked), "k_min_recall_1": minimum_k})

    def summarize(group: list[dict]) -> dict:
        required = np.asarray([row["k_min_recall_1"] for row in group], dtype=float)
        return {"n": int(len(group)), "mean_k_min": float(required.mean()),
                "median_k_min": float(np.median(required)),
                "p25_k_min": float(np.percentile(required, 25)),
                "p75_k_min": float(np.percentile(required, 75)),
                "p90_k_min": float(np.percentile(required, 90)),
                "max_k_min": int(required.max()),
                # Sample-level Recall@k: all GT labels of an image must be retrieved.
                "recall_1_at_k": {str(k): float(np.mean(required <= k)) for k in range(1, len(ISSUES) + 1)}}

    positives = [row for row in per_image if row["gt_positive_count"] > 0]
    by_cardinality = {str(count): summarize([row for row in positives if row["gt_positive_count"] == count])
                      for count in range(1, len(ISSUES) + 1)
                      if any(row["gt_positive_count"] == count for row in positives)}
    return per_image, {"definition": "k_min_recall_1 is the maximum rank among an image's GT-positive labels; ties use ISSUES order.",
                       "labels_ranked": list(ISSUES), "positive_label_images": len(positives),
                       "zero_positive_label_images": len(per_image) - len(positives),
                       "overall_positive_label_images": summarize(positives),
                       "by_gt_positive_count": by_cardinality}


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images", type=Path, default=root / "test")
    parser.add_argument("--labels", type=Path,
                        default=root / "VizWiz_quality_issues_train_val_test.csv")
    parser.add_argument("--output", type=Path,
                        default=root / "vizwiz_non_ML_baseline/rule_based_test_scores.csv")
    parser.add_argument("--summary", type=Path,
                        default=root / "vizwiz_non_ML_baseline/rule_based_test_summary.json")
    parser.add_argument("--input-scores", type=Path,
                        help="reuse an existing rule score CSV and only recompute labels/metrics")
    parser.add_argument("--topk-output", type=Path,
                        help="CSV containing per-image minimum k for Recall@k=1")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--offset", type=int, default=0,
                        help="start at this zero-based label row (useful for resumable score batches)")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    annotations = load_target_labels(args.labels)
    if args.offset:
        annotations = annotations[args.offset:]
    if args.limit:
        annotations = annotations[:args.limit]
    rows = []
    failures = []
    if args.input_scores:
        with args.input_scores.open(newline="", encoding="utf-8") as handle:
            existing = {row["image"]: row for row in csv.DictReader(handle)}
        required = {f"score_{key}" for key in ISSUES} | {"score_composite"}
        for ann in annotations:
            row = existing.get(ann["image"])
            if row is None or not required.issubset(row):
                failures.append(ann["image"])
                continue
            rows.append({"image": ann["image"], **{key: float(value) if key.startswith("score_") else value
                                                   for key, value in row.items() if key != "image"},
                         **{f"label_{key}": ann["labels"][key] for key in ISSUES},
                         "label_any_issue": ann["any_issue"]})
    else:
        for index, ann in enumerate(annotations, 1):
            path = args.images / ann["image"]
            image = cv2.imread(str(path), cv2.IMREAD_COLOR)
            if image is None:
                failures.append(ann["image"])
                continue
            scores = extract_scores(image)
            rows.append({
                "image": ann["image"], **scores,
                **{f"label_{key}": ann["labels"][key] for key in ISSUES},
                "label_any_issue": ann["any_issue"],
            })
            if index % 250 == 0 or index == len(annotations):
                print(f"Processed {index}/{len(annotations)}", file=sys.stderr)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    evaluations = {}
    for flaw in ISSUES:
        evaluations[flaw] = classification_metrics(
            [int(row[f"label_{flaw}"]) for row in rows],
            [float(row[f"score_{flaw}"]) for row in rows],
        )
    composite_scores = [float(row["score_composite"]) for row in rows]
    evaluations["composite_vs_any_issue"] = classification_metrics(
        [int(row["label_any_issue"]) for row in rows], composite_scores)
    topk_rows, topk_summary = top_k_coverage(rows)
    topk_output = args.topk_output or args.summary.with_name(args.summary.stem + "_topk.csv")
    with topk_output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(topk_rows[0]))
        writer.writeheader()
        writer.writerows(topk_rows)

    summary = {
        "method": "pure hand-crafted OpenCV rules; no learned parameters",
        "label_rule": "labels supplied by --labels (compact test.csv uses its provided binary labels)",
        "metric_protocol": "Aligned with vizwiz_baseline/evaluate.py: fixed score > 0.5 and AP/AUROC/F1/Brier/log loss/ECE.",
        "scored_images": len(rows), "failed_images": failures,
        "evaluations": evaluations,
        "top_k_recall": topk_summary,
    }
    args.summary.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote scores to {args.output}")
    print(f"Wrote metrics to {args.summary}")
    print(f"Wrote top-k coverage to {topk_output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
