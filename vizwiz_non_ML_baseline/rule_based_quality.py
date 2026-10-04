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

from brisque_quality import average_precision, load_annotations, roc_auc


FLAWS = ("FRM", "BLR", "BRT", "DRK", "OBS", "ROT", "OTH", "NON")


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


def best_balanced_threshold(labels: list[int], scores: list[float]) -> tuple[float, float]:
    positives, negatives = sum(labels), len(labels) - sum(labels)
    if not positives or not negatives:
        return math.nan, math.nan
    ordered = sorted(zip(scores, labels), reverse=True)
    tp, tn = 0, negatives
    best = (0.5, math.nextafter(ordered[0][0], math.inf))
    index = 0
    while index < len(ordered):
        end = index
        while end < len(ordered) and ordered[end][0] == ordered[index][0]:
            end += 1
        group = [label for _, label in ordered[index:end]]
        tp += sum(group)
        tn -= len(group) - sum(group)
        balanced = 0.5 * (tp / positives + tn / negatives)
        if balanced > best[0]:
            best = (balanced, ordered[index][0])
        index = end
    return best


def classification_metrics(labels: list[int], scores: list[float]) -> dict:
    balanced, threshold = best_balanced_threshold(labels, scores)
    predictions = [int(score >= threshold) for score in scores]
    tp = sum(y == p == 1 for y, p in zip(labels, predictions))
    tn = sum(y == p == 0 for y, p in zip(labels, predictions))
    fp = sum(y == 0 and p == 1 for y, p in zip(labels, predictions))
    fn = sum(y == 1 and p == 0 for y, p in zip(labels, predictions))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    return {
        "positives": sum(labels),
        "average_precision": average_precision(labels, scores),
        "roc_auc": roc_auc(labels, scores),
        "balanced_accuracy": balanced,
        "threshold": threshold,
        "accuracy": (tp + tn) / len(labels),
        "precision": precision,
        "recall": recall,
        "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
        "confusion_matrix": {"tp": tp, "fp": fp, "tn": tn, "fn": fn},
    }


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
    parser.add_argument("--limit", type=int)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    annotations = load_annotations(args.labels, "test")
    if args.limit:
        annotations = annotations[:args.limit]
    rows = []
    failures = []
    for index, ann in enumerate(annotations, 1):
        path = args.images / ann["image"]
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            failures.append(ann["image"])
            continue
        scores = extract_scores(image)
        flaws = ann["flaws"]
        rows.append({
            "image": ann["image"], **scores,
            **{f"label_{key}": int(flaws[key] >= 2) for key in FLAWS},
            "reject_votes": ann["unrecognizable"],
            "label_unrecognizable": ann["unrecognizable_label"],
            "label_any_issue": int(any(flaws[key] >= 2 for key in FLAWS if key != "NON")),
        })
        if index % 250 == 0 or index == len(annotations):
            print(f"Processed {index}/{len(annotations)}", file=sys.stderr)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    evaluations = {}
    for flaw in FLAWS:
        evaluations[flaw] = classification_metrics(
            [int(row[f"label_{flaw}"]) for row in rows],
            [float(row[f"score_{flaw}"]) for row in rows],
        )
    composite_scores = [float(row["score_composite"]) for row in rows]
    evaluations["composite_vs_any_issue"] = classification_metrics(
        [int(row["label_any_issue"]) for row in rows], composite_scores)
    evaluations["composite_vs_unrecognizable"] = classification_metrics(
        [int(row["label_unrecognizable"]) for row in rows], composite_scores)

    summary = {
        "method": "pure hand-crafted OpenCV rules; no learned parameters",
        "label_rule": "at least 2 of 5 crowd votes (official VizWiz threshold)",
        "threshold_note": "Thresholds maximize balanced accuracy on this same test set and are descriptive, not held-out estimates.",
        "scored_images": len(rows), "failed_images": failures,
        "evaluations": evaluations,
    }
    args.summary.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote scores to {args.output}")
    print(f"Wrote metrics to {args.summary}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
