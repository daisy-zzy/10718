#!/usr/bin/env python3
"""Score VizWiz images with OpenCV's no-reference BRISQUE metric."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import shutil
import ssl
import sys
import urllib.request
from collections import defaultdict
from pathlib import Path
from statistics import mean, median


MODEL_URL = (
    "https://raw.githubusercontent.com/opencv/opencv_contrib/4.x/"
    "modules/quality/samples/brisque_model_live.yml"
)
RANGE_URL = (
    "https://raw.githubusercontent.com/opencv/opencv_contrib/4.x/"
    "modules/quality/samples/brisque_range_live.yml"
)
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
DEFAULT_THRESHOLD = 13.576980590820312


def parse_args() -> argparse.Namespace:
    here = Path(__file__).resolve().parent
    root = here.parent
    parser = argparse.ArgumentParser(
        description=(
            "Compute OpenCV BRISQUE scores (lower is generally better). If a labeled "
            "VizWiz train/val annotation file is supplied, also evaluate the ranking."
        )
    )
    parser.add_argument("--images", type=Path, default=root / "test")
    parser.add_argument(
        "--annotations",
        type=Path,
        default=(root / "VizWiz_quality_issues_train_val_test.csv"
                 if (root / "VizWiz_quality_issues_train_val_test.csv").is_file()
                 else root / "VizWiz-QualityIssues/data/quality.json"),
        help="crowd-label CSV, compiled quality.json, or raw quality annotation JSON",
    )
    parser.add_argument("--split", choices=("train", "val", "test"), default="test",
                        help="split to read when --annotations is the compiled quality.json")
    parser.add_argument("--output", type=Path, default=here / "brisque_test_scores.csv")
    parser.add_argument("--summary", type=Path, default=here / "brisque_test_summary.json")
    parser.add_argument("--model", type=Path, default=here / "models/brisque_model_live.yml")
    parser.add_argument("--range", dest="range_file", type=Path,
                        default=here / "models/brisque_range_live.yml")
    parser.add_argument(
        "--download-models", action="store_true",
        help="download the two official OpenCV BRISQUE model files when missing",
    )
    parser.add_argument("--limit", type=int, help="only process the first N images")
    parser.add_argument(
        "--threshold", type=float, default=DEFAULT_THRESHOLD,
        help=(
            "BRISQUE cutoff for thresholded metrics: predict unrecognizable when "
            "score >= threshold (default: %(default)s)"
        ),
    )
    return parser.parse_args()


def ensure_model_file(path: Path, url: str, allow_download: bool) -> None:
    if path.is_file():
        return
    if not allow_download:
        raise FileNotFoundError(
            f"Missing BRISQUE model file: {path}\n"
            "Run again with --download-models, or provide it explicitly with --model/--range."
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {url} -> {path}", file=sys.stderr)
    try:
        import certifi
    except ImportError as exc:
        raise RuntimeError(
            "The 'certifi' package is required for verified HTTPS downloads; "
            "run: pip install -r requirements.txt"
        ) from exc
    ssl_context = ssl.create_default_context(cafile=certifi.where())
    temporary_path = path.with_suffix(path.suffix + ".part")
    try:
        with urllib.request.urlopen(url, context=ssl_context) as response:
            with temporary_path.open("wb") as output:
                shutil.copyfileobj(response, output)
        temporary_path.replace(path)
    finally:
        temporary_path.unlink(missing_ok=True)


def load_annotations(path: Path, split: str) -> list[dict]:
    if path.suffix.lower() == ".csv":
        grouped: dict[str, list[dict]] = defaultdict(list)
        with path.open(newline="", encoding="utf-8-sig") as handle:
            reader = csv.DictReader(handle)
            required = {"IMG", "SPLIT", "REJECT", "FRM", "BLR", "DRK", "BRT",
                        "OBS", "OTH", "NON", "ROT"}
            if not reader.fieldnames or not required.issubset(reader.fieldnames):
                raise ValueError(f"Crowd CSV is missing required columns: {path}")
            for row in reader:
                if row["SPLIT"].lower() == split.lower():
                    grouped[row["IMG"]].append(row)
        if split != "test":
            raise ValueError(
                "The crowd CSV filename conversion is currently defined only for TEST; "
                "use data/quality.json for train/val."
            )
        data = []
        for source_name, votes in grouped.items():
            # The supplied TEST CSV appends the zero-padded test index to a fixed
            # source prefix, e.g. ...000000020042.jpg -> ...00000042.jpg.
            match = re.search(r"(\d{4})\.[^.]+$", source_name)
            if not match:
                raise ValueError(f"Cannot recover the TEST index from {source_name!r}")
            image = f"VizWiz_test_{int(match.group(1)):08d}.jpg"
            reject_votes = sum(int(row["REJECT"]) for row in votes)
            flaw_votes = {
                key: sum(int(row[key]) for row in votes)
                for key in ("FRM", "BLR", "DRK", "BRT", "OBS", "OTH", "NON", "ROT")
            }
            data.append({
                "image": image,
                "unrecognizable": reject_votes,
                # Match the official VizWiz notebook's THRESHOLD = 2.
                "unrecognizable_label": int(reject_votes >= 2),
                "flaws": flaw_votes,
            })
        return sorted(data, key=lambda item: item["image"])

    with path.open(encoding="utf-8") as handle:
        data = json.load(handle)
    if isinstance(data, dict):
        if split not in data:
            raise ValueError(f"Split {split!r} is not present in {path}")
        packed = data[split]
        names = packed.get("image", [])
        recognizable = packed.get("recognizable", [])
        flaws = packed.get("flaws", [])
        data = []
        for index, name in enumerate(names):
            item = {"image": name}
            if index < len(recognizable):
                value = recognizable[index]
                if isinstance(value, list):
                    value = value[0]
                item["unrecognizable_label"] = int(float(value) < 0.5)
            if index < len(flaws):
                item["compiled_flaws"] = flaws[index]
            data.append(item)
    if not isinstance(data, list) or any(not isinstance(x, dict) for x in data):
        raise ValueError(f"Unsupported annotation structure in {path}")
    return data


def image_records(images_dir: Path, annotations: list[dict]) -> list[tuple[Path, dict]]:
    if annotations:
        records = []
        for ann in annotations:
            name = ann.get("image")
            if not isinstance(name, str):
                raise ValueError("Every annotation must contain a string 'image' field")
            path = images_dir / name
            if path.is_file():
                records.append((path, ann))
        return records
    return [(p, {}) for p in sorted(images_dir.iterdir()) if p.suffix.lower() in IMAGE_SUFFIXES]


def average_precision(labels: list[int], scores: list[float]) -> float | None:
    positives = sum(labels)
    if positives == 0:
        return None
    ranked = sorted(zip(scores, labels), reverse=True)
    found = 0
    precision_sum = 0.0
    for rank, (_, label) in enumerate(ranked, 1):
        if label:
            found += 1
            precision_sum += found / rank
    return precision_sum / positives


def roc_auc(labels: list[int], scores: list[float]) -> float | None:
    positive_count = sum(labels)
    negative_count = len(labels) - positive_count
    if not positive_count or not negative_count:
        return None
    ordered = sorted(zip(scores, labels))
    wins = 0.0
    negatives_below = 0
    index = 0
    while index < len(ordered):
        end = index
        while end < len(ordered) and ordered[end][0] == ordered[index][0]:
            end += 1
        group = ordered[index:end]
        group_positives = sum(label for _, label in group)
        group_negatives = len(group) - group_positives
        wins += group_positives * (negatives_below + 0.5 * group_negatives)
        negatives_below += group_negatives
        index = end
    return wins / (positive_count * negative_count)


def best_accuracy(labels: list[int], scores: list[float]) -> tuple[float, float] | None:
    """Return optimistic in-sample accuracy and threshold for score >= threshold."""
    if not labels:
        return None
    ordered = sorted(zip(scores, labels), reverse=True)
    correct = len(labels) - sum(labels)  # threshold > max: predict everything negative
    best_correct = correct
    best_threshold = math.nextafter(ordered[0][0], math.inf)
    index = 0
    while index < len(ordered):
        end = index
        while end < len(ordered) and ordered[end][0] == ordered[index][0]:
            end += 1
        group_labels = [label for _, label in ordered[index:end]]
        correct += sum(group_labels) - (len(group_labels) - sum(group_labels))
        if correct > best_correct:
            best_correct = correct
            best_threshold = ordered[index][0]
        index = end
    return best_correct / len(labels), best_threshold


def classification_metrics(
    labels: list[int], scores: list[float], threshold: float
) -> dict:
    """Return classification metrics at one explicit score threshold."""
    predictions = [int(score >= threshold) for score in scores]
    tp = sum(y == p == 1 for y, p in zip(labels, predictions))
    tn = sum(y == p == 0 for y, p in zip(labels, predictions))
    fp = sum(y == 0 and p == 1 for y, p in zip(labels, predictions))
    fn = sum(y == 1 and p == 0 for y, p in zip(labels, predictions))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    return {
        "threshold": threshold,
        "precision": precision,
        "recall": recall,
        "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
        "accuracy": (tp + tn) / len(labels),
        "confusion_matrix": {"tp": tp, "fp": fp, "tn": tn, "fn": fn},
    }


def threshold_metrics(labels: list[int], scores: list[float]) -> dict | None:
    """Choose the threshold that maximizes balanced accuracy and evaluate it."""
    positives = sum(labels)
    negatives = len(labels) - positives
    if not positives or not negatives:
        return None
    ordered = sorted(zip(scores, labels), reverse=True)
    tp, tn = 0, negatives
    best_balanced = 0.5
    best_threshold = math.nextafter(ordered[0][0], math.inf)
    best_counts = (0, 0, negatives, positives)
    index = 0
    while index < len(ordered):
        end = index
        while end < len(ordered) and ordered[end][0] == ordered[index][0]:
            end += 1
        group = [label for _, label in ordered[index:end]]
        tp += sum(group)
        tn -= len(group) - sum(group)
        fp, fn = negatives - tn, positives - tp
        balanced = 0.5 * (tp / positives + tn / negatives)
        if balanced > best_balanced:
            best_balanced = balanced
            best_threshold = ordered[index][0]
            best_counts = (tp, fp, tn, fn)
        index = end
    tp, fp, tn, fn = best_counts
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    return {
        "threshold": best_threshold,
        "threshold_selection": "maximized balanced accuracy on this same split",
        "precision": precision,
        "recall": recall,
        "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
        "accuracy": (tp + tn) / len(labels),
        "balanced_accuracy": best_balanced,
        "confusion_matrix": {"tp": tp, "fp": fp, "tn": tn, "fn": fn},
    }


def pearson(xs: list[float], ys: list[float]) -> float | None:
    if len(xs) < 2:
        return None
    mx, my = mean(xs), mean(ys)
    numerator = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    denominator = math.sqrt(sum((x - mx) ** 2 for x in xs) * sum((y - my) ** 2 for y in ys))
    return numerator / denominator if denominator else None


def percentile(sorted_values: list[float], fraction: float) -> float:
    if len(sorted_values) == 1:
        return sorted_values[0]
    position = fraction * (len(sorted_values) - 1)
    low = int(position)
    high = min(low + 1, len(sorted_values) - 1)
    weight = position - low
    return sorted_values[low] * (1 - weight) + sorted_values[high] * weight


def main() -> int:
    args = parse_args()
    if not args.images.is_dir():
        raise FileNotFoundError(f"Image directory does not exist: {args.images}")
    if not args.annotations.is_file():
        raise FileNotFoundError(f"Annotation file does not exist: {args.annotations}")
    if args.limit is not None and args.limit < 1:
        raise ValueError("--limit must be a positive integer")

    ensure_model_file(args.model, MODEL_URL, args.download_models)
    ensure_model_file(args.range_file, RANGE_URL, args.download_models)

    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError("OpenCV is missing; run: pip install -r requirements.txt") from exc
    if not hasattr(cv2, "quality") or not hasattr(cv2.quality, "QualityBRISQUE_create"):
        raise RuntimeError("This OpenCV build lacks cv2.quality; install opencv-contrib-python")

    annotations = load_annotations(args.annotations, args.split)
    records = image_records(args.images, annotations)
    missing_count = len(annotations) - len(records)
    if args.limit is not None:
        records = records[: args.limit]
    if not records:
        raise RuntimeError("No matching images were found")

    scorer = cv2.quality.QualityBRISQUE_create(str(args.model), str(args.range_file))
    rows = []
    failed = []
    total = len(records)
    for index, (path, ann) in enumerate(records, 1):
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            failed.append(path.name)
            continue
        raw_score = scorer.compute(image)
        score = float(raw_score[0] if isinstance(raw_score, (tuple, list)) else raw_score)
        votes = ann.get("unrecognizable")
        binary_label = ann.get("unrecognizable_label")
        if binary_label is None and isinstance(votes, (int, float)):
            binary_label = int(votes >= 2)
        flaws = ann.get("flaws") or {}
        rows.append({
            "image": path.name,
            "brisque_score": score,
            "unrecognizable_votes": votes if votes is not None else "",
            "unrecognizable_label": binary_label if binary_label is not None else "",
            **{f"flaw_{key}": flaws.get(key, "") for key in
               ("FRM", "BLR", "DRK", "BRT", "OBS", "OTH", "NON", "ROT")},
        })
        if index % 250 == 0 or index == total:
            print(f"Processed {index}/{total}", file=sys.stderr)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0])
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    scores = [row["brisque_score"] for row in rows]
    ordered = sorted(scores)
    summary = {
        "images_directory": str(args.images.resolve()),
        "annotations": str(args.annotations.resolve()),
        "output_csv": str(args.output.resolve()),
        "scored_images": len(rows),
        "missing_annotated_images": missing_count,
        "failed_images": failed,
        "score_interpretation": "lower BRISQUE usually means better perceptual quality",
        "brisque": {
            "mean": mean(scores), "median": median(scores), "min": ordered[0],
            "q1": percentile(ordered, 0.25), "q3": percentile(ordered, 0.75),
            "max": ordered[-1],
        },
    }
    labeled = [row for row in rows if row["unrecognizable_label"] != ""]
    if labeled:
        labels = [int(row["unrecognizable_label"]) for row in labeled]
        labeled_scores = [float(row["brisque_score"]) for row in labeled]
        optimum = best_accuracy(labels, labeled_scores)
        fixed_threshold_metrics = classification_metrics(labels, labeled_scores, args.threshold)
        summary["label_evaluation"] = {
            "positive_class": "unrecognizable",
            "positive_images": sum(labels),
            "average_precision": average_precision(labels, labeled_scores),
            "roc_auc": roc_auc(labels, labeled_scores),
            "threshold_metrics": fixed_threshold_metrics,
            "best_in_sample_accuracy": optimum[0] if optimum else None,
            "best_in_sample_threshold": optimum[1] if optimum else None,
            "accuracy_rule": "predict unrecognizable when BRISQUE >= threshold",
            "metric_note": "AP is threshold-free; recall and F1 use --threshold.",
            "balanced_threshold_metrics": threshold_metrics(labels, labeled_scores),
        }
        vote_rows = [row for row in labeled if row["unrecognizable_votes"] != ""]
        if vote_rows:
            summary["label_evaluation"]["pearson_score_vs_votes"] = pearson(
                [float(row["brisque_score"]) for row in vote_rows],
                [float(row["unrecognizable_votes"]) for row in vote_rows],
            )
    else:
        summary["label_evaluation"] = None
        summary["label_note"] = "The VizWiz test annotations do not publish quality labels."

    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Wrote {len(rows)} scores to {args.output}")
    print(f"Wrote summary to {args.summary}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(2)
