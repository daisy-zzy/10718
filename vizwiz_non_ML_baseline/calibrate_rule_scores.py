#!/usr/bin/env python3
"""Fit per-label Platt scaling on validation rule scores and apply it to test scores."""
import argparse, csv, json, math
from pathlib import Path
import numpy as np

ISSUES = ("BLR", "BRT", "DRK", "OBS", "FRM", "ROT", "OTH")
SIX_NAMED_DEFECTS = ("BLR", "BRT", "DRK", "OBS", "FRM", "ROT")
RECALLS = (1.0, 0.9, 0.8, 0.7)


def load_labels(path):
    with Path(path).open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    required = {"image_id", *ISSUES}
    if not rows or not required.issubset(rows[0]):
        raise ValueError(f"Expected compact label CSV with columns {sorted(required)}")
    return {r["image_id"]: {k: int(r[k]) for k in ISSUES} for r in rows}


def load_scores(path):
    with Path(path).open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    required = {"image", *(f"score_{k}" for k in ISSUES)}
    if not rows or not required.issubset(rows[0]):
        raise ValueError(f"Missing score columns in {path}")
    return {r["image"]: {k: float(r[f"score_{k}"]) for k in ISSUES} for r in rows}


def logit(p, eps=1e-6):
    p = min(max(p, eps), 1 - eps)
    return math.log(p / (1 - p))


def sigmoid(z):
    return 1 / (1 + math.exp(-z)) if z >= 0 else math.exp(z) / (1 + math.exp(z))


def fit_platt(y, scores, iters=100):
    """Same monotone, maximum-likelihood Platt fit as vizwiz_llm_baseline."""
    x = [logit(score) for score in scores]
    a, b = 1.0, 0.0
    def nll(slope, intercept):
        return -sum(label * math.log(max(sigmoid(slope * value + intercept), 1e-300))
                    + (1 - label) * math.log(max(1 - sigmoid(slope * value + intercept), 1e-300))
                    for value, label in zip(x, y))
    current = nll(a, b)
    for _ in range(iters):
        ga = gb = haa = hab = hbb = 0.0
        for value, label in zip(x, y):
            probability = sigmoid(a * value + b)
            weight = probability * (1 - probability)
            ga += (probability - label) * value; gb += probability - label
            haa += weight * value * value; hab += weight * value; hbb += weight
        determinant = haa * hbb - hab * hab
        if determinant <= 1e-16:
            break
        da = (hbb * ga - hab * gb) / determinant
        db = (haa * gb - hab * ga) / determinant
        step = 1.0
        while step > 1e-8 and nll(a - step * da, b - step * db) > current:
            step /= 2
        a, b = a - step * da, b - step * db
        updated = nll(a, b)
        if current - updated < 1e-10:
            break
        current = updated
    if a < 0:  # preserve the heuristic's within-label ordering; fall back to validation prior.
        prior = sum(y) / len(y)
        a, b = 0.0, logit(prior)
    return a, b


def ece(y, probabilities, bins=10):
    return sum(np.mean(ids == bucket) * abs(np.mean(np.asarray(probabilities)[ids == bucket]) - np.mean(np.asarray(y)[ids == bucket]))
               for ids in [np.minimum((np.asarray(probabilities) * bins).astype(int), bins - 1)]
               for bucket in range(bins) if np.any(ids == bucket))


def pr_points(y, scores):
    """Precision/recall after every distinct score threshold, high scores first."""
    order = sorted(range(len(scores)), key=lambda index: -scores[index])
    positives, true_positive = sum(y), 0
    points = []
    for index, row_index in enumerate(order):
        true_positive += y[row_index]
        if index + 1 == len(order) or scores[order[index + 1]] != scores[row_index]:
            points.append((true_positive / (index + 1), true_positive / positives))
    return points


def average_precision(y, scores):
    result, previous_recall = 0.0, 0.0
    for precision, recall in pr_points(y, scores):
        result += (recall - previous_recall) * precision
        previous_recall = recall
    return result


def precision_at_recall(y, scores, target):
    return max(precision for precision, recall in pr_points(y, scores) if recall >= target - 1e-12)


def ranks_desc(values):
    """One-based descending ranks; ties receive their average rank."""
    order = sorted(range(len(values)), key=lambda index: -values[index])
    ranks, start = [0.0] * len(values), 0
    while start < len(order):
        end = start
        while end + 1 < len(order) and values[order[end + 1]] == values[order[start]]:
            end += 1
        for index in order[start:end + 1]:
            ranks[index] = (start + end) / 2 + 1
        start = end + 1
    return ranks


def ranking_metrics(truth_rows, score_rows):
    """LLM-baseline ranking metrics across labels within every image."""
    mean_rank = coverage = top1_hit = 0.0
    count = 0
    for truth, scores in zip(truth_rows, score_rows):
        positives = [index for index, value in enumerate(truth) if value]
        if not positives:
            continue
        ranks = ranks_desc(scores)
        best_rank = min(ranks)
        tied_top = [index for index, value in enumerate(ranks) if value == best_rank]
        mean_rank += sum(ranks[index] for index in positives) / len(positives)
        coverage += max(ranks[index] for index in positives)
        top1_hit += sum(truth[index] for index in tied_top) / len(tied_top)
        count += 1
    return {"images": count, "mean_rank": mean_rank / count, "coverage": coverage / count,
            "top1_hit": top1_hit / count}


def random_ranking(truth_rows):
    label_count = len(ISSUES)
    cardinalities = [sum(row) for row in truth_rows if sum(row)]
    return {"images": len(cardinalities), "mean_rank": (label_count + 1) / 2,
            "coverage": sum(cardinality * (label_count + 1) / (cardinality + 1)
                            for cardinality in cardinalities) / len(cardinalities),
            "top1_hit": sum(cardinality / label_count for cardinality in cardinalities) / len(cardinalities)}


def oracle_ranking(truth_rows):
    cardinalities = [sum(row) for row in truth_rows if sum(row)]
    return {"images": len(cardinalities),
            "mean_rank": sum((cardinality + 1) / 2 for cardinality in cardinalities) / len(cardinalities),
            "coverage": sum(cardinalities) / len(cardinalities), "top1_hit": 1.0}


def topk_rows(labels, calibrated):
    rows = []
    for image, truth in labels.items():
        ranked = sorted(ISSUES, key=lambda key: -calibrated[image][key])
        positives = [key for key in ISSUES if truth[key]]
        rank = {key: index + 1 for index, key in enumerate(ranked)}
        rows.append({"image": image, "gt_positive_count": len(positives),
                     "gt_positive_labels": ",".join(positives), "ranked_calibrated_labels": ",".join(ranked),
                     "k_min_recall_1_calibrated": max((rank[key] for key in positives), default=0)})
    return rows


def summarize_topk(rows):
    positive = [r for r in rows if r["gt_positive_count"]]
    def one(group):
        values = np.asarray([r["k_min_recall_1_calibrated"] for r in group])
        return {"n": int(len(values)), "mean_k_min": float(values.mean()), "median_k_min": float(np.median(values)),
                "p25_k_min": float(np.percentile(values, 25)), "p75_k_min": float(np.percentile(values, 75)),
                "p90_k_min": float(np.percentile(values, 90)), "max_k_min": int(values.max()),
                "recall_1_at_k": {str(k): float(np.mean(values <= k)) for k in range(1, len(ISSUES) + 1)}}
    return {"definition": "minimum top-k that includes every GT-positive issue; ties use ISSUES order",
            "positive_label_images": len(positive), "zero_positive_label_images": len(rows) - len(positive),
            "overall_positive_label_images": one(positive),
            "by_gt_positive_count": {str(k): one([r for r in positive if r["gt_positive_count"] == k])
                                     for k in range(1, len(ISSUES) + 1) if any(r["gt_positive_count"] == k for r in positive)}}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--val-labels", required=True, type=Path)
    parser.add_argument("--val-scores", required=True, type=Path)
    parser.add_argument("--test-labels", required=True, type=Path)
    parser.add_argument("--test-scores", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args(); args.output_dir.mkdir(parents=True, exist_ok=True)
    val_labels, val_scores = load_labels(args.val_labels), load_scores(args.val_scores)
    test_labels, test_scores = load_labels(args.test_labels), load_scores(args.test_scores)
    val_names, test_names = sorted(set(val_labels) & set(val_scores)), sorted(set(test_labels) & set(test_scores))
    if len(val_names) != len(val_labels) or len(test_names) != len(test_labels):
        raise ValueError("Labels are missing scores")
    params, calibrated, diagnostics = {}, {}, {}
    for issue in ISSUES:
        y = [val_labels[name][issue] for name in val_names]
        raw = [val_scores[name][issue] for name in val_names]
        a, b = fit_platt(y, raw); params[issue] = {"a": a, "b": b}
        val_cal = [sigmoid(a * logit(score) + b) for score in raw]
        diagnostics[issue] = {"validation_positives": int(sum(y)), "ECE_raw": float(ece(y, raw)),
                              "ECE_calibrated": float(ece(y, val_cal))}
    for name in test_names:
        calibrated[name] = {issue: sigmoid(params[issue]["a"] * logit(test_scores[name][issue]) + params[issue]["b"])
                            for issue in ISSUES}
    test_truth_rows = [[test_labels[name][issue] for issue in ISSUES] for name in test_names]
    raw_score_rows = [[test_scores[name][issue] for issue in ISSUES] for name in test_names]
    calibrated_score_rows = [[calibrated[name][issue] for issue in ISSUES] for name in test_names]
    per_label, metric_rows = {}, []
    for issue in ISSUES:
        truth = [test_labels[name][issue] for name in test_names]
        scores = [test_scores[name][issue] for name in test_names]
        metrics = {"AP": average_precision(truth, scores),
                   **{f"P@R{recall:.1f}": precision_at_recall(truth, scores, recall) for recall in RECALLS}}
        per_label[issue] = metrics
        metric_rows.append({"label": issue, "positives": int(sum(truth)), "prevalence": float(np.mean(truth)),
                            "random_ranking_precision": float(np.mean(truth)), **metrics})
    output_scores = args.output_dir / "rule_based_test_scores_platt.csv"
    with output_scores.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["image", *(f"score_{k}_platt" for k in ISSUES)])
        writer.writeheader(); writer.writerows({"image": name, **{f"score_{k}_platt": calibrated[name][k] for k in ISSUES}} for name in test_names)
    with (args.output_dir / "per_label_precision_at_recall.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(metric_rows[0])); writer.writeheader(); writer.writerows(metric_rows)
    rows = topk_rows({name: test_labels[name] for name in test_names}, calibrated)
    topk_path = args.output_dir / "rule_based_test_topk_platt.csv"
    with topk_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    validation_prior_order = [np.mean([val_labels[name][issue] for name in val_names]) for issue in ISSUES]
    summary = {"recall_targets": list(RECALLS), "ranked_labels": list(ISSUES),
               "calibration": {"method": "per-label monotone Platt scaling: sigmoid(a * logit(raw_score) + b)",
                                "fit_split": "validation", "parameters": params, "validation_diagnostics": diagnostics},
               "evaluation_split": "test", "evaluation_images": len(test_names), "per_label": per_label,
               "six_named_defect_macro": {metric: float(np.mean([per_label[issue][metric] for issue in SIX_NAMED_DEFECTS]))
                                           for metric in per_label[ISSUES[0]]},
               "ranking_reference": {"oracle": oracle_ranking(test_truth_rows),
                                     "random": random_ranking(test_truth_rows),
                                     "prior_only": ranking_metrics(test_truth_rows, [validation_prior_order] * len(test_names))},
               "ranking_raw": ranking_metrics(test_truth_rows, raw_score_rows),
               "ranking_calibrated": ranking_metrics(test_truth_rows, calibrated_score_rows),
               "top_k_recall": summarize_topk(rows)}
    (args.output_dir / "rule_based_platt_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
