"""Fixed-recall operating points and uncapped, tie-aware seven-label rankings."""
import numpy as np
from scipy.special import expit
from scipy.stats import rankdata
from sklearn.metrics import average_precision_score

from . import LABELS

NAMED_LABELS = ("BLR", "BRT", "DRK", "OBS", "FRM", "ROT")
RECALL_TARGETS = (1.0, 0.9, 0.8, 0.7)
TABLE_COLUMNS = (*NAMED_LABELS, "Macro", "OTH", "NON", "UNREC")


def _arrays(y, scores):
    y, scores = np.asarray(y), np.asarray(scores, dtype=np.float64)
    if (y.shape != scores.shape or scores.ndim != 2 or scores.shape[1] != len(LABELS)
            or not np.isfinite(scores).all() or not np.isin(y, [0, 1]).all()):
        raise ValueError("Expected matching binary targets and finite N x 7 scores")
    return y.astype(bool), scores


def macro6(values):
    """Always average exactly the six named defects; never silently drop one."""
    selected = [values[label] for label in NAMED_LABELS]
    return None if any(value is None for value in selected) else float(np.mean(selected))


def precision_at_recall(y, scores, targets=RECALL_TARGETS):
    """Enumerate realizable score>=threshold decisions, keeping ties together.

    Maximize precision subject to recall>=target. Equal-precision optima choose
    the highest threshold (fewest predictions). No PR interpolation is used.
    """
    y, scores = np.asarray(y), np.asarray(scores, dtype=np.float64)
    if y.ndim != 1 or y.shape != scores.shape or not np.isfinite(scores).all() or not np.isin(y, [0, 1]).all():
        raise ValueError("Expected finite 1D scores and matching binary targets")
    if any(not 0 < target <= 1 for target in targets):
        raise ValueError("Recall targets must be in (0, 1]")
    keys = [f"{target:.1f}" for target in targets]
    positives = int(y.sum())
    if positives == 0:
        return dict.fromkeys(keys)
    order = np.argsort(-scores, kind="stable")
    sorted_scores = scores[order]
    ends = np.r_[np.flatnonzero(np.diff(sorted_scores)), len(y) - 1]
    true_positives = np.cumsum(y[order])[ends]
    predicted = ends + 1
    recall = true_positives / positives
    precision = true_positives / predicted
    result = {}
    for key, target in zip(keys, targets):
        eligible = np.flatnonzero(recall >= target - 1e-12)
        best = eligible[np.argmax(precision[eligible])]
        threshold = float(sorted_scores[ends[best]])
        result[key] = {
            "threshold_logit": threshold, "threshold_probability": float(expit(threshold)),
            "precision": float(precision[best]), "recall": float(recall[best]),
            "true_positives": int(true_positives[best]), "predicted_positives": int(predicted[best])}
    return result


def per_label_metrics(y, scores):
    """Descriptive test-curve metrics; thresholds here use the supplied labels."""
    y, scores = _arrays(y, scores)
    per_label = {}
    for j, label in enumerate(LABELS):
        truth = y[:, j]
        per_label[label] = {
            "images": len(y), "positives": int(truth.sum()),
            "prevalence": float(truth.mean()) if len(y) else None,
            "ap": float(average_precision_score(truth, scores[:, j])) if truth.any() else None,
            "precision_at_recall": precision_at_recall(truth, scores[:, j])}
    return {
        "per_label": per_label,
        "macro6_ap": macro6({label: per_label[label]["ap"] for label in LABELS}),
        "macro6_precision_at_recall": {
            f"{r:.1f}": macro6({label: (per_label[label]["precision_at_recall"][f"{r:.1f}"] or {}).get("precision")
                               for label in LABELS}) for r in RECALL_TARGETS}}


def evaluate_operating_points(y, scores, operating_points):
    """Apply already-frozen validation thresholds; do not optimize on test."""
    y, scores = _arrays(y, scores)
    per_label = {}
    for j, label in enumerate(LABELS):
        per_label[label] = {}
        for r in RECALL_TARGETS:
            key = f"{r:.1f}"
            point = operating_points[label][key]
            if point is None:
                per_label[label][key] = None
                continue
            predicted = scores[:, j] >= point["threshold_logit"]
            tp = int((predicted & y[:, j]).sum())
            count, positives = int(predicted.sum()), int(y[:, j].sum())
            recall = tp / positives if positives else None
            per_label[label][key] = {
                "target_validation_recall": r, "threshold_logit": point["threshold_logit"],
                "threshold_probability": point["threshold_probability"],
                "validation_precision": point["precision"], "validation_recall": point["recall"],
                "test_precision": tp / count if count else None, "test_recall": recall,
                "test_target_recall_met": recall >= r - 1e-12 if recall is not None else None,
                "true_positives": tp, "predicted_positives": count, "positives": positives}
    return {
        "per_label": per_label,
        "macro6_precision": {f"{r:.1f}": macro6({
            label: (per_label[label][f"{r:.1f}"] or {}).get("test_precision") for label in LABELS})
                             for r in RECALL_TARGETS},
        "macro6_recall": {f"{r:.1f}": macro6({
            label: (per_label[label][f"{r:.1f}"] or {}).get("test_recall") for label in LABELS})
                          for r in RECALL_TARGETS}}


def image_ranking(y, scores):
    """Rank all seven labels; no decision threshold or five-prediction cap.

    MeanRank uses average ranks. Completion is E[last true position] under
    uniform permutations within tied blocks; Top-1 gives fractional tie credit.
    """
    y, scores = _arrays(y, scores)
    indices = np.flatnonzero(y.any(axis=1))
    if not len(indices):
        return {"images": len(y), "issue_images": 0, "mean_rank": None,
                "completion_rank": None, "top1_hit": None}, []
    truth, values = y[indices], scores[indices]
    k = truth.sum(axis=1)
    mean_rank = (rankdata(-values, method="average", axis=1) * truth).sum(axis=1) / k
    last_score = np.where(truth, values, np.inf).min(axis=1)
    last_block = values == last_score[:, None]
    before = (values > last_score[:, None]).sum(axis=1)
    block_size = last_block.sum(axis=1)
    true_in_block = (truth & last_block).sum(axis=1)
    completion = before + true_in_block * (block_size + 1) / (true_in_block + 1)
    top_block = values == values.max(axis=1, keepdims=True)
    top1 = (truth & top_block).sum(axis=1) / top_block.sum(axis=1)
    details = [{"row_index": int(index), "true_issues": int(count),
                "mean_rank": float(mean), "completion_rank": float(last), "top1_hit": float(hit)}
               for index, count, mean, last, hit in zip(indices, k, mean_rank, completion, top1)]
    return {"images": len(y), "issue_images": len(indices), "mean_rank": float(mean_rank.mean()),
            "completion_rank": float(completion.mean()), "top1_hit": float(top1.mean())}, details


def ranking_references(y):
    y = np.asarray(y)
    if y.ndim != 2 or y.shape[1] != len(LABELS) or not np.isin(y, [0, 1]).all():
        raise ValueError("Expected binary N x 7 targets")
    k = y.sum(axis=1)
    k = k[k > 0]
    if not len(k):
        return {name: {"mean_rank": None, "completion_rank": None, "top1_hit": None}
                for name in ("oracle", "random")}
    return {
        "oracle": {"mean_rank": float(((k + 1) / 2).mean()),
                   "completion_rank": float(k.mean()), "top1_hit": 1.0},
        "random": {"mean_rank": (len(LABELS) + 1) / 2,
                   "completion_rank": float(((len(LABELS) + 1) * k / (k + 1)).mean()),
                   "top1_hit": float((k / len(LABELS)).mean())}}
