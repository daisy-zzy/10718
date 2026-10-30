"""Metric plugins receive (targets, probabilities, observed_mask, policy).

    Returns JSON-safe scalars/dictionaries; undefined quantities are None.
"""
import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

from . import LABELS
from .report_metrics import NAMED_LABELS, macro6, precision_at_recall

METRIC_REGISTRY = {}


def register_metric(name):
    def decorate(function):
        if name in METRIC_REGISTRY:
            raise ValueError(f"Duplicate metric: {name}")
        METRIC_REGISTRY[name] = function
        return function
    return decorate


def prediction_lists(probabilities, threshold=0.5, max_predictions=5):
    p = np.asarray(probabilities, dtype=float)
    if p.ndim != 2 or p.shape[1] != len(LABELS) or not np.isfinite(p).all():
        raise ValueError(f"Expected finite N x {len(LABELS)} probabilities")
    if ((p < 0) | (p > 1)).any():
        raise ValueError("Probabilities must be in [0, 1]")
    if not 0 <= threshold <= 1 or type(max_predictions) is not int or not 1 <= max_predictions <= 5:
        raise ValueError("Use threshold in [0, 1] and max_predictions in [1, 5]")
    # Stable ties follow LABELS order; threshold equality is positive.
    ranked = np.argsort(-p, axis=1, kind="stable")
    return [[int(j) for j in order if p[i, j] >= threshold][:max_predictions]
            for i, order in enumerate(ranked)]


def _mean(values):
    return float(np.mean(values)) if len(values) else None


def _binary(y, prediction):
    tp = int((y & prediction).sum())
    fp = int((~y & prediction).sum())
    fn = int((y & ~prediction).sum())
    tn = int((~y & ~prediction).sum())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    return {"precision": precision, "recall": recall,
            "f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0,
            "tp": tp, "fp": fp, "fn": fn, "tn": tn}


@register_metric("classification")
def classification(y, p, mask, policy, selection_scores=None):
    lists = prediction_lists(p, **policy)
    capped = np.zeros_like(y, dtype=bool)
    for i, indices in enumerate(lists):
        capped[i, indices] = True
    per_label = {}
    # Raw logits avoid artificial sigmoid saturation ties during selection.
    ranking = p if selection_scores is None else selection_scores
    for j, label in enumerate(LABELS):
        known = mask[:, j]
        truth, prob = y[known, j], p[known, j]
        point = precision_at_recall(truth, ranking[known, j], targets=(0.9,))["0.9"]
        per_label[label] = {
            "observed": int(known.sum()), "positives": int(truth.sum()),
            "ap": float(average_precision_score(truth, prob)) if truth.any() else None,
            "auroc": float(roc_auc_score(truth, prob)) if len(np.unique(truth)) == 2 else None,
            "p_at_r90": point["precision"] if point is not None else None,
            "recall_at_p_at_r90": point["recall"] if point is not None else None,
            **_binary(truth, capped[known, j])}
    aps = [m["ap"] for m in per_label.values() if m["ap"] is not None]
    f1s = [m["f1"] for m in per_label.values() if m["observed"]]
    complete = mask.all(axis=1)
    micro = _binary(y[mask], capped[mask])
    return {"per_label": per_label, "macro_ap": _mean(aps),
            "macro6_p_at_r90": macro6({label: per_label[label]["p_at_r90"] for label in LABELS}),
            "p_at_r90_macro_labels": list(NAMED_LABELS),
            "macro_ap_defined_labels": len(aps), "macro_f1": _mean(f1s),
            "micro_f1": micro["f1"] if mask.any() else None,
            "hamming_loss": _mean((y != capped)[mask]),
            "exact_match": _mean((y[complete] == capped[complete]).all(axis=1)),
            "mean_predicted_issues": _mean(capped.sum(axis=1))}


@register_metric("correction")
def correction(y, p, mask, policy):
    complete = mask.all(axis=1)
    y, p = y[complete], p[complete]
    lists = prediction_lists(p, **policy)
    covered, nonempty_covered = [], []
    expected, ranked_steps, excess, ratios = [], [], [], []
    covered_minimum, ranked_ratios = [], []
    fallback_steps, fallback_ratios = [], []
    for truth, indices in zip(y, lists):
        real = set(np.flatnonzero(truth).tolist())
        r, m = len(real), len(indices)
        success = real.issubset(indices)
        covered.append(success)
        if r:
            nonempty_covered.append(success)
        if success:
            # Uniform random permutation without replacement; expected maximum
            # of r true positions among m suggestions is r*(m+1)/(r+1).
            steps = r * (m + 1) / (r + 1) if r else 0.0
            ranked = max((i + 1 for i, j in enumerate(indices) if j in real), default=0)
            expected.append(steps)
            ranked_steps.append(ranked)
            excess.append(steps - r)
            covered_minimum.append(r)
            if r:
                ratios.append(steps / r)
                ranked_ratios.append(ranked / r)
            fallback = ranked
        else:
            # Explicit oracle fallback: exhaust the list, then fix each missing
            # real issue in one extra action. This is not a deployed capability.
            fallback = m + len(real.difference(indices))
        fallback_steps.append(fallback)
        if r:
            fallback_ratios.append(fallback / r)
    topk = {}
    order = np.argsort(-p, axis=1, kind="stable")
    for k in range(policy["max_predictions"] + 1):
        success = [set(np.flatnonzero(row)).issubset(indices[:k]) for row, indices in zip(y, order)]
        topk[str(k)] = _mean(success)
    failures = len(covered) - sum(covered)
    return {
        "complete_label_images": len(y), "nonempty_truth_images": int(y.any(axis=1).sum()),
        "coverage": _mean(covered), "coverage_nonempty": _mean(nonempty_covered),
        "unresolved_images": int(failures),
        "cap_feasible_fraction": _mean(y.sum(axis=1) <= policy["max_predictions"]),
        "top_k_coverage_ignoring_threshold": topk,
        "minimum_steps_mean": _mean(y.sum(axis=1)),
        "minimum_steps_covered_mean": _mean(covered_minimum),
        "expected_steps_covered_mean": _mean(expected),
        "ranked_steps_covered_mean": _mean(ranked_steps),
        "expected_excess_steps_covered_mean": _mean(excess),
        "expected_over_minimum_covered_nonempty_mean": _mean(ratios),
        "ranked_over_minimum_covered_nonempty_mean": _mean(ranked_ratios),
        "unconditional_expected_steps": None if failures else _mean(expected),
        "unconditional_expected_steps_is_infinite": bool(failures),
        "oracle_fallback_ranked_steps_mean": _mean(fallback_steps),
        "oracle_fallback_over_minimum_nonempty_mean": _mean(fallback_ratios)}


def evaluate_metrics(y, probabilities, mask, policy, metrics=("classification", "correction"),
                     selection_scores=None):
    y, p, mask = np.asarray(y), np.asarray(probabilities), np.asarray(mask, dtype=bool)
    if y.shape != p.shape or mask.shape != p.shape or p.ndim != 2:
        raise ValueError(f"Targets, probabilities and mask must have identical N x {len(LABELS)} shapes")
    if not np.isin(y[mask], [0, 1]).all():
        raise ValueError("Observed labels must be binary")
    prediction_lists(p, **policy)  # Validate even when no complete-label rows exist.
    if selection_scores is not None:
        selection_scores = np.asarray(selection_scores, dtype=np.float64)
        if selection_scores.shape != p.shape or not np.isfinite(selection_scores).all():
            raise ValueError("Selection scores must be finite and match the target shape")
    y = y.astype(bool)
    result = {"images": len(y), "observed_label_entries": int(mask.sum()),
              "missing_label_images": int((~mask.all(axis=1)).sum()), "policy": policy}
    for name in metrics:
        if name not in METRIC_REGISTRY:
            raise ValueError(f"Unknown metric {name!r}")
        if name == "classification" and selection_scores is not None:
            result[name] = classification(y, p, mask, policy, selection_scores=selection_scores)
        else:
            result[name] = METRIC_REGISTRY[name](y, p, mask, policy)
    return result
