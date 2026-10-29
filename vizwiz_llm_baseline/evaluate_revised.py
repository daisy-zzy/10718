"""Revised metrics: per-label precision at fixed recall, and Platt-calibrated label ranking.

Standard library only; reads saved scores, no inference. Platt parameters and the prior-only
label order are fit on a calibration split (validation by default) and applied unchanged to the
evaluation split. Pointing the calibration arguments at the evaluation split gives in-sample
calibration, which is recorded in the output.
"""
import argparse, csv, json, math
from pathlib import Path
from protocol import DATA, LABELS

SIX = ['BLR', 'BRT', 'DRK', 'OBS', 'FRM', 'ROT']
ISSUES = SIX + ['OTH']  # labels ranked per image; NON is not an issue, UNREC is a separate task
RECALLS = [1.0, 0.9, 0.8, 0.7]
MODELS = [('Qwen3.5-0.8B', 'qwen35_08b'), ('Qwen3.5-2B', 'qwen35_2b')]

def load_truth(path):
    """Accepted-worker, >=2-vote labels; images missing any label are dropped.

    Reads either a labels JSON written by prepare_data.py or a joined predictions CSV written
    by evaluate.py (only its *_truth columns are used).
    """
    truth = {}
    if path.suffix == '.json':
        for r in json.loads(path.read_text()):
            labels = r['accepted']['labels']
            if all(labels[k] is not None for k in LABELS): truth[r['image']] = {k: labels[k] for k in LABELS}
    else:
        for r in csv.DictReader(open(path)):
            vals = [r[k + '_truth'] for k in LABELS]
            if all(v not in ('', 'nan') for v in vals):
                truth[r['image']] = {k: int(float(v)) for k, v in zip(LABELS, vals)}
    return truth

def aligned_scores(folder, names, model):
    scores = load_scores(folder)
    if not set(names) <= set(scores): raise ValueError(f'{model}: missing predictions in {folder}')
    return {k: [scores[n][k] for n in names] for k in LABELS}

def load_scores(folder):
    scores = {}
    for path in sorted(folder.glob('shard*.jsonl')):
        for line in open(path):
            r = json.loads(line)
            if r['image'] in scores: raise ValueError('Duplicate prediction: ' + r['image'])
            scores[r['image']] = r['scores']
    return scores

def pr_points(y, s):
    """(precision, recall) after each distinct score threshold, highest score first."""
    order = sorted(range(len(s)), key=lambda i: -s[i])
    pos = sum(y); tp = fp = 0; points = []
    for n, i in enumerate(order):
        tp += y[i]; fp += 1 - y[i]
        if n + 1 == len(order) or s[order[n + 1]] != s[i]:
            points.append((tp / (tp + fp), tp / pos))
    return points

def average_precision(y, s):
    ap = prev = 0.0
    for p, r in pr_points(y, s): ap += (r - prev) * p; prev = r
    return ap

def precision_at_recall(y, s, target):
    """Best precision over thresholds whose recall reaches the target."""
    return max(p for p, r in pr_points(y, s) if r >= target - 1e-12)

def logit(p, eps=1e-6):
    p = min(max(p, eps), 1 - eps)
    return math.log(p / (1 - p))

def sigmoid(z):
    return 1 / (1 + math.exp(-z)) if z >= 0 else math.exp(z) / (1 + math.exp(z))

def fit_platt(y, s, iters=100):
    """Maximum-likelihood p = sigmoid(a * logit(s) + b) with a >= 0, by damped Newton steps."""
    x = [logit(v) for v in s]; a, b = 1.0, 0.0
    def nll(a, b):
        return -sum(yi * math.log(max(sigmoid(a * xi + b), 1e-300)) + (1 - yi) * math.log(max(1 - sigmoid(a * xi + b), 1e-300)) for xi, yi in zip(x, y))
    cur = nll(a, b)
    for _ in range(iters):
        ga = gb = haa = hab = hbb = 0.0
        for xi, yi in zip(x, y):
            p = sigmoid(a * xi + b); w = p * (1 - p)
            ga += (p - yi) * xi; gb += p - yi; haa += w * xi * xi; hab += w * xi; hbb += w
        det = haa * hbb - hab * hab
        da = (hbb * ga - hab * gb) / det; db = (haa * gb - hab * ga) / det; step = 1.0
        while step > 1e-8 and nll(a - step * da, b - step * db) > cur: step /= 2
        a, b = a - step * da, b - step * db; new = nll(a, b)
        if cur - new < 1e-10: cur = new; break
        cur = new
    if a < 0:  # never let labels reverse the model's ordering; a = 0 falls back to the label prior
        prior = sum(y) / len(y); a, b = 0.0, math.log(prior / (1 - prior))
    return a, b

def ece(y, p, bins=10):
    total = 0.0
    for k in range(bins):
        idx = [i for i, v in enumerate(p) if min(int(v * bins), bins - 1) == k]
        if idx: total += len(idx) / len(p) * abs(sum(p[i] for i in idx) / len(idx) - sum(y[i] for i in idx) / len(idx))
    return total

def ranks_desc(values):
    """1-based ranks, highest value first; tied values share their average rank."""
    order = sorted(range(len(values)), key=lambda i: -values[i]); ranks = [0.0] * len(values); n = 0
    while n < len(order):
        m = n
        while m + 1 < len(order) and values[order[m + 1]] == values[order[n]]: m += 1
        for i in order[n:m + 1]: ranks[i] = (n + m) / 2 + 1
        n = m + 1
    return ranks

def ranking_metrics(truth_rows, score_rows):
    """Mean rank of true issues (averaged within image, then over images), coverage, top-1 hit."""
    mean_rank = coverage = top1 = 0.0; n = 0
    for y, s in zip(truth_rows, score_rows):
        positives = [j for j, v in enumerate(y) if v]
        if not positives: continue
        r = ranks_desc(s); best = min(r)
        tied_top = [j for j, v in enumerate(r) if v == best]
        mean_rank += sum(r[j] for j in positives) / len(positives)
        coverage += max(r[j] for j in positives)
        top1 += sum(y[j] for j in tied_top) / len(tied_top)  # expected hit under random tie-breaking
        n += 1
    return {'images': n, 'mean_rank': mean_rank / n, 'coverage': coverage / n, 'top1_hit': top1 / n}

def random_ranking(truth_rows):
    """Closed-form expectations for a uniformly random order of the issue labels."""
    L = len(ISSUES); rows = [sum(y) for y in truth_rows if sum(y)]
    return {'images': len(rows), 'mean_rank': (L + 1) / 2,
            'coverage': sum(k * (L + 1) / (k + 1) for k in rows) / len(rows),
            'top1_hit': sum(k / L for k in rows) / len(rows)}

def oracle_ranking(truth_rows):
    """Best achievable values: all true issues ranked first."""
    rows = [sum(y) for y in truth_rows if sum(y)]
    return {'images': len(rows), 'mean_rank': sum((k + 1) / 2 for k in rows) / len(rows),
            'coverage': sum(rows) / len(rows), 'top1_hit': 1.0}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--results', type=Path, default=Path('results/independent'),
                    help='Evaluation-split scores: one folder of shard*.jsonl per model')
    ap.add_argument('--truth', type=Path, default=DATA / 'test_labels.json',
                    help='Evaluation-split labels (prepare_data.py JSON or evaluate.py predictions CSV)')
    ap.add_argument('--calibration-results', type=Path, default=Path('results/independent_val'),
                    help='Calibration-split scores, same layout as --results')
    ap.add_argument('--calibration-truth', type=Path, default=DATA / 'val_labels.json',
                    help='Calibration-split labels, same formats as --truth')
    ap.add_argument('--output', type=Path)
    args = ap.parse_args()
    out = args.output or args.results / 'evaluation_revised'; out.mkdir(parents=True, exist_ok=True)
    truth = load_truth(args.truth); names = sorted(truth)
    y = {k: [truth[n][k] for n in names] for k in LABELS}
    issue_truth = [[truth[n][k] for k in ISSUES] for n in names]
    prevalence = {k: sum(v) / len(v) for k, v in y.items()}
    cal_truth = load_truth(args.calibration_truth); cal_names = sorted(cal_truth)
    cal_y = {k: [cal_truth[n][k] for n in cal_names] for k in ISSUES}
    prior_order = [sum(cal_y[k]) / len(cal_names) for k in ISSUES]
    in_sample = bool(set(cal_names) & set(names))
    result = {'evaluation_images': len(names), 'prevalence': prevalence, 'recall_targets': RECALLS,
              'ranked_labels': ISSUES,
              'calibration': {'method': 'Per-label Platt scaling with slope >= 0; prior-only order from the same split',
                              'results': str(args.calibration_results), 'truth': str(args.calibration_truth),
                              'images': len(cal_names), 'overlaps_evaluation_images': in_sample},
              'ranking_reference': {'oracle': oracle_ranking(issue_truth), 'random': random_ranking(issue_truth),
                                    'prior_only': ranking_metrics(issue_truth, [prior_order] * len(names))},
              'models': {}}
    if in_sample: print('Warning: calibration and evaluation images overlap; calibrated ranking is in-sample')
    rows = []
    for model, folder in MODELS:
        s = aligned_scores(args.results / folder, names, model)
        cal_s = aligned_scores(args.calibration_results / folder, cal_names, model)
        per = {}
        for k in LABELS:
            per[k] = {'AP': average_precision(y[k], s[k]),
                      **{f'P@R{r:.1f}': precision_at_recall(y[k], s[k], r) for r in RECALLS}}
            rows.append({'model': model, 'label': k, 'positives': sum(y[k]), 'prevalence': prevalence[k], **per[k]})
        macro = {m: sum(per[k][m] for k in SIX) / len(SIX) for m in per['BLR']}
        platt = {}; calibrated = {}
        for k in ISSUES:
            a, b = fit_platt(cal_y[k], cal_s[k]); calibrated[k] = [sigmoid(a * logit(v) + b) for v in s[k]]
            platt[k] = {'a': a, 'b': b, 'ECE_raw': ece(y[k], s[k]), 'ECE_calibrated': ece(y[k], calibrated[k])}
        raw_rows = [[s[k][i] for k in ISSUES] for i in range(len(names))]
        cal_rows = [[calibrated[k][i] for k in ISSUES] for i in range(len(names))]
        result['models'][model] = {'per_label': per, 'six_defect_macro': macro, 'platt': platt,
                                   'ranking_raw': ranking_metrics(issue_truth, raw_rows),
                                   'ranking_calibrated': ranking_metrics(issue_truth, cal_rows)}
    (out / 'metrics.json').write_text(json.dumps(result, indent=2))
    with open(out / 'per_label_precision_at_recall.csv', 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    print(json.dumps(result, indent=2))

if __name__ == '__main__': main()
