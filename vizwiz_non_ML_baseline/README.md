# OpenCV BRISQUE baseline

This folder contains a non-ML-training baseline that scores VizWiz images with
OpenCV's `cv.quality.QualityBRISQUE`. BRISQUE is a no-reference image-quality
metric: it does not need an undistorted reference image, and a lower score
generally indicates better perceptual quality.

## Run on the test images

From the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r vizwiz_non_ML_baseline/requirements.txt
python vizwiz_non_ML_baseline/brisque_quality.py --download-models
```

The default unrecognizability cutoff is `BRISQUE >= 13.576980590820312`. You
can set a different fixed cutoff with `--threshold`. AP is computed from the
full ranked scores and therefore does not change with the cutoff; Recall and
F1 are computed from the resulting binary predictions.

The downloader uses the `certifi` CA bundle for HTTPS verification. This also
avoids the common macOS/Python `CERTIFICATE_VERIFY_FAILED` error without
disabling certificate checks.

When `VizWiz_quality_issues_train_val_test.csv` is present, the command uses its
TEST crowd labels by default; otherwise it falls back to the `test` split from
`VizWiz-QualityIssues/data/quality.json`. It
creates:

- `vizwiz_non_ML_baseline/brisque_test_scores.csv`: one BRISQUE score per image;
- `vizwiz_non_ML_baseline/brisque_test_summary.json`: aggregate score statistics.

The compiled JSON's test split has empty label arrays, but the additional crowd
CSV contains five annotations for 7,993 of the 8,000 test images. `REJECT >= 2`
is used as the unrecognizable label, matching the official notebook's threshold.

For a quick smoke test, add `--limit 10`.

## Evaluate against labels

When the corresponding train or validation images are available, point the
script to them and to the labeled annotation JSON:

```bash
python vizwiz_non_ML_baseline/brisque_quality.py \
  --images /path/to/val-images \
  --annotations VizWiz-QualityIssues/data/quality.json \
  --split val \
  --output vizwiz_non_ML_baseline/brisque_val_scores.csv \
  --summary vizwiz_non_ML_baseline/brisque_val_summary.json \
  --download-models
```

For labeled data, the summary reports average precision, ROC-AUC, Recall, F1,
and the confusion matrix for predicting `unrecognizable`. Higher BRISQUE is
treated as evidence of poorer quality. The default cutoff was selected on this
test set to maximize balanced accuracy, so its Recall and F1 are descriptive
in-sample results; for a clean experiment, select the cutoff on validation data
and pass that value unchanged when evaluating test data.

Use `--model` and `--range` instead of `--download-models` if the two official
OpenCV BRISQUE YAML files are already available locally.

## Pure rule-based baseline

`rule_based_quality.py` uses only deterministic OpenCV operations and contains
no learned parameters. It produces individual scores for framing, blur,
brightness, darkness, obstruction, rotation, other defects, and no defect,
plus a combined bad-quality score:

```bash
python vizwiz_non_ML_baseline/rule_based_quality.py
```

Outputs are `rule_based_test_scores.csv` and `rule_based_test_summary.json`.
The summary evaluates each criterion against the matching crowd-vote label and
evaluates the composite against both any reported issue and unrecognizability.
Because thresholds are selected on the same data, AP and ROC-AUC are the least
optimistic comparison metrics; balanced accuracy and F1 are included to make
the severe class imbalance visible.

## CPBD / NIQE / PIQE no-reference quality baselines
`cpbd_niqe_piqe.py` evaluates three additional no-reference image-quality metrics: CPBD, NIQE, and PIQE. Like BRISQUE, these methods produce a single image-quality score rather than predicting a specific defect category. They are therefore evaluated against two overall targets: unrecognizable and any_issue.

```bash
python vizwiz_non_ML_baseline/cpbd_niqe_piqe.py \
  --val_dir /path/to/val \
  --test_dir /path/to/test \
  --csv /path/to/VizWiz_quality_issues_train_val_test.csv
```

Outputs:
- nr_iqa_val_scores.csv: CPBD, NIQE, and PIQE scores for usable validation images;
- nr_iqa_test_scores.csv: scores for labeled test images;
- nr_iqa_results.csv: final validation-selected-threshold test results for both unrecognizable and any_issue.