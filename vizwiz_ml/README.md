# Supervised CNN / ResNet pipeline

The pipeline reads the cleaned `train.csv`, `val.csv`, and `test.csv` directly.
Both `simple_cnn` and `small_resnet` have seven independent binary prediction
heads, in this order:

```text
BLR, BRT, DRK, FRM, OBS, ROT, OTH
```

## Inputs and configuration

A dataset directory can contain:

```text
dataset/
  train.csv
  val.csv
  test.csv
  train/VizWiz_train_00000000.jpg
  val/VizWiz_val_00000000.jpg
  test/VizWiz_test_00000000.jpg
```

All CSVs use the same header:

```text
image_id,BLR,BRT,DRK,FRM,OBS,ROT,OTH
```

The seven values must already be 0 or 1. A value of 1 is used directly as a
positive target; no vote threshold is applied again. Every CSV row is retained,
including all-zero rows. No worker filtering, raw JSON loading, caption loading,
filename conversion or further eligibility filtering takes place. Extra images
in a folder are not used unless their filenames appear in that split's CSV.

CSV names must be image basenames in the corresponding split, with the original
`VizWiz_train_`, `VizWiz_val_`, or `VizWiz_test_` prefix. Duplicate IDs, malformed
rows, nonbinary/missing labels, wrong split names, missing files and unreadable
images cause errors rather than silent exclusions. There are no fixed split-size
requirements; the cleaned CSVs define membership.

Edit `configs/vizwiz.json`:

```json
"data": {
  "labels": "/path/to/dataset",
  "images": "/path/to/dataset"
}
```

`labels` is the directory containing the three CSVs. `images` is the parent of
the three image folders. These may be different directories. Relative paths
resolve against the config file's directory. The supplied config uses `..` for
both, meaning CSVs and image folders beside the repository README.

The sample config uses CUDA, 224-pixel images, batch size 64 and four loader
workers. Adjust batch size for the AWS GPU's memory. Dependencies are listed in
`requirements-ml.txt`; use a CUDA-compatible PyTorch installation on the GPU host.
The package itself never downloads images or pretrained weights.

## Train, select on validation, then test

Run these commands later from the repository root:

```bash
# Train all configured models and freeze the validation-selected winner.
python -m vizwiz_ml.cli experiment \
  --config configs/vizwiz.json --output runs/experiment01

# Calibrate on validation and report the selected checkpoint on test.
python -m vizwiz_ml.cli report \
  --selection runs/experiment01/selected.json \
  --output runs/experiment01/report
```

Training and model selection read only train/validation CSVs and images. Test
files are first opened by the test command. Each candidate's best epoch is
selected by validation six-defect macro P@R0.9 by default, followed by selection of the best
candidate using that same metric. Training completes all configured epochs;
there is no early stopping. AdamW, cosine learning-rate
decay, optional train-only class weighting, gradient clipping and CUDA mixed
precision are retained. All seven heads train with binary cross-entropy logits;
there is no softmax or restriction on the number of positive training targets.

With CUDA AMP enabled, overflowing gradient updates are skipped by GradScaler,
which reduces its loss scale before subsequent batches. Epoch logs/history report
successful updates, skipped AMP steps and the scale. An epoch with no successful
updates stops with an error. Full-precision clipping still raises on nonfinite
gradients. If diagnosing numeric instability, set `training.amp` to `false` to
keep CUDA training in FP32; this may use more GPU memory. Use a new output run
directory when retrying a failed run.

For separate GPU jobs:

```bash
python -m vizwiz_ml.cli train --config configs/vizwiz.json --model cnn --output runs/cnn01
python -m vizwiz_ml.cli train --config configs/vizwiz.json --model resnet --output runs/resnet01
python -m vizwiz_ml.cli select --runs runs/cnn01 runs/resnet01 --output runs/selected01.json
python -m vizwiz_ml.cli report --selection runs/selected01.json --output runs/report01
```

A checkpoint can also be exported on validation with
`python -m vizwiz_ml.cli validate --checkpoint runs/cnn01/best.pt --output runs/val01`.
Use `--device cpu` or `--device cuda:N` on validation/test to override the training
device. Choose new output directories and selection filenames for each run.

Best-epoch ties keep the earlier epoch; candidate ties keep the first candidate.

The default selection configuration is:

~~~json
"selection": {
  "metric": "classification.macro6_p_at_r90",
  "mode": "max"
}
~~~

At each epoch, each label independently chooses the threshold maximizing
validation precision subject to validation recall >= 0.9. Equal scores stay
together. The selection value averages exactly BLR, BRT, DRK, OBS, FRM and ROT;
OTH is reported separately and excluded from this macro. The fixed prediction
threshold and five-item cap do not affect this metric. Raw validation logits
preserve ordering without sigmoid saturation ties; calibration is unnecessary
for this per-label metric. If any of the six labels has no observed validation
positives, the macro is undefined and selection fails rather than dropping it.

This metric controls best.pt and selection across
candidate runs. Per-label values and the macro appear in validation metrics
and history.jsonl. No test data is used. Report calibration and frozen deployment
thresholds are still fitted after checkpoint selection.

Training always runs for training.epochs epochs unless interrupted or an error
occurs. Existing patience settings in older configs are accepted and ignored.
best.pt keeps the best validation checkpoint across the full run; last.pt
contains the final epoch.

To retain the earlier seven-label macro AP selection, explicitly use
"metric": "classification.macro_ap" with "mode": "max". Classification must
remain in the metrics list for either option. Updated defaults apply to new
training runs; existing checkpoints and their saved configurations are unchanged.
Use a new run directory, and compare candidates trained under the same protocol.

Selection requires matching label order, train/validation CSV hashes, prediction
policy and selection metric across runs. Test verifies the frozen checkpoint
hash and CSV hashes. Fix threshold/hyperparameter choices using validation before
running the final test command.

Each run saves `best.pt`, `last.pt`, validation metrics, per-epoch history,
configuration and runtime information. Checkpoints include optimizer/scheduler/
scaler states, but automatic resume is not implemented. CSV-backed seven-label
checkpoints use format version 2; old six-label checkpoints require retraining.
Paths in checkpoints/selections are absolute, so preserve those locations when
moving an experiment. The legacy `vizwiz_ml.prepare` worker-to-JSON helper is
not used by this workflow.

## Evaluation and prediction policy

### Revised metrics and tables from an existing checkpoint

An existing seven-label CNN checkpoint is sufficient; no retraining or
selected.json is required. After syncing the updated code to the training
machine, run from the repository root:

~~~bash
python -m vizwiz_ml.cli report \
  --checkpoint runs/experiment01/cnn/best.pt \
  --output runs/experiment01/cnn/revised_report \
  --model-name "Initial ML model" \
  --device cuda
~~~

The output directory must be new. The checkpoint supplies the architecture,
weights, preprocessing and dataset paths saved during training. Keep the same
train/val CSVs and image locations; CSV hashes are checked against training.
The command makes one validation pass, fits per-label Platt scaling, and freezes
thresholds before opening test data. It then makes one test pass. Network weights
and training checkpoint selection are unchanged.

| Output (inside the report directory) | Contents |
| --- | --- |
| precision_at_recall.csv / .tex | Test prevalence, AP and descriptive test-curve P@R at recall 1.0, 0.9, 0.8 and 0.7 |
| validation_threshold_precision.csv / .tex | Test precision using thresholds selected on validation |
| validation_threshold_recall.csv / .tex | Actual test recall at those same frozen thresholds |
| label_ranking.csv / .tex | Oracle, analytic random order, raw model and calibrated model ranking |
| tables.tex | All four tables; requires booktabs and xcolor with the table option |
| metrics.json | Full metrics, populations and provenance |
| calibration.json | Validation-fitted Platt parameters bound to this checkpoint |
| operating_points.json | Frozen validation thresholds, with validation precision and recall |
| operating_points_test.csv | Thresholds and achieved validation/test metrics by label and target |
| predictions.jsonl | Test labels, logits and raw/calibrated probabilities by image |
| image_ranking.csv | Per-image raw/calibrated ranking diagnostics for images with issues |

Test-curve P@R maximizes precision over realizable score thresholds satisfying
the requested test recall. These descriptive thresholds use test labels.
Operational thresholds instead maximize validation precision subject to the
validation recall target, then stay fixed on test. Test recall can miss the
target; report the achieved recall alongside precision. Tied scores stay
together; equal-precision optima choose the highest feasible threshold.

Macro averages exactly BLR, BRT, DRK, OBS, FRM and ROT. OTH is separate.
NON and UNREC columns contain dashes because the model does not predict them.
AP remains supplementary; the revised tables omit ROC-AUC and fixed-0.5 F1.

Ranking uses all seven issues without a threshold or prediction cap, and only
test images with at least one true issue. MeanRank uses average ranks for ties.
Coverage is expected completion rank under uniformly random ordering within
tied blocks. Top-1 gives fractional credit for ties. Lower MeanRank/Coverage
and higher Top-1 are better. Counts and oracle/random references come from the
actual test labels. Both raw and calibrated rows are exported; Qwen and
prior-only results are not processed.

Ranking and threshold comparisons use raw or calibrated logits, which preserve
sigmoid probability ordering while avoiding artificial ties from saturation.
Top-1 is a fraction in metrics.json and a percentage in the summary tables.
If a label has no test positives, its AP/recall metrics are undefined; a macro
with an undefined component is also undefined. Platt fitting requires both
classes for every label on validation.

To reuse a previously fitted calibration for this exact checkpoint, add
--calibration /path/to/platt.json. Alternatively, use --selection instead of
--checkpoint to report the frozen validation-selected winner.

The original validate/test commands and their capped metrics remain available
as described below. Use report for these revised metrics and paper tables.
Tests for the new definitions and exports are in tests/test_report_metrics.py;
they have been written but not executed here.

### Positive class weighting and post-training calibration

`training.positive_weighting` controls the training loss. When true (the current
default), each label's positive examples receive weight `negatives / positives`
computed on train.csv. For 900 negatives and 100 positives, the positive weight
is 9. Labels with no positives or no negatives use weight 1. This compensates for
imbalance but changes the interpretation of raw sigmoid scores. To disable it,
set `"positive_weighting": false` inside `training` before starting a new run.
Changing the config does not undo weighting in an already-trained checkpoint.

Platt scaling can be fitted after training without changing or retraining the
model. It fits `sigmoid(a_c * logit_c + b_c)` separately for all seven labels,
using validation images/labels only, unweighted binary log loss, and small L2
regularization toward identity. Slopes are constrained to `a_c >= 1e-6` to avoid
reversing per-label rankings. Each validation label must have both positive and
negative examples; otherwise calibration stops with an explanatory error.

After the experiment finishes, run later:

```bash
python -m vizwiz_ml.cli calibrate \
  --selection runs/experiment01/selected.json \
  --output runs/experiment01/platt.json

python -m vizwiz_ml.cli test \
  --selection runs/experiment01/selected.json \
  --calibration runs/experiment01/platt.json \
  --output runs/experiment01/test_calibrated
```

An individual model can be calibrated with `--checkpoint runs/cnn01/best.pt`
instead of `--selection`. Both `calibrate` and evaluation support `--device`.
`calibrate --l2` changes the regularization strength (default 0.0001).
Calibration is a separate JSON file bound to the checkpoint hash, seven-label
order and dataset protocol. Existing seven-label checkpoints remain usable.
Evaluation rejects a calibration file belonging to another model or dataset.

The `--calibration` flag also works with `validate`. With that flag, probabilities,
predicted issue ordering and metrics use calibrated scores; predictions also
contain `raw_probabilities`, and metrics include `raw_metrics` for comparison.
The calibration file is copied into the results directory. Without the flag,
scores remain uncalibrated. The prediction threshold and five-item cap remain
as configured in those legacy commands. The report command above selects
validation recall thresholds and computes the revised ranking metrics.

Calibration never reads test data or updates network weights. Its validation
NLL diagnostics are measured on the same data used for fitting, so they are not
an independent estimate of calibration quality. Validation is already used for
checkpoint selection; a reserved validation calibration subset is preferable
when data permits. Freeze calibration before inspecting final test results.

### Metrics

All seven probabilities and the ordered predicted issue list are written to
`predictions.jsonl`, keyed by `image_id`. Metrics and the evaluated CSV's hash
are written to `metrics.json`.

The original prediction cap stays at five: include probabilities >= 0.5, sort
in descending order, then keep at most five. Ties follow LABELS order; an empty
list is allowed. Threshold and cap are configured before selection, and the cap
can be lowered to 1–4. Training targets themselves are never capped.

Per-label AP/AUROC use continuous probabilities. Precision/recall/F1, exact match
and Hamming loss evaluate the capped predictions. Macro AP excludes labels with
no positives and reports how many labels have defined AP; selection fails if its
metric is entirely undefined. Class weighting may affect probability calibration.

Coverage asks whether all real issues appear in the predicted list. Empty true
sets count as covered; coverage excluding those images is also reported. Images
with six or seven true issues cannot fit a five-item list, which is reflected by
`cap_feasible_fraction`. Top-k coverage diagnostics are included.

Correction metrics assume each action successfully fixes one suggested real
issue, false suggestions waste an action, and completion is observable. The
minimum is the number of true issues r. For covered images with m suggestions:
ranked completion is the position of the last true issue; uniformly random
suggestion order has expected completion r*(m+1)/(r+1). Empty true sets require
zero actions and are excluded from ratios to the minimum. Covered-only step
averages, excess steps and ratios are reported alongside failure counts.
Uncovered images cannot finish using that list: unconditional expected steps
are marked infinite with JSON null and an explicit flag. A separately named
oracle-fallback diagnostic exhausts a failed list and then fixes its missed
issues in one action each. No image updates or re-inference after fixes are
simulated.

Full-image letterboxing, gray padding and normalization to [-1, 1] are retained.
There are no crops, flips, rotations, exposure/blur augmentation or EXIF rotation
corrections that could change quality labels. Downsampling can hide subtle flaws;
image size is a validation-tuned choice.

## Extending and checking later

The default configuration includes four models:

| Model ID | Backbone | Conventional depth | Stage channels |
| --- | --- | --- | --- |
| cnn | Four stride-2 3x3 convolutions with GroupNorm/ReLU | 5 | 32, 64, 128, 128 |
| resnet | 3x3 stride-2 stem and four two-block residual stages; GroupNorm | 18 | 32, 64, 128, 256 |
| resnet18 | 7x7 stride-2 stem, stride-2 max pool and four two-block residual stages; BatchNorm | 18 | 64, 128, 256, 512 |
| resnet34 | Same standard stem; residual stages with 3, 4, 6, 3 blocks; BatchNorm | 34 | 64, 128, 256, 512 |

The CNN has four convolutions and one linear output layer along each prediction
path. All residual networks have two 3x3 convolutions per block and projection
skips when changing shape. Conventional depth counts the stem, main-branch
convolutions (16 for the 18-layer models, 32 for ResNet-34), and one output
linear layer; projection skips, normalization, pooling and activations are not
counted.
All models use global average pooling, dropout (0.2 by default), and seven
independent linear heads producing logits for BCEWithLogitsLoss.

ResNet-18 and ResNet-34 are implemented locally and trained from scratch, with no pretrained
weights or torchvision dependency. Its standard backbone uses the existing
letterboxing/normalization and seven-head classifier for this task.
Train just this new candidate without repeating the older runs:

~~~bash
python -m vizwiz_ml.cli train \
  --config configs/vizwiz.json --model resnet18 --output runs/resnet18_01
python -m vizwiz_ml.cli report \
  --checkpoint runs/resnet18_01/best.pt \
  --output runs/resnet18_01/revised_report --model-name "ResNet-18" --device cuda
~~~

For ResNet-34, use --model resnet34 and a new output directory, for example:

~~~bash
python -m vizwiz_ml.cli train \
  --config configs/vizwiz.json --model resnet34 --output runs/resnet34_01
python -m vizwiz_ml.cli report \
  --checkpoint runs/resnet34_01/best.pt \
  --output runs/resnet34_01/revised_report --model-name "ResNet-34" --device cuda
~~~

The experiment command now trains all four configured candidates.

Training and inference use tqdm progress bars showing batch completion, elapsed
time, estimated remaining time and throughput. Training also shows running mean
loss and AMP-skipped updates. Bars display automatically in terminals/notebooks
and are disabled for noninteractive output; epoch summaries are still printed.
Install the added dependency on the training machine if needed:

~~~bash
python -m pip install "tqdm>=4.66,<5"
~~~

Each epoch summary prints train, validation and total epoch durations in seconds.
The same values are saved as train_seconds, validation_seconds and epoch_seconds
in history.jsonl. Validation duration includes metric calculation; total epoch
duration also includes checkpoint saving, excluding the final log write.

Additional models can register builders in `models.py` and add candidate specs
to configuration. They must return a batch x 7 tensor of binary logits. Metrics
remain extensible through `register_metric` in `metrics.py`.

Checks in `tests/test_supervised.py` cover direct binary CSV targets, OTH,
malformed/split-mismatched rows, model heads, correction metrics, selection and
a small CPU train/select/test workflow. They are written for later execution:

```bash
python -m unittest discover -s tests -p test_supervised.py -v
```

No Python code, training, evaluation, tests, installs or downloads were run while
making these changes.
