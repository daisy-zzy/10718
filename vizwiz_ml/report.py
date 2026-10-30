"""Reload a trained checkpoint and export the revised evaluation tables."""
import csv
import json
from pathlib import Path

import numpy as np
import torch
from scipy.special import expit

from . import LABELS
from .calibration import check_calibration, fit_platt, platt_logits
from .data import load_records, targets
from .engine import get_device, load_checkpoint, make_loader, predict
from .io import read_json, sha256, write_json
from .report_metrics import (NAMED_LABELS, RECALL_TARGETS, TABLE_COLUMNS,
                             evaluate_operating_points, image_ranking,
                             macro6, per_label_metrics, precision_at_recall,
                             ranking_references)


def _csv(path, columns, rows):
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def _table_values(values, macro=True):
    return {column: macro6(values) if column == "Macro" and macro else values.get(column)
            for column in TABLE_COLUMNS}


def _escape(value):
    replacements = {"\\": r"\textbackslash{}", "&": r"\&", "%": r"\%", "$": r"\$",
                    "#": r"\#", "_": r"\_", "{": r"\{", "}": r"\}",
                    "~": r"\textasciitilde{}", "^": r"\textasciicircum{}"}
    return "".join(replacements.get(char, char) for char in str(value))


def _number(value, decimals=3):
    return "---" if value is None else f"{value:.{decimals}f}"


def _precision_table(caption, model_name, rows):
    lines = [r"\begin{table}[htbp]", r"\centering\small",
             r"\providecolor{TargetRow}{RGB}{235,242,250}",
             r"\caption{" + _escape(caption) + "}",
             r"\begin{tabular}{@{}lrrrrrrrrrr@{}}", r"\toprule",
             " & ".join(["Metric", *TABLE_COLUMNS]) + r" \\", r"\midrule",
             r"\multicolumn{11}{@{}l}{\textit{" + _escape(model_name) + r"}} \\"]
    for name, values in rows:
        if name.endswith("0.9"):
            lines.append(r"\rowcolor{TargetRow}")
        lines.append(" & ".join([_escape(name), *[_number(values[column]) for column in TABLE_COLUMNS]]) + r" \\")
    return "\n".join([*lines, r"\bottomrule", r"\end{tabular}", r"\end{table}", ""])


def write_tables(output, model_name, test_curve, operational, rankings):
    per_label = test_curve["per_label"]
    prevalence = _table_values({label: per_label[label]["prevalence"] for label in LABELS}, macro=False)
    ap = _table_values({label: per_label[label]["ap"] for label in LABELS})
    curve_rows = [("Prevalence", prevalence), ("AP", ap)]
    precision_rows = [("Prevalence", prevalence), ("AP", ap)]
    recall_rows = []
    for r in RECALL_TARGETS:
        key = f"{r:.1f}"
        curve_rows.append((f"P@R{key}", _table_values({
            label: (per_label[label]["precision_at_recall"][key] or {}).get("precision") for label in LABELS})))
        precision_rows.append((f"P@valR{key}", _table_values({
            label: (operational["per_label"][label][key] or {}).get("test_precision") for label in LABELS})))
        recall_rows.append((f"Recall@valR{key}", _table_values({
            label: (operational["per_label"][label][key] or {}).get("test_recall") for label in LABELS})))
    specifications = [
        ("precision_at_recall", curve_rows,
         "Descriptive test-curve AP and maximum precision at fixed test recall. "
         "These thresholds use test labels and are not deployment thresholds. "
         "Macro averages six named defects; OTH is separate. NON and UNREC are not predicted."),
        ("validation_threshold_precision", precision_rows,
         "Test precision at thresholds selected on validation to maximize precision subject to each "
         "recall target. Test recall may miss the target. AP is measured on test. "
         "Macro averages six named defects; OTH is separate."),
        ("validation_threshold_recall", recall_rows,
         "Achieved test recall at the same validation-selected thresholds. "
         "Row names indicate validation recall targets, not guaranteed test recall.")]
    tables = []
    for stem, rows, caption in specifications:
        _csv(output / f"{stem}.csv", ["Metric", *TABLE_COLUMNS],
             [{"Metric": name, **values} for name, values in rows])
        tex = _precision_table(caption, model_name, rows)
        (output / f"{stem}.tex").write_text(tex, encoding="utf-8")
        tables.append(tex)

    ranking_rows = []
    for method, scores, values in (
            ("Oracle", "---", rankings["oracle"]),
            ("Random order", "---", rankings["random"]),
            (model_name, "raw", rankings["raw"]),
            (model_name, "calibrated", rankings["calibrated"])):
        ranking_rows.append({"Method": method, "Scores": scores, "MeanRank": values["mean_rank"],
                             "Coverage": values["completion_rank"],
                             "Top1HitPercent": None if values["top1_hit"] is None else 100 * values["top1_hit"]})
    _csv(output / "label_ranking.csv", ["Method", "Scores", "MeanRank", "Coverage", "Top1HitPercent"], ranking_rows)
    lines = [r"\begin{table}[htbp]", r"\centering\small",
             r"\caption{Seven-issue ranking over " + str(rankings["calibrated"]["issue_images"]) +
             r" test images with at least one issue. Platt scaling is fitted on validation only. "
             r"Coverage is expected completion rank under random tie-breaking; Top-1 uses fractional tie credit.}",
             r"\begin{tabular}{@{}llccc@{}}", r"\toprule",
             r"Method & Scores & Mean rank $\downarrow$ & Coverage $\downarrow$ & Top-1 hit (\%) $\uparrow$ \\",
             r"\midrule"]
    for row in ranking_rows:
        lines.append(" & ".join([_escape(row["Method"]), _escape(row["Scores"]),
                                 _number(row["MeanRank"]), _number(row["Coverage"]),
                                 _number(row["Top1HitPercent"], 1)]) + r" \\")
    tex = "\n".join([*lines, r"\bottomrule", r"\end{tabular}", r"\end{table}", ""])
    (output / "label_ranking.tex").write_text(tex, encoding="utf-8")
    tables.append(tex)
    (output / "tables.tex").write_text(
        "% Requires booktabs and \\usepackage[table]{xcolor}.\n\n" + "\n".join(tables), encoding="utf-8")


def report_checkpoint(path, output, device_name=None, selection=None,
                      calibration_path=None, model_name=None, l2=1e-4):
    """Fit/freeze everything on val before opening test; never retrain the CNN."""
    output = Path(output)
    if output.exists():
        raise FileExistsError(f"Report directory already exists: {output}")
    checkpoint, model = load_checkpoint(path, torch.device("cpu"))
    checkpoint_hash = sha256(path)
    if selection is not None and (
            checkpoint["protocol"] != selection["protocol"]
            or checkpoint_hash != selection["winner"]["checkpoint_sha256"]):
        raise ValueError("Checkpoint is not the frozen validation winner")
    config, protocol = checkpoint["config"], checkpoint["protocol"]
    labels_dir = Path(config["data"]["labels"])
    for split in ("train", "val"):
        if sha256(labels_dir / f"{split}.csv") != protocol[f"{split}_sha256"]:
            raise ValueError(f"{split}.csv changed since training")
    train_names = {row["image"] for row in load_records(labels_dir / "train.csv", "train")}
    val_records = load_records(labels_dir / "val.csv", "val")
    val_names = {row["image"] for row in val_records}
    if train_names & val_names:
        raise ValueError("Training and validation images overlap")
    device = get_device(device_name or config["device"])
    model.to(device)
    loader = make_loader(val_records, config, "val", device)
    names, val_logits = predict(model, loader, device, config["training"]["amp"], return_logits=True)
    if names != [row["image"] for row in val_records]:
        raise RuntimeError("Validation prediction order differs from val.csv")
    val_y, _ = targets(val_records)
    if calibration_path is not None:
        calibration = read_json(calibration_path)
        check_calibration(calibration, checkpoint_hash, protocol)
    else:
        calibration = fit_platt(val_logits, val_y, l2=l2)
        calibration.update({"fit_split": "val", "checkpoint_sha256": checkpoint_hash,
                            "checkpoint": str(Path(path).resolve()), "protocol": protocol,
                            "validation_csv_sha256": protocol["val_sha256"],
                            "fit_diagnostics": "NLL is in-sample on validation; test is held out"})
    val_scores = platt_logits(val_logits, calibration)
    operating_points = {label: precision_at_recall(val_y[:, j], val_scores[:, j])
                        for j, label in enumerate(LABELS)}
    if sha256(path) != checkpoint_hash or sha256(labels_dir / "val.csv") != protocol["val_sha256"]:
        raise RuntimeError("Model or validation data changed during fitting")
    # These files are frozen before any test labels or test images are opened.
    output.mkdir(parents=True)
    write_json(output / "calibration.json", calibration)
    frozen = {"fit_split": "val", "checkpoint_sha256": checkpoint_hash, "protocol": protocol,
              "calibration_sha256": sha256(output / "calibration.json"), "labels": list(LABELS),
              "score_space": "calibrated logits; compare score >= threshold_logit",
              "recall_targets": list(RECALL_TARGETS), "per_label": operating_points,
              "tie_rule": "highest threshold among equal-precision feasible decisions"}
    write_json(output / "operating_points.json", frozen)

    test_path = labels_dir / "test.csv"
    test_hash = sha256(test_path)
    test_records = load_records(test_path, "test")
    if {row["image"] for row in test_records} & (train_names | val_names):
        raise ValueError("Test images overlap training/validation")
    loader = make_loader(test_records, config, "test", device)
    test_names, logits = predict(model, loader, device, config["training"]["amp"], return_logits=True)
    if test_names != [row["image"] for row in test_records]:
        raise RuntimeError("Test prediction order differs from test.csv")
    y, _ = targets(test_records)
    scores = platt_logits(logits, calibration)
    test_curve = per_label_metrics(y, scores)
    operational = evaluate_operating_points(y, scores, operating_points)
    raw_ranking, raw_details = image_ranking(y, logits)
    calibrated_ranking, calibrated_details = image_ranking(y, scores)
    rankings = {"raw": raw_ranking, "calibrated": calibrated_ranking, **ranking_references(y)}
    name = model_name or checkpoint["model_spec"]["id"]
    result = {
        "model_name": name, "checkpoint": str(Path(path).resolve()), "checkpoint_sha256": checkpoint_hash,
        "epoch": checkpoint["epoch"], "labels": list(LABELS), "macro_labels": list(NAMED_LABELS),
        "test_images": len(y), "test_csv_sha256": test_hash, "protocol": protocol,
        "calibration_sha256": frozen["calibration_sha256"], "recall_targets": list(RECALL_TARGETS),
        "test_curve": test_curve, "test_at_validation_thresholds": operational, "image_ranking": rankings,
        "definitions": {
            "test_curve": "Descriptive test-label-selected thresholds; never used for deployment",
            "operational": "Thresholds selected on validation only; report actual test precision/recall",
            "ranking": "All seven labels, no threshold or prediction cap; positive-issue images only",
            "score_space": "Logits preserve probability ordering and avoid sigmoid saturation ties",
            "ties": "Average MeanRank; expected completion and fractional Top-1 under random tie-breaking",
            "macro": "Arithmetic mean of exactly six named defects; undefined if any component is undefined"}}
    if sha256(path) != checkpoint_hash or sha256(test_path) != test_hash:
        raise RuntimeError("Model or test data changed during reporting")
    write_json(output / "metrics.json", result)
    operational_rows = []
    for label in LABELS:
        for target in RECALL_TARGETS:
            point = operational["per_label"][label][f"{target:.1f}"]
            if point is not None:
                operational_rows.append({"label": label, **point})
    _csv(output / "operating_points_test.csv",
         ["label", "target_validation_recall", "threshold_logit", "threshold_probability",
          "validation_precision", "validation_recall", "test_precision", "test_recall",
          "test_target_recall_met", "true_positives", "predicted_positives", "positives"],
         operational_rows)
    per_image = []
    for raw, calibrated in zip(raw_details, calibrated_details):
        per_image.append({"image_id": test_names[raw["row_index"]], "true_issues": raw["true_issues"],
                          **{f"raw_{key}": raw[key] for key in ("mean_rank", "completion_rank", "top1_hit")},
                          **{f"calibrated_{key}": calibrated[key]
                             for key in ("mean_rank", "completion_rank", "top1_hit")}})
    _csv(output / "image_ranking.csv",
         ["image_id", "true_issues", "raw_mean_rank", "raw_completion_rank", "raw_top1_hit",
          "calibrated_mean_rank", "calibrated_completion_rank", "calibrated_top1_hit"], per_image)
    with (output / "predictions.jsonl").open("w", encoding="utf-8") as handle:
        for i, image in enumerate(test_names):
            row = {"image_id": image, "labels": dict(zip(LABELS, y[i].astype(int).tolist())),
                   "raw_logits": dict(zip(LABELS, logits[i].tolist())),
                   "raw_probabilities": dict(zip(LABELS, expit(logits[i].astype(np.float64)).tolist())),
                   "calibrated_logits": dict(zip(LABELS, scores[i].tolist())),
                   "probabilities": dict(zip(LABELS, expit(scores[i]).tolist()))}
            handle.write(json.dumps(row, allow_nan=False) + "\n")
    write_tables(output, name, test_curve, operational, rankings)
    if selection is not None:
        write_json(output / "selection.json", selection)
    return result
