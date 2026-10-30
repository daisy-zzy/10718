"""Train candidates, freeze validation selection, then evaluate only the winner."""
import argparse
from pathlib import Path

from .config import load_config
from .engine import calibrate_checkpoint, evaluate_checkpoint, metric_value, train_model
from .io import read_json, sha256, write_json
from .report import report_checkpoint


def select_best(run_dirs, output):
    summaries = [read_json(Path(run) / "summary.json") for run in run_dirs]
    if not summaries:
        raise ValueError("No candidates provided")
    protocol = summaries[0]["protocol"]
    if any(row["protocol"] != protocol for row in summaries):
        raise ValueError("Candidates have different datasets, policies or selection metrics")
    candidates = []
    seen = set()
    for row in summaries:
        if row["model_id"] in seen:
            raise ValueError("Candidate model IDs must be unique")
        seen.add(row["model_id"])
        if sha256(row["checkpoint"]) != row["checkpoint_sha256"]:
            raise ValueError(f"Checkpoint changed: {row['checkpoint']}")
        value = metric_value(row["validation_metrics"], protocol["selection"]["metric"])
        if value != row["selection_value"]:
            raise ValueError("Stored selection value disagrees with validation metrics")
        candidates.append({key: row[key] for key in
                           ("model_id", "checkpoint", "checkpoint_sha256", "best_epoch", "selection_value")})
    # Stable sort: exact ties go to the first explicitly listed candidate.
    candidates.sort(key=lambda row: row["selection_value"], reverse=protocol["selection"]["mode"] == "max")
    result = {"selection_source": "validation_only", "protocol": protocol,
              "winner": candidates[0], "candidates": candidates,
              "tie_rule": "first candidate in supplied run order"}
    output = Path(output)
    if output.exists():
        raise FileExistsError(f"Selection already exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    write_json(output, result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    train = commands.add_parser("train")
    train.add_argument("--config", required=True, type=Path)
    train.add_argument("--model", required=True, help="Model ID from config")
    train.add_argument("--output", required=True, type=Path)
    experiment = commands.add_parser("experiment", help="Train all configured models and select on validation")
    experiment.add_argument("--config", required=True, type=Path)
    experiment.add_argument("--output", required=True, type=Path)
    select = commands.add_parser("select")
    select.add_argument("--runs", required=True, nargs="+", type=Path)
    select.add_argument("--output", required=True, type=Path)
    validate = commands.add_parser("validate")
    validate.add_argument("--checkpoint", required=True, type=Path)
    validate.add_argument("--output", required=True, type=Path)
    validate.add_argument("--device")
    validate.add_argument("--calibration", type=Path, help="Optional validation-fitted Platt JSON")
    calibrate = commands.add_parser("calibrate", help="Fit per-label Platt scaling on validation only")
    source = calibrate.add_mutually_exclusive_group(required=True)
    source.add_argument("--checkpoint", type=Path)
    source.add_argument("--selection", type=Path, help="Use the frozen validation-selected winner")
    calibrate.add_argument("--output", required=True, type=Path, help="New calibration JSON filename")
    calibrate.add_argument("--device")
    calibrate.add_argument("--l2", type=float, default=1e-4)
    test = commands.add_parser("test")
    test.add_argument("--selection", required=True, type=Path)
    test.add_argument("--output", required=True, type=Path)
    test.add_argument("--device")
    test.add_argument("--calibration", type=Path, help="Optional validation-fitted Platt JSON")
    report = commands.add_parser("report", help="Revised P@R and tie-aware ranking tables from a trained model")
    source = report.add_mutually_exclusive_group(required=True)
    source.add_argument("--checkpoint", type=Path, help="Existing best.pt; no selected.json required")
    source.add_argument("--selection", type=Path, help="Alternatively use the validation-selected winner")
    report.add_argument("--output", required=True, type=Path, help="New report directory")
    report.add_argument("--device")
    report.add_argument("--calibration", type=Path, help="Reuse an existing validation-fitted Platt JSON")
    report.add_argument("--model-name", help="Display name in CSV/LaTeX tables")
    report.add_argument("--l2", type=float, default=1e-4, help="Regularization when fitting a new calibration")
    args = parser.parse_args()
    if args.command in ("train", "experiment"):
        config = load_config(args.config)
        if args.command == "train":
            matches = [spec for spec in config["models"] if spec["id"] == args.model]
            if not matches:
                parser.error(f"Unknown configured model ID: {args.model}")
            train_model(config, matches[0], args.output)
        else:
            args.output.mkdir(parents=True, exist_ok=False)
            runs = []
            for spec in config["models"]:
                run = args.output / spec["id"]
                train_model(config, spec, run)
                runs.append(run)
            select_best(runs, args.output / "selected.json")
    elif args.command == "select":
        select_best(args.runs, args.output)
    elif args.command == "validate":
        evaluate_checkpoint(args.checkpoint, "val", args.output, args.device,
                            calibration_path=args.calibration)
    elif args.command == "calibrate":
        selection = read_json(args.selection) if args.selection is not None else None
        if selection is not None and selection["selection_source"] != "validation_only":
            raise ValueError("Calibration selection must come from validation")
        checkpoint = selection["winner"]["checkpoint"] if selection is not None else args.checkpoint
        calibrate_checkpoint(checkpoint, args.output, args.device, selection, l2=args.l2)
        print(f"Saved validation-fitted Platt calibration to {args.output}")
    elif args.command == "test":
        selection = read_json(args.selection)
        if selection["selection_source"] != "validation_only":
            raise ValueError("Test requires a validation-only selection")
        winner = selection["winner"]
        if sha256(winner["checkpoint"]) != winner["checkpoint_sha256"]:
            raise ValueError("Selected checkpoint changed after validation selection")
        evaluate_checkpoint(winner["checkpoint"], "test", args.output, args.device, selection,
                            calibration_path=args.calibration)
    elif args.command == "report":
        selection = read_json(args.selection) if args.selection is not None else None
        if selection is not None and selection["selection_source"] != "validation_only":
            raise ValueError("Report selection must come from validation")
        checkpoint = selection["winner"]["checkpoint"] if selection is not None else args.checkpoint
        result = report_checkpoint(checkpoint, args.output, args.device, selection,
                                   calibration_path=args.calibration, model_name=args.model_name, l2=args.l2)
        print(f"Saved revised metrics and tables for {result['test_images']} test images to {args.output}")


if __name__ == "__main__":
    main()
