"""Predict feedback for unlabeled images using frozen validation thresholds."""
from math import isfinite

import numpy as np
from PIL import Image
import torch
from torch.utils.data import DataLoader

from . import LABELS
from .calibration import check_calibration, platt_logits
from .data import QualityTransform
from .engine import get_device, load_checkpoint, predict
from .io import read_json, sha256


def _images(inputs):
    extensions = Image.registered_extensions()
    paths = []
    for path in inputs:
        if path.is_dir():
            children = sorted(child for child in path.iterdir()
                              if child.is_file() and child.suffix.lower() in extensions)
            if not children:
                raise ValueError(f"No supported image files in {path}")
            paths.extend(children)
        elif path.is_file():
            paths.append(path)
        else:
            raise FileNotFoundError(path)
    return list(dict.fromkeys(path.resolve() for path in paths))


class _ExternalImages:
    def __init__(self, paths, image_size):
        self.paths = paths
        self.transform = QualityTransform(image_size)

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        path = self.paths[index]
        with Image.open(path) as image:
            # The prediction loader ignores both target fields.
            return self.transform(image), (), (), str(path)


def _thresholds(checkpoint, checkpoint_path, points_path, calibration_path):
    calibration = None
    if points_path is None:
        point = checkpoint.get("validation_operating_point")
        if point is None:
            raise ValueError("Checkpoint has no embedded validation thresholds. "
                             "Pass --operating-points from a report for this exact checkpoint.")
        if (point.get("fit_split"), point.get("recall_target"), point.get("score_space")) != (
                "val", 0.9, "raw logits"):
            raise ValueError("Expected embedded validation R@0.9 thresholds in raw logits")
        per_label = point["per_label"]
    else:
        points = read_json(points_path)
        checkpoint_hash = sha256(checkpoint_path)
        if (points.get("fit_split") != "val" or points.get("labels") != list(LABELS)
                or points.get("checkpoint_sha256") != checkpoint_hash
                or points.get("protocol") != checkpoint["protocol"]
                or points.get("score_space") != "calibrated logits; compare score >= threshold_logit"):
            raise ValueError("Expected validation operating points from a report for this exact checkpoint")
        calibration_path = calibration_path or points_path.with_name("calibration.json")
        if sha256(calibration_path) != points.get("calibration_sha256"):
            raise ValueError("Calibration file does not match the report operating points")
        calibration = read_json(calibration_path)
        check_calibration(calibration, checkpoint_hash, checkpoint["protocol"])
        per_label = {label: points["per_label"].get(label, {}).get("0.9") for label in LABELS}

    thresholds = []
    for label in LABELS:
        threshold = (per_label.get(label) or {}).get("threshold_logit")
        if type(threshold) not in (int, float) or not isfinite(threshold):
            raise ValueError(f"No finite validation R@0.9 threshold for {label}")
        thresholds.append(threshold)
    return np.asarray(thresholds, dtype=np.float64), calibration


def predict_external(checkpoint_path, inputs, device_name=None, points_path=None,
                     calibration_path=None):
    if calibration_path is not None and points_path is None:
        raise ValueError("--calibration requires --operating-points")
    paths = _images(inputs)
    checkpoint, model = load_checkpoint(checkpoint_path, torch.device("cpu"))
    thresholds, calibration = _thresholds(checkpoint, checkpoint_path, points_path, calibration_path)
    config = checkpoint["config"]
    device = get_device(device_name or config["device"])
    model.to(device)
    loader = DataLoader(_ExternalImages(paths, config["image_size"]),
                        batch_size=config["training"]["batch_size"])
    names, logits = predict(model, loader, device, config["training"]["amp"],
                            return_logits=True, description="External images")
    scores = platt_logits(logits, calibration) if calibration is not None else logits
    decisions = scores >= thresholds
    return [(name, dict(zip(LABELS, row.tolist()))) for name, row in zip(names, decisions)]
