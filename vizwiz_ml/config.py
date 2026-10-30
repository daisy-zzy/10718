"""JSON configuration, resolved relative to the config file location."""
from copy import deepcopy
import math
from pathlib import Path

from .io import read_json

DEFAULT_TRAINING = {"epochs": 30, "batch_size": 64, "learning_rate": 0.001,
                    "weight_decay": 0.0001, "num_workers": 4, "amp": True,
                    "gradient_clip": 1.0, "positive_weighting": True}


def load_config(path):
    path = Path(path).resolve()
    config = deepcopy(read_json(path))
    unknown = set(config) - {"data", "training", "seed", "device", "image_size", "policy",
                             "selection", "metrics", "models"}
    if unknown:
        raise ValueError(f"Unknown config fields: {sorted(unknown)}")
    data = config["data"]
    if set(data) != {"labels", "images"}:
        raise ValueError("data requires labels (train.csv/val.csv/test.csv directory) "
                         "and images (train/val/test folder parent)")
    for key in data:
        value = Path(data[key]).expanduser()
        data[key] = str((path.parent / value).resolve())
    config.setdefault("seed", 42)
    config.setdefault("device", "cuda")
    config.setdefault("image_size", 224)
    config.setdefault("policy", {"threshold": 0.5, "max_predictions": 5})
    config.setdefault("selection", {"metric": "classification.macro6_p_at_r90", "mode": "max"})
    config.setdefault("metrics", ["classification", "correction"])
    policy = config["policy"]
    if set(policy) != {"threshold", "max_predictions"}:
        raise ValueError("policy requires threshold and max_predictions")
    if not 0 <= policy["threshold"] <= 1 or type(policy["max_predictions"]) is not int or not 1 <= policy["max_predictions"] <= 5:
        raise ValueError("Use threshold in [0, 1] and max_predictions in [1, 5]")
    if not config["metrics"] or len(set(config["metrics"])) != len(config["metrics"]):
        raise ValueError("metrics must be a nonempty list of unique metric plugin names")
    supplied = config.get("training", {})
    # Accept older config files, but patience no longer controls training.
    unknown_training = set(supplied) - (set(DEFAULT_TRAINING) | {"patience"})
    if unknown_training:
        raise ValueError(f"Unknown training options: {unknown_training}")
    supplied = {key: value for key, value in supplied.items() if key != "patience"}
    config["training"] = {**DEFAULT_TRAINING, **supplied}
    training = config["training"]
    for key in ("epochs", "batch_size"):
        if type(training[key]) is not int or training[key] < 1:
            raise ValueError(f"{key} must be a positive integer")
    if type(training["num_workers"]) is not int or training["num_workers"] < 0:
        raise ValueError("num_workers must be a nonnegative integer")
    if training["learning_rate"] <= 0 or training["weight_decay"] < 0 or training["gradient_clip"] < 0:
        raise ValueError("Invalid optimizer configuration")
    if any(not math.isfinite(training[k]) for k in ("learning_rate", "weight_decay", "gradient_clip")):
        raise ValueError("Optimizer settings must be finite")
    for key in ("amp", "positive_weighting"):
        if type(training[key]) is not bool:
            raise ValueError(f"{key} must be boolean")
    if type(config["seed"]) is not int or not 0 <= config["seed"] < 2 ** 32:
        raise ValueError("seed must be an integer in [0, 2**32)")
    if config["selection"]["mode"] not in ("min", "max"):
        raise ValueError("selection.mode must be min or max")
    selection_metric = config["selection"]["metric"]
    if selection_metric.split(".")[0] not in config["metrics"]:
        raise ValueError("The selection metric's plugin must be enabled in metrics")
    if selection_metric == "classification.macro6_p_at_r90" and config["selection"]["mode"] != "max":
        raise ValueError("classification.macro6_p_at_r90 must be maximized")
    if type(config["image_size"]) is not int or config["image_size"] < 32:
        raise ValueError("image_size must be an integer >= 32")
    models = config["models"]
    if not models or len({model["id"] for model in models}) != len(models):
        raise ValueError("Provide one or more models with unique IDs")
    for model in models:
        identifier = model["id"]
        if not identifier or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for c in identifier):
            raise ValueError("Model IDs may contain only letters, numbers, underscores and hyphens")
        if set(model) - {"id", "name", "kwargs"}:
            raise ValueError("Model spec supports only id, name, kwargs")
    return config
