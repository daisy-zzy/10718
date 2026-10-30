"""Single-device PyTorch training and inference, suitable for an AWS GPU."""
import json
import math
import platform
import random
from pathlib import Path
from time import perf_counter

import numpy as np
import torch
from scipy.special import expit
from torch import nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from . import LABELS
from .calibration import apply_platt, check_calibration, fit_platt
from .data import QualityDataset, load_records, targets
from .io import read_json, sha256, write_json
from .metrics import evaluate_metrics, prediction_lists
from .models import build_model


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def seed_worker(worker_id):
    seed = torch.initial_seed() % (2 ** 32)
    random.seed(seed)
    np.random.seed(seed)


def get_device(name):
    device = torch.device(name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable. Install the matching GPU PyTorch build "
                           "on AWS, or explicitly use device=cpu for development.")
    if device.type not in ("cpu", "cuda"):
        raise ValueError("Supported devices: cpu, cuda, cuda:N")
    return device


def make_loader(records, config, split, device, shuffle=False):
    dataset = QualityDataset(records, Path(config["data"]["images"]) / split, config["image_size"])
    settings = config["training"]
    generator = torch.Generator().manual_seed(config["seed"])
    return DataLoader(dataset, batch_size=settings["batch_size"], shuffle=shuffle,
                      num_workers=settings["num_workers"], pin_memory=device.type == "cuda",
                      persistent_workers=settings["num_workers"] > 0, drop_last=False,
                      generator=generator, worker_init_fn=seed_worker)


def metric_value(result, key):
    value = result
    for part in key.split("."):
        value = value[part]
    if value is None or isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"Selection metric {key} is undefined or not a finite number")
    return float(value)


def atomic_checkpoint(path, checkpoint):
    temporary = path.with_suffix(".tmp")
    torch.save(checkpoint, temporary)
    temporary.replace(path)


def optimizer_step(model, optimizer, scaler, gradient_clip):
    """Let GradScaler skip overflowing AMP steps; keep FP32 clipping strict."""
    if gradient_clip:
        scaler.unscale_(optimizer)
        # unscale_ records inf/NaN gradients. Raising here would prevent
        # scaler.step() from skipping the update and update() lowering the scale.
        nn.utils.clip_grad_norm_(model.parameters(), gradient_clip,
                                 error_if_nonfinite=not scaler.is_enabled())
    previous_scale = scaler.get_scale()
    scaler.step(optimizer)
    scaler.update()
    return scaler.get_scale() >= previous_scale


def protocol_signature(config, train_hash, val_hash):
    # Models may use different image sizes/architectures; compare the same targets,
    # preprocessing rule, output policy and metric on the same validation images.
    return {"labels": list(LABELS), "train_sha256": train_hash, "val_sha256": val_hash,
            "target_format": "cleaned_binary_csv_v1",
            "preprocessing": "rgb_letterbox_gray128_no_exif_v1",
            "policy": config["policy"], "selection": config["selection"],
            "metrics": config["metrics"]}


@torch.inference_mode()
def predict(model, loader, device, amp=False, return_logits=False, description="Inference"):
    model.eval()
    names, scores = [], []
    for images, _, _, image_names in tqdm(
            loader, desc=description, unit="batch", dynamic_ncols=True, disable=None):
        images = images.to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, dtype=torch.float16,
                            enabled=amp and device.type == "cuda"):
            logits = model(images)
        if logits.shape != (len(images), len(LABELS)):
            raise ValueError(f"Models must return a batch x {len(LABELS)} tensor of binary logits")
        if not torch.isfinite(logits).all():
            raise FloatingPointError("Nonfinite model logits during inference")
        scores.append((logits.float() if return_logits else logits.float().sigmoid()).cpu().numpy())
        names.extend(image_names)
    return names, np.concatenate(scores)


def train_model(config, spec, output):
    """Never opens test labels, test images or a test loader."""
    seed_everything(config["seed"])
    device = get_device(config["device"])
    settings = config["training"]
    labels_dir = Path(config["data"]["labels"])
    train_records = load_records(labels_dir / "train.csv", "train")
    val_records = load_records(labels_dir / "val.csv", "val")
    if {r["image"] for r in train_records} & {r["image"] for r in val_records}:
        raise ValueError("Training and validation image IDs overlap")
    y, mask = targets(train_records)
    val_y, val_mask = targets(val_records)
    train_hash, val_hash = sha256(labels_dir / "train.csv"), sha256(labels_dir / "val.csv")
    train_loader = make_loader(train_records, config, "train", device, shuffle=True)
    val_loader = make_loader(val_records, config, "val", device)
    model = build_model(spec).to(device)
    positives = (y * mask).sum(axis=0)
    negatives = mask.sum(axis=0) - positives
    weights = np.ones(len(LABELS), dtype=np.float32)
    if settings["positive_weighting"]:
        # No-positive/no-negative heads use unit weight; avoid zero gradients/inf.
        np.divide(negatives, positives, out=weights, where=(positives > 0) & (negatives > 0))
    criterion = nn.BCEWithLogitsLoss(reduction="none", pos_weight=torch.tensor(weights, device=device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=settings["learning_rate"],
                                  weight_decay=settings["weight_decay"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=settings["epochs"])
    amp = settings["amp"] and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "config.json", {**config, "model": spec})
    environment = {"python": platform.python_version(), "torch": str(torch.__version__),
                   "numpy": np.__version__, "cuda": torch.version.cuda, "device": str(device),
                   "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None}
    write_json(output / "environment.json", environment)
    signature = protocol_signature(config, train_hash, val_hash)
    best_value, best_epoch, best_metrics = None, None, None
    for epoch in range(1, settings["epochs"] + 1):
        epoch_started = perf_counter()
        model.train()
        loss_sum, observed = 0.0, 0
        optimizer_steps, amp_skipped_steps = 0, 0
        learning_rate = optimizer.param_groups[0]["lr"]
        train_started = perf_counter()
        progress = tqdm(train_loader,
                        desc=f"{spec['id']} epoch {epoch}/{settings['epochs']} train",
                        unit="batch", dynamic_ncols=True, disable=None)
        for images, truth, known, _ in progress:
            images = images.to(device, non_blocking=True)
            truth, known = truth.to(device), known.to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp):
                logits = model(images)
                if logits.shape != truth.shape:
                    raise ValueError(f"Models must return a batch x {len(LABELS)} tensor")
                losses = criterion(logits.float(), truth)
                loss = (losses * known).sum() / known.sum().clamp_min(1)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Nonfinite training loss at epoch {epoch}")
            scaler.scale(loss).backward()
            updated = optimizer_step(model, optimizer, scaler, settings["gradient_clip"])
            optimizer_steps += int(updated)
            amp_skipped_steps += int(not updated)
            count = int(known.sum().item())
            observed += count
            loss_sum += loss.item() * count
            progress.set_postfix(loss=f"{loss_sum / observed:.5f}",
                                 amp_skipped=amp_skipped_steps, refresh=False)
        train_seconds = perf_counter() - train_started
        if not optimizer_steps:
            raise FloatingPointError(f"No optimizer updates succeeded in epoch {epoch}; "
                                     "AMP overflow persisted across every batch. "
                                     "Set training.amp=false to diagnose in full precision.")
        validation_started = perf_counter()
        _, val_logits = predict(model, val_loader, device, amp, return_logits=True,
                            description=f"{spec['id']} epoch {epoch}/{settings['epochs']} val")
        scores = expit(val_logits.astype(np.float64))
        metrics = evaluate_metrics(val_y, scores, val_mask, config["policy"], config["metrics"],
                                   selection_scores=val_logits)
        validation_seconds = perf_counter() - validation_started
        value = metric_value(metrics, config["selection"]["metric"])
        improved = best_value is None or (value > best_value if config["selection"]["mode"] == "max" else value < best_value)
        if improved:
            best_value, best_epoch, best_metrics = value, epoch, metrics
        scheduler.step()
        checkpoint = {"format_version": 2, "labels": list(LABELS), "model_spec": spec,
                      "config": config, "protocol": signature, "epoch": epoch,
                      "model_state": model.state_dict(), "optimizer_state": optimizer.state_dict(),
                      "scheduler_state": scheduler.state_dict(), "scaler_state": scaler.state_dict(),
                      "validation_metrics": metrics, "positive_weights": weights.tolist(),
                      "best_epoch": best_epoch, "best_value": best_value}
        atomic_checkpoint(output / "last.pt", checkpoint)
        if improved:
            atomic_checkpoint(output / "best.pt", checkpoint)
            write_json(output / "best_validation.json", metrics)
        log = {"epoch": epoch, "training_loss": loss_sum / observed,
               "learning_rate": learning_rate, "selection_value": value, "validation": metrics,
               "optimizer_steps": optimizer_steps, "amp_skipped_steps": amp_skipped_steps,
               "amp_scale": scaler.get_scale(), "train_seconds": train_seconds,
               "validation_seconds": validation_seconds,
               "epoch_seconds": perf_counter() - epoch_started}
        with (output / "history.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(log, allow_nan=False) + "\n")
        tqdm.write(f"{spec['id']} epoch={epoch} loss={log['training_loss']:.5f} "
              f"val_{config['selection']['metric']}={value:.5f} "
              f"updates={optimizer_steps} amp_skipped={amp_skipped_steps} "
              f"train={train_seconds:.1f}s val={validation_seconds:.1f}s "
              f"epoch={log['epoch_seconds']:.1f}s")
    # Detect concurrent edits to cleaned CSVs before publishing this run.
    if sha256(labels_dir / "train.csv") != train_hash or sha256(labels_dir / "val.csv") != val_hash:
        raise RuntimeError("Label CSVs changed during training")
    summary = {"model_id": spec["id"], "checkpoint": str(output / "best.pt"),
               "checkpoint_sha256": sha256(output / "best.pt"), "best_epoch": best_epoch,
               "selection_value": best_value, "protocol": signature,
               "validation_metrics": best_metrics, "training_images_total": len(train_records),
               "training_images_supervised": len(train_records), "epochs_completed": epoch}
    write_json(output / "summary.json", summary)
    return summary


def load_checkpoint(path, device):
    # weights_only supports tensors and primitive metadata. Never unpickle arbitrary objects.
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if checkpoint["format_version"] != 2 or checkpoint["labels"] != list(LABELS):
        raise ValueError("Expected a seven-label cleaned-CSV checkpoint (format 2). "
                         "Old six-label checkpoints need retraining.")
    model = build_model(checkpoint["model_spec"])
    model.load_state_dict(checkpoint["model_state"], strict=True)
    return checkpoint, model.to(device)


def calibrate_checkpoint(path, output, device_name=None, selection=None, l2=1e-4):
    """Fit on validation images only; never modify the model or open test data."""
    output = Path(output)
    if output.exists():
        raise FileExistsError(f"Calibration already exists: {output}")
    checkpoint, model = load_checkpoint(path, torch.device("cpu"))
    checkpoint_hash = sha256(path)
    if selection is not None:
        if checkpoint["protocol"] != selection["protocol"] or checkpoint_hash != selection["winner"]["checkpoint_sha256"]:
            raise ValueError("Checkpoint is not the frozen validation winner")
    config = checkpoint["config"]
    labels_dir = Path(config["data"]["labels"])
    for source in ("train", "val"):
        if sha256(labels_dir / f"{source}.csv") != checkpoint["protocol"][f"{source}_sha256"]:
            raise ValueError(f"{source}.csv changed since training")
    records = load_records(labels_dir / "val.csv", "val")
    device = get_device(device_name or config["device"])
    model.to(device)
    loader = make_loader(records, config, "val", device)
    names, logits = predict(model, loader, device, config["training"]["amp"], return_logits=True)
    if names != [row["image"] for row in records]:
        raise RuntimeError("Calibration prediction order differs from val.csv")
    y, _ = targets(records)
    calibration = fit_platt(logits, y, l2=l2)
    calibration.update({"fit_split": "val", "checkpoint_sha256": checkpoint_hash,
                        "checkpoint": str(Path(path).resolve()), "protocol": checkpoint["protocol"],
                        "validation_csv_sha256": checkpoint["protocol"]["val_sha256"],
                        "fit_diagnostics": "NLL values are measured on the same validation images used for fitting"})
    if sha256(path) != checkpoint_hash or sha256(labels_dir / "val.csv") != checkpoint["protocol"]["val_sha256"]:
        raise RuntimeError("Checkpoint or validation CSV changed while fitting calibration")
    output.parent.mkdir(parents=True, exist_ok=True)
    write_json(output, calibration)
    return calibration


def evaluate_checkpoint(path, split, output, device_name=None, selection=None, calibration_path=None):
    if split not in ("val", "test"):
        raise ValueError("Evaluation split must be val or test")
    if split == "test" and selection is None:
        raise ValueError("Testing requires a frozen validation selection")
    # Load on CPU first so the requested runtime device can override training CUDA IDs.
    checkpoint, model = load_checkpoint(path, torch.device("cpu"))
    config = checkpoint["config"]
    if selection is not None and checkpoint["protocol"] != selection["protocol"]:
        raise ValueError("Selected protocol does not match checkpoint")
    if selection is not None and sha256(path) != selection["winner"]["checkpoint_sha256"]:
        raise ValueError("Checkpoint is not the frozen validation winner")
    calibration = read_json(calibration_path) if calibration_path is not None else None
    if calibration is not None:
        check_calibration(calibration, sha256(path), checkpoint["protocol"])
    device = get_device(device_name or config["device"])
    model.to(device)
    labels_dir = Path(config["data"]["labels"])
    for source in ("train", "val"):
        if sha256(labels_dir / f"{source}.csv") != checkpoint["protocol"][f"{source}_sha256"]:
            raise ValueError(f"{source}.csv changed since training")
    records = load_records(labels_dir / f"{split}.csv", split)
    if split == "test":
        test_names = {row["image"] for row in records}
        for source in ("train", "val"):
            source_names = {row["image"] for row in load_records(labels_dir / f"{source}.csv", source)}
            if test_names & source_names:
                raise ValueError(f"Test images overlap {source}")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    loader = make_loader(records, config, split, device)
    names, logits = predict(model, loader, device, config["training"]["amp"], return_logits=True)
    raw_probabilities = expit(logits.astype(np.float64))
    probabilities = apply_platt(logits, calibration) if calibration is not None else raw_probabilities
    if names != [row["image"] for row in records]:
        raise RuntimeError("Prediction coverage/order differs from split CSV")
    y, mask = targets(records)
    # Positive-slope Platt scaling preserves each label's PR curve, so original
    # logits also provide the exact ordering for calibrated per-label P@R.
    metrics = evaluate_metrics(y, probabilities, mask, config["policy"], config["metrics"],
                               selection_scores=logits)
    metrics.update({"split": split, "checkpoint_sha256": sha256(path),
                    "labels_csv_sha256": sha256(labels_dir / f"{split}.csv"),
                    "labels": list(LABELS), "epoch": checkpoint["epoch"],
                    "score_type": "platt_calibrated" if calibration is not None else "raw_sigmoid"})
    if calibration is not None:
        metrics["calibration_sha256"] = sha256(calibration_path)
        metrics["raw_metrics"] = evaluate_metrics(
            y, raw_probabilities, mask, config["policy"], config["metrics"], selection_scores=logits)
    lists = prediction_lists(probabilities, **config["policy"])
    with (output / "predictions.jsonl").open("w", encoding="utf-8") as handle:
        for name, prob, raw_prob, indices in zip(names, probabilities, raw_probabilities, lists):
            record = {"image_id": name, "probabilities": dict(zip(LABELS, prob.tolist())),
                      "predicted_issues": [LABELS[j] for j in indices]}
            if calibration is not None:
                record["raw_probabilities"] = dict(zip(LABELS, raw_prob.tolist()))
            handle.write(json.dumps(record, allow_nan=False) + "\n")
    write_json(output / "metrics.json", metrics)
    if selection is not None:
        write_json(output / "selection.json", selection)
    if calibration is not None:
        write_json(output / "calibration.json", calibration)
    return metrics
