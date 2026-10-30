"""Cleaned split CSVs and quality-preserving image preprocessing."""
import csv
from pathlib import Path

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset

from . import LABELS, SPLITS
from .io import image_name


def load_records(path, split):
    """Read final 0/1 targets as-is; no vote thresholding or further filtering."""
    if split not in SPLITS:
        raise ValueError(f"Unknown split: {split}")
    records, seen = [], set()
    with Path(path).open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != ["image_id", *LABELS]:
            raise ValueError(f"{path}: expected CSV header image_id,{','.join(LABELS)}")
        for line, row in enumerate(reader, 2):
            if None in row or any(value is None for value in row.values()):
                raise ValueError(f"{path}, line {line}: malformed CSV row")
            name = image_name(row["image_id"].strip())
            if not name.startswith(f"VizWiz_{split}_"):
                raise ValueError(f"{path}, line {line}: image {name} does not belong to {split}")
            if name in seen:
                raise ValueError(f"{path}: duplicate image {name}")
            values = {label: row[label].strip() for label in LABELS}
            if any(value not in ("0", "1") for value in values.values()):
                raise ValueError(f"{path}, line {line}: all seven labels must be 0 or 1")
            seen.add(name)
            records.append({"image": name, "labels": {k: int(v) for k, v in values.items()}})
    if not records:
        raise ValueError(f"Empty split CSV: {path}")
    return records


def targets(records):
    y = np.asarray([[row["labels"][label] for label in LABELS] for row in records],
                   dtype=np.float32).reshape(-1, len(LABELS))
    # Retain the metric/loader mask interface. Cleaned CSVs have complete labels.
    return y, np.ones_like(y, dtype=bool)


class QualityTransform:
    """Letterbox full images. No crops, rotation, flips, blur, or exposure jitter.

    Do not apply EXIF orientation: it could erase the ROT target. Gray padding
    preserves aspect ratio and the original field of view.
    """
    def __init__(self, size):
        self.size = size

    def __call__(self, image):
        image = image.convert("RGB")
        width, height = image.size
        scale = self.size / max(width, height)
        shape = (max(1, round(width * scale)), max(1, round(height * scale)))
        image = image.resize(shape, Image.Resampling.BILINEAR)
        canvas = Image.new("RGB", (self.size, self.size), (128, 128, 128))
        canvas.paste(image, ((self.size - shape[0]) // 2, (self.size - shape[1]) // 2))
        array = np.asarray(canvas, dtype=np.float32).copy() / 255.0
        tensor = torch.from_numpy(array).permute(2, 0, 1)
        return (tensor - 0.5) / 0.5


class QualityDataset(Dataset):
    def __init__(self, records, image_dir, image_size):
        self.records = records
        self.image_dir = Path(image_dir)
        self.y, self.mask = targets(records)
        self.transform = QualityTransform(image_size)
        missing = [row["image"] for row in records if not (self.image_dir / row["image"]).is_file()]
        if missing:
            raise FileNotFoundError(f"{len(missing)} missing images in {image_dir}; first: {missing[0]}")

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        name = self.records[index]["image"]
        # Corrupt images fail loudly; never silently drop evaluation samples.
        with Image.open(self.image_dir / name) as image:
            x = self.transform(image)
        return x, torch.from_numpy(self.y[index]), torch.from_numpy(self.mask[index]), name
