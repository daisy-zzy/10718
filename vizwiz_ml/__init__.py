"""Supervised seven-issue VizWiz training from cleaned image-level CSVs."""

LABELS = ("BLR", "BRT", "DRK", "FRM", "OBS", "ROT", "OTH")
SPLITS = ("train", "val", "test")
OFFICIAL_SIZES = {"train": 23431, "val": 7750, "test": 8000}
