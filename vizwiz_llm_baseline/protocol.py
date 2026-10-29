"""Shared storage paths and label order."""
import os
from pathlib import Path

ROOT = Path(os.environ.get("VIZWIZ_ROOT", Path(__file__).resolve().parent / "data")).expanduser().resolve()
DATA = ROOT / "datasets/vizwiz_quality"
LABELS = ["UNREC", "NON", "BLR", "BRT", "DRK", "OBS", "FRM", "ROT", "OTH"]
