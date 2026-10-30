"""Small, strict serialization helpers. No networking is used by this package."""
import hashlib
import json
from pathlib import Path


def read_json(path):
    with Path(path).open(encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, allow_nan=False)
        handle.write("\n")
    temporary.replace(path)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def image_name(value):
    if not isinstance(value, str) or not value or Path(value).name != value or "\\" in value:
        raise ValueError(f"Expected an image basename, got {value!r}")
    if value in (".", ".."):
        raise ValueError(f"Invalid image basename: {value!r}")
    return value
