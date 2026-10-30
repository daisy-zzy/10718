"""Legacy raw-worker to JSON preparation, unused by the cleaned-CSV pipeline.

Current training reads train.csv/val.csv/test.csv directly. Their labels are
already final; do not use this legacy accepted-worker preprocessing for them.
"""
import argparse
import csv
from collections import Counter
from pathlib import Path

from . import LABELS, OFFICIAL_SIZES, SPLITS
from .io import image_name, read_json, sha256, write_json


def prepare(csv_path, manifests, output, mapping_path=None, allow_nonstandard_sizes=False):
    """CSV columns: IMG, WORKERID, REJECT, SPLIT and the six binary issues.

    Aggregated official flaw JSONs are used ONLY for split IDs. They cannot
    establish which workers were accepted, so their flaw counts are not used.
    """
    mapping = read_json(mapping_path) if mapping_path else {}
    mapping = mapping.get("mapping", mapping)  # Also accepts the baseline mapping audit.
    if not isinstance(mapping, dict):
        raise ValueError("Filename mapping must be a JSON object")
    for old, new in mapping.items():
        image_name(old)
        image_name(new)
    if len(set(mapping.values())) != len(mapping):
        raise ValueError("Filename mapping must be one-to-one")

    names, membership = {}, {}
    for split in SPLITS:
        rows = read_json(manifests[split])
        if not isinstance(rows, list):
            raise ValueError(f"{split}: official manifest must be a JSON list")
        names[split] = [image_name(row["image"]) for row in rows]
        if not allow_nonstandard_sizes and len(rows) != OFFICIAL_SIZES[split]:
            raise ValueError(f"{split}: expected {OFFICIAL_SIZES[split]} official images")
        if not rows:
            raise ValueError(f"Empty {split} manifest")
        for name in names[split]:
            if name in membership:
                raise ValueError(f"Duplicate or overlapping split image: {name}")
            membership[name] = split

    votes, accepted, seen = {}, Counter(), set()
    matched_sources = {}
    split_aliases = {"TRAIN": "train", "VAL": "val", "VALIDATION": "val", "TEST": "test"}
    with Path(csv_path).open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"IMG", "WORKERID", "REJECT", "SPLIT", *LABELS}
        if not required <= set(reader.fieldnames or []):
            raise ValueError(f"Worker CSV needs columns {sorted(required)}")
        for line, row in enumerate(reader, 2):
            old = image_name(row["IMG"].strip())
            name = mapping.get(old, old)
            if name not in membership:
                raise ValueError(f"CSV line {line}: {old} is absent from official manifests. "
                                 "Supply a verified filename mapping; names are never guessed.")
            previous_source = matched_sources.setdefault(name, old)
            if previous_source != old:
                raise ValueError(f"Multiple source images map to {name}")
            split = split_aliases.get(row["SPLIT"].strip().upper())
            if split != membership[name]:
                raise ValueError(f"CSV line {line}: split mismatch for {name}")
            worker = row["WORKERID"].strip()
            if not worker or (name, worker) in seen:
                raise ValueError(f"CSV line {line}: missing or duplicate worker")
            seen.add((name, worker))
            rejected = row["REJECT"].strip()
            if rejected not in ("0", "1"):
                raise ValueError(f"CSV line {line}: REJECT must be 0 or 1")
            votes.setdefault(name, {label: 0 for label in LABELS})
            if rejected == "1":
                continue
            for label in LABELS:
                value = row[label].strip()
                if value not in ("0", "1"):
                    raise ValueError(f"CSV line {line}: {label} must be 0 or 1")
                votes[name][label] += int(value)
            accepted[name] += 1

    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    audit = {"label_rule": "At least two accepted worker votes; REJECT=0 only",
             "csv_sha256": sha256(csv_path), "splits": {},
             "mapping_sha256": sha256(mapping_path) if mapping_path else None}
    for split in SPLITS:
        records = [{"image": name, "votes": votes.get(name),
                    "accepted_workers": accepted[name] if name in votes else None}
                   for name in names[split]]
        write_json(output / f"{split}.json", {"split": split, "labels": LABELS,
                                              "records": records})
        audit["splits"][split] = {
            "official_manifest_sha256": sha256(manifests[split]), "images": len(records),
            "missing_annotation_images": [r["image"] for r in records if r["votes"] is None],
            "accepted_worker_histogram": dict(Counter(str(r["accepted_workers"]) for r in records)),
            "positives": {label: sum(r["votes"] is not None and r["votes"][label] >= 2
                                     for r in records) for label in LABELS}}
    write_json(output / "audit.json", audit)
    return audit


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker-csv", required=True, type=Path)
    for split in SPLITS:
        parser.add_argument(f"--{split}-manifest", required=True, type=Path)
    parser.add_argument("--filename-mapping", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--allow-nonstandard-sizes", action="store_true",
                        help="For small fixtures or intentionally different official releases only")
    args = parser.parse_args()
    prepare(args.worker_csv, {s: getattr(args, f"{s}_manifest") for s in SPLITS}, args.output,
            args.filename_mapping, args.allow_nonstandard_sizes)


if __name__ == "__main__":
    main()
