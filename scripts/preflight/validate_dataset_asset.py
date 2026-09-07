#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from datasets import load_dataset


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset_file", type=Path)
    ap.add_argument("--output", type=Path, default=Path("dataset_asset.json"))
    ap.add_argument("--max-rows", type=int, default=None, help="Only affects validation load, not the file digest")
    ap.add_argument(
        "--require-columns",
        nargs="*",
        default=[],
        help="Fail closed unless every named column is present in the materialized dataset.",
    )
    args = ap.parse_args()

    path = args.dataset_file.resolve()
    if not path.is_file() or path.suffix != ".parquet":
        raise SystemExit(f"DATASET_ASSET_INVALID: expected local .parquet file: {path}")

    ds = load_dataset("parquet", data_files=str(path), split="train")
    columns = list(ds.column_names)
    missing = sorted(set(args.require_columns) - set(columns))
    if missing:
        raise SystemExit(f"DATASET_SCHEMA_MISMATCH: missing required columns: {missing}; observed={columns}")

    if args.max_rows is not None:
        ds = ds.select(range(min(args.max_rows, len(ds))))
    payload = {
        "schema_version": 2,
        "path": str(path),
        "size_bytes": path.stat().st_size,
        "sha256": sha256(path),
        "validated_rows": len(ds),
        "columns": columns,
        "required_columns": list(args.require_columns),
        "fingerprint": getattr(ds, "_fingerprint", None),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    print(args.output.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
