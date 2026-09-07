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


def load_binding(manifest_path: Path, stage: str) -> dict:
    if not manifest_path.is_file():
        raise SystemExit(f"DATASET_BINDING_MANIFEST_MISSING: {manifest_path}")
    try:
        manifest = json.loads(manifest_path.read_text())
    except Exception as exc:
        raise SystemExit(f"DATASET_BINDING_MANIFEST_INVALID: {manifest_path}: {exc}") from exc
    entry = manifest.get("datasets", {}).get(stage)
    if not isinstance(entry, dict):
        raise SystemExit(f"DATASET_BINDING_STAGE_MISSING: stage={stage} manifest={manifest_path}")
    required = ["hf_id", "resolved_revision", "local_path", "num_rows", "columns", "size_bytes", "sha256"]
    missing = [key for key in required if entry.get(key) in (None, "")]
    if missing:
        raise SystemExit(f"DATASET_BINDING_INCOMPLETE: stage={stage} missing={missing}")
    return entry


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
    ap.add_argument("--binding-manifest", type=Path, default=None)
    ap.add_argument("--binding-stage", choices=["sft", "dpo", "rlvr"], default=None)
    args = ap.parse_args()

    if (args.binding_manifest is None) != (args.binding_stage is None):
        raise SystemExit("DATASET_BINDING_ARGS_INCOMPLETE: --binding-manifest and --binding-stage must be used together")

    path = args.dataset_file.resolve()
    if not path.is_file() or path.suffix != ".parquet":
        raise SystemExit(f"DATASET_ASSET_INVALID: expected local .parquet file: {path}")

    binding = None
    if args.binding_manifest is not None:
        binding = load_binding(args.binding_manifest.resolve(), args.binding_stage)
        bound_path = Path(binding["local_path"]).resolve()
        if bound_path != path:
            raise SystemExit(f"DATASET_BINDING_PATH_MISMATCH: expected={bound_path} actual={path}")
        if int(binding["size_bytes"]) != path.stat().st_size:
            raise SystemExit(
                f"DATASET_BINDING_SIZE_MISMATCH: expected={binding['size_bytes']} actual={path.stat().st_size}"
            )

    digest = sha256(path)
    if binding is not None and digest != binding["sha256"]:
        raise SystemExit(f"DATASET_BINDING_SHA256_MISMATCH: expected={binding['sha256']} actual={digest}")

    ds = load_dataset("parquet", data_files=str(path), split="train")
    total_rows = len(ds)
    columns = list(ds.column_names)

    if binding is not None:
        if total_rows != int(binding["num_rows"]):
            raise SystemExit(f"DATASET_BINDING_ROW_MISMATCH: expected={binding['num_rows']} actual={total_rows}")
        if columns != list(binding["columns"]):
            raise SystemExit(f"DATASET_BINDING_COLUMNS_MISMATCH: expected={binding['columns']} actual={columns}")

    missing = sorted(set(args.require_columns) - set(columns))
    if missing:
        raise SystemExit(f"DATASET_SCHEMA_MISMATCH: missing required columns: {missing}; observed={columns}")

    validated_rows = total_rows
    if args.max_rows is not None:
        ds = ds.select(range(min(args.max_rows, total_rows)))
        validated_rows = len(ds)

    payload = {
        "schema_version": 3,
        "path": str(path),
        "size_bytes": path.stat().st_size,
        "sha256": digest,
        "total_rows": total_rows,
        "validated_rows": validated_rows,
        "columns": columns,
        "required_columns": list(args.require_columns),
        "fingerprint": getattr(ds, "_fingerprint", None),
        "binding": None
        if binding is None
        else {
            "manifest": str(args.binding_manifest.resolve()),
            "stage": args.binding_stage,
            "hf_id": binding["hf_id"],
            "resolved_revision": binding["resolved_revision"],
            "matched": True,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    print(args.output.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
