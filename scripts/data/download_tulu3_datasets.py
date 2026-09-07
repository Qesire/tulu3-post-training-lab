#!/usr/bin/env python3
"""Materialize Lecture-6 datasets as immutable local Parquet files.

Why Parquet instead of datasets.save_to_disk(): the pinned lecture-era
Open-Instruct recognizes local .jsonl/.parquet paths directly. A save_to_disk
Arrow directory is not a valid dataset_mixer_list local binding for that code.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

from datasets import load_dataset
from huggingface_hub import HfApi

DATASETS = {
    "sft": "allenai/tulu-3-sft-mixture",
    "dpo": "allenai/llama-3.1-tulu-3-8b-preference-mixture",
    "rlvr": "allenai/RLVR-GSM-MATH-IF-Mixed-Constraints",
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load_existing_manifest(path: Path, split: str) -> dict:
    if not path.exists():
        return {
            "schema_version": 3,
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "split": split,
            "datasets": {},
        }
    payload = json.loads(path.read_text())
    if not isinstance(payload, dict) or not isinstance(payload.get("datasets"), dict):
        raise SystemExit(f"MANIFEST_INVALID: {path}")
    existing_split = payload.get("split")
    if existing_split not in (None, split):
        raise SystemExit(f"MANIFEST_SPLIT_MISMATCH: existing={existing_split} requested={split}")
    payload["schema_version"] = 3
    payload["split"] = split
    return payload


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--split", default="train")
    ap.add_argument("--revision", default="main")
    ap.add_argument("--stage", choices=["all", *DATASETS], default="all")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    stages = DATASETS if args.stage == "all" else {args.stage: DATASETS[args.stage]}
    args.root.mkdir(parents=True, exist_ok=True)
    api = HfApi()
    mf = args.root / "dataset_manifest.json"
    manifest = load_existing_manifest(mf, args.split)

    for stage, hf_id in stages.items():
        info = api.dataset_info(hf_id, revision=args.revision)
        resolved_revision = info.sha
        out = args.root / f"{stage}.parquet"
        if out.exists() and not args.overwrite:
            raise SystemExit(f"OUTPUT_EXISTS: {out}; use --overwrite intentionally")
        print(f"[{stage}] {hf_id}@{resolved_revision}")
        ds = load_dataset(hf_id, split=args.split, revision=resolved_revision)
        ds.to_parquet(str(out))
        manifest["datasets"][stage] = {
            "hf_id": hf_id,
            "requested_revision": args.revision,
            "resolved_revision": resolved_revision,
            "split": args.split,
            "local_path": str(out.resolve()),
            "num_rows": len(ds),
            "columns": list(ds.column_names),
            "fingerprint": getattr(ds, "_fingerprint", None),
            "size_bytes": out.stat().st_size,
            "sha256": sha256(out),
        }
        print(f"[{stage}] rows={len(ds):,} sha256={manifest['datasets'][stage]['sha256']}")

    manifest["timestamp_utc"] = datetime.now(timezone.utc).isoformat()
    tmp = mf.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    tmp.replace(mf)
    print(mf.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
