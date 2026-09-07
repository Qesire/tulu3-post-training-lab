#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

REQUIRED = ("config.json",)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("model_dir", type=Path)
    ap.add_argument("--output", type=Path, default=Path("model_manifest.json"))
    args = ap.parse_args()

    root = args.model_dir.resolve()
    if not root.is_dir():
        raise SystemExit(f"MODEL_ASSET_MISSING: {root}")
    for name in REQUIRED:
        if not (root / name).is_file():
            raise SystemExit(f"MODEL_ASSET_INCOMPLETE: missing {root / name}")

    weight_files = sorted(
        p for p in root.iterdir() if p.is_file() and p.suffix in {".safetensors", ".bin"}
    )
    if not weight_files:
        raise SystemExit(f"MODEL_ASSET_INCOMPLETE: no weight files under {root}")

    payload = {
        "schema_version": 1,
        "model_dir": str(root),
        "config_sha256": sha256(root / "config.json"),
        "weight_files": [
            {"name": p.name, "size": p.stat().st_size, "sha256": sha256(p)} for p in weight_files
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(args.output.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
