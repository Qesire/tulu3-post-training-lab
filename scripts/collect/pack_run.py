#!/usr/bin/env python3
"""Create a compact, checksummed return packet from one experiment run root.

By default checkpoints are excluded so the packet remains small enough to move
through chat/upload workflows. The packet contains audit/config/log/result/
scheduler evidence plus RETURN_MANIFEST.json with per-file SHA256.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import tarfile
from datetime import datetime, timezone
from pathlib import Path

DEFAULT_DIRS = ("audit", "config", "logs", "results", "scheduler")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def collect_files(root: Path, include_checkpoints: bool) -> list[Path]:
    dirs = list(DEFAULT_DIRS)
    if include_checkpoints:
        dirs.append("checkpoints")
    files: list[Path] = []
    for name in dirs:
        base = root / name
        if not base.exists():
            continue
        for p in sorted(base.rglob("*")):
            if p.is_symlink():
                raise SystemExit(f"RETURN_PACKET_SYMLINK_FORBIDDEN: {p}")
            if p.is_file():
                files.append(p)
    return files


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_root", type=Path)
    ap.add_argument("--output", type=Path, default=None)
    ap.add_argument("--include-checkpoints", action="store_true")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    root = args.run_root.resolve()
    if not root.is_dir():
        raise SystemExit(f"RUN_ROOT_MISSING: {root}")
    files = collect_files(root, args.include_checkpoints)
    if not files:
        raise SystemExit(f"RUN_ROOT_HAS_NO_RETURN_EVIDENCE: {root}")

    output = (
        args.output.resolve()
        if args.output is not None
        else (root.parent / f"{root.name}_RETURN.tar.gz").resolve()
    )
    if output.exists() and not args.overwrite:
        raise SystemExit(f"RETURN_PACKET_EXISTS: {output}; use --overwrite intentionally")
    output.parent.mkdir(parents=True, exist_ok=True)

    entries = []
    for p in files:
        rel = p.relative_to(root).as_posix()
        entries.append({"path": rel, "size_bytes": p.stat().st_size, "sha256": sha256_file(p)})

    manifest = {
        "schema_version": 1,
        "status": "RETURN_PACKET",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "run_root_name": root.name,
        "source_run_root": str(root),
        "include_checkpoints": args.include_checkpoints,
        "file_count": len(entries),
        "files": entries,
    }
    manifest_bytes = (json.dumps(manifest, indent=2, ensure_ascii=False) + "\n").encode("utf-8")

    with tarfile.open(output, "w:gz") as tf:
        prefix = root.name
        for p in files:
            rel = p.relative_to(root).as_posix()
            tf.add(p, arcname=f"{prefix}/{rel}", recursive=False)
        info = tarfile.TarInfo(name=f"{prefix}/RETURN_MANIFEST.json")
        info.size = len(manifest_bytes)
        info.mtime = int(datetime.now().timestamp())
        tf.addfile(info, io.BytesIO(manifest_bytes))

    print(f"RETURN_PACKET={output}")
    print(f"RETURN_PACKET_SHA256={sha256_file(output)}")
    print(f"RETURN_FILE_COUNT={len(entries)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
