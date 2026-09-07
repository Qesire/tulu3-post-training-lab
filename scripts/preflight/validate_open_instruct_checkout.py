#!/usr/bin/env python3
"""Fail-closed attestation for the pinned Lecture-6 Open-Instruct checkout.

The project intentionally applies one explicit compatibility patch to
``open_instruct/finetune.py``. Therefore a generic ``git status == clean`` gate
is incorrect: the expected checkout is pinned *and deterministically patched*.
This validator proves that the only source-tree deviations are exactly the
recorded patch and its pristine backup.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import subprocess
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PATCH_SCRIPT = PROJECT_ROOT / "scripts" / "setup" / "apply_lecture6_open_instruct_patch.py"
PATCH_TARGET = Path("open_instruct/finetune.py")
PATCH_BACKUP = Path("open_instruct/finetune.py.lecture6.orig")


def run_git(root: Path, *args: str) -> str:
    proc = subprocess.run(
        ["git", "-C", str(root), *args],
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"git {' '.join(args)} failed rc={proc.returncode}: {proc.stderr.strip()}")
    return proc.stdout


def load_patch_text():
    spec = importlib.util.spec_from_file_location("lecture6_patch", PATCH_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import patch authority: {PATCH_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.patch_text


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("open_instruct_root", type=Path)
    ap.add_argument(
        "--expected-commit",
        default="8fcf9c6bae3e3a58e1fcb8c79c3bbfbd97065377",
    )
    ap.add_argument("--output", type=Path, default=Path("open_instruct_checkout.json"))
    args = ap.parse_args()

    root = args.open_instruct_root.resolve()
    if not (root / ".git").exists():
        raise SystemExit(f"OPEN_INSTRUCT_GIT_CHECKOUT_MISSING: {root}")

    actual_commit = run_git(root, "rev-parse", "HEAD").strip()
    if actual_commit != args.expected_commit:
        raise SystemExit(
            f"OPEN_INSTRUCT_COMMIT_MISMATCH: expected={args.expected_commit} actual={actual_commit}"
        )

    target = root / PATCH_TARGET
    backup = root / PATCH_BACKUP
    if not target.is_file():
        raise SystemExit(f"OPEN_INSTRUCT_PATCH_TARGET_MISSING: {target}")
    if not backup.is_file():
        raise SystemExit(f"OPEN_INSTRUCT_PATCH_BACKUP_MISSING: {backup}")

    pristine = run_git(root, "show", f"HEAD:{PATCH_TARGET.as_posix()}")
    if backup.read_text() != pristine:
        raise SystemExit("OPEN_INSTRUCT_PATCH_BACKUP_MISMATCH: backup is not pristine HEAD content")

    patch_text = load_patch_text()
    expected_patched, patch_state = patch_text(pristine)
    if patch_state != "patched":
        raise SystemExit(
            f"OPEN_INSTRUCT_PATCH_AUTHORITY_UNEXPECTED: pristine HEAD returned state={patch_state}"
        )
    if target.read_text() != expected_patched:
        raise SystemExit("OPEN_INSTRUCT_PATCH_CONTENT_MISMATCH: finetune.py is not the deterministic Lecture-6 patch")

    status_lines = [line for line in run_git(root, "status", "--porcelain=v1", "--untracked-files=all").splitlines() if line]
    allowed = {
        f" M {PATCH_TARGET.as_posix()}",
        f"?? {PATCH_BACKUP.as_posix()}",
    }
    unexpected = sorted(set(status_lines) - allowed)
    missing = sorted(allowed - set(status_lines))
    if unexpected or missing:
        raise SystemExit(
            "OPEN_INSTRUCT_WORKTREE_ATTESTATION_FAILED: "
            f"unexpected={unexpected} missing_expected={missing} observed={status_lines}"
        )

    payload = {
        "schema_version": 1,
        "status": "PASS",
        "root": str(root),
        "expected_commit": args.expected_commit,
        "actual_commit": actual_commit,
        "patch_target": PATCH_TARGET.as_posix(),
        "patch_backup": PATCH_BACKUP.as_posix(),
        "patch_state_from_pristine": patch_state,
        "expected_worktree_status": sorted(allowed),
        "observed_worktree_status": status_lines,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    print(args.output.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
