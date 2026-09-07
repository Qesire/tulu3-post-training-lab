#!/usr/bin/env python3
"""Apply the Lecture-6 W&B compatibility patch to pinned Open-Instruct.

Fail-closed: the patch only applies to the expected three get_url() call sites and
one tracker initialization site. Re-running is idempotent.
"""
from __future__ import annotations

import argparse
from pathlib import Path

OLD_URL = "wandb_tracker.run.get_url()"
NEW_URL = "wandb_tracker.tracker.url"
OLD_INIT = '''        wandb_tracker = accelerator.get_tracker("wandb")\n        maybe_update_beaker_description(wandb_url=wandb_tracker.tracker.url)'''
NEW_INIT = '''        if accelerator.is_main_process:\n            wandb_tracker = accelerator.get_tracker("wandb")\n            maybe_update_beaker_description(wandb_url=wandb_tracker.tracker.url)\n        else:\n            wandb_tracker = None'''


def patch_text(text: str) -> tuple[str, str]:
    if NEW_INIT in text and OLD_URL not in text:
        return text, "already_patched"

    count = text.count(OLD_URL)
    if count != 3:
        raise RuntimeError(f"expected exactly 3 {OLD_URL!r} occurrences, found {count}")
    text = text.replace(OLD_URL, NEW_URL)

    init_count = text.count(OLD_INIT)
    if init_count != 1:
        raise RuntimeError(f"expected exactly 1 tracker-init site after URL replacement, found {init_count}")
    text = text.replace(OLD_INIT, NEW_INIT)
    return text, "patched"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("open_instruct_root", type=Path)
    args = ap.parse_args()

    path = args.open_instruct_root / "open_instruct" / "finetune.py"
    if not path.is_file():
        raise SystemExit(f"OPEN_INSTRUCT_FINETUNE_MISSING: {path}")

    original = path.read_text()
    patched, status = patch_text(original)
    if status == "patched":
        backup = path.with_suffix(".py.lecture6.orig")
        if not backup.exists():
            backup.write_text(original)
        path.write_text(patched)
    print(f"LECTURE6_WANDB_PATCH={status} path={path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
