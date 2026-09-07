# Open-Instruct patches

The Lecture-6 W&B compatibility fix is applied by `scripts/setup/apply_lecture6_open_instruct_patch.py` against the pinned lecture-era Open-Instruct checkout.

The patch is fail-closed and expects exactly three `wandb_tracker.run.get_url()` call sites plus one tracker initialization site. No formal run may depend on an unrecorded manual edit to upstream source.
