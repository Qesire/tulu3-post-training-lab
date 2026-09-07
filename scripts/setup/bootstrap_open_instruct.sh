#!/bin/bash
# Prepare the Lecture-6-era Open-Instruct checkout and dedicated Python 3.12 env.
# Run on a network-enabled preparation/login node; formal GPU jobs never git-pull.
set -Eeuo pipefail

REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
UPSTREAM_BASE="${UPSTREAM_BASE:-/home/scc/pb23061276/projects/tulu3-upstream}"
OPEN_INSTRUCT_ROOT="${OPEN_INSTRUCT_ROOT:-$UPSTREAM_BASE/open-instruct-lecture6}"
OPEN_INSTRUCT_COMMIT="${OPEN_INSTRUCT_COMMIT:-8fcf9c6bae3e3a58e1fcb8c79c3bbfbd97065377}"

mkdir -p "$UPSTREAM_BASE"
if [[ ! -d "$OPEN_INSTRUCT_ROOT/.git" ]]; then
  git clone https://github.com/allenai/open-instruct.git "$OPEN_INSTRUCT_ROOT"
fi

git -C "$OPEN_INSTRUCT_ROOT" fetch --tags origin
git -C "$OPEN_INSTRUCT_ROOT" checkout --detach "$OPEN_INSTRUCT_COMMIT"
ACTUAL_COMMIT="$(git -C "$OPEN_INSTRUCT_ROOT" rev-parse HEAD)"
[[ "$ACTUAL_COMMIT" == "$OPEN_INSTRUCT_COMMIT" ]] || {
  echo "UPSTREAM_COMMIT_MISMATCH expected=$OPEN_INSTRUCT_COMMIT actual=$ACTUAL_COMMIT" >&2
  exit 40
}

command -v uv >/dev/null 2>&1 || { echo "UV_NOT_FOUND" >&2; exit 40; }
(
  cd "$OPEN_INSTRUCT_ROOT"
  uv sync --python 3.12
)

"$OPEN_INSTRUCT_ROOT/.venv/bin/python" \
  "$REPO_ROOT/scripts/setup/apply_lecture6_open_instruct_patch.py" "$OPEN_INSTRUCT_ROOT"

BOOTSTRAP_ATTESTATION="${TMPDIR:-/tmp}/tulu3_open_instruct_checkout_${USER:-user}.json"
"$OPEN_INSTRUCT_ROOT/.venv/bin/python" \
  "$REPO_ROOT/scripts/preflight/validate_open_instruct_checkout.py" \
  "$OPEN_INSTRUCT_ROOT" \
  --expected-commit "$OPEN_INSTRUCT_COMMIT" \
  --output "$BOOTSTRAP_ATTESTATION"

"$OPEN_INSTRUCT_ROOT/.venv/bin/python" - <<'PY'
import sys
import torch, transformers, flash_attn, accelerate, datasets
print("python", sys.version.split()[0])
print("torch", torch.__version__, "cuda", torch.version.cuda)
print("transformers", transformers.__version__)
print("flash_attn", flash_attn.__version__)
print("accelerate", accelerate.__version__)
print("datasets", datasets.__version__)
PY

cat <<EOF
OPEN_INSTRUCT_READY=1
OPEN_INSTRUCT_ROOT=$OPEN_INSTRUCT_ROOT
OPEN_INSTRUCT_COMMIT=$ACTUAL_COMMIT
OPEN_INSTRUCT_ATTESTATION=$BOOTSTRAP_ATTESTATION
EOF
