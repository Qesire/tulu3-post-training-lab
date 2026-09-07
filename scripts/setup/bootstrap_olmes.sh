#!/bin/bash
# Prepare the lecture-era OLMES checkout in a dedicated GPU evaluation environment.
set -Eeuo pipefail

UPSTREAM_BASE="${UPSTREAM_BASE:-/home/scc/pb23061276/projects/tulu3-upstream}"
OLMES_ROOT="${OLMES_ROOT:-$UPSTREAM_BASE/olmes-lecture6}"
OLMES_COMMIT="${OLMES_COMMIT:-f122a7bad99551ffb3f5feec14169af794926f6b}"

mkdir -p "$UPSTREAM_BASE"
if [[ ! -d "$OLMES_ROOT/.git" ]]; then
  git clone https://github.com/allenai/olmes.git "$OLMES_ROOT"
fi

git -C "$OLMES_ROOT" fetch --tags origin
git -C "$OLMES_ROOT" checkout --detach "$OLMES_COMMIT"
ACTUAL_COMMIT="$(git -C "$OLMES_ROOT" rev-parse HEAD)"
[[ "$ACTUAL_COMMIT" == "$OLMES_COMMIT" ]] || {
  echo "OLMES_COMMIT_MISMATCH expected=$OLMES_COMMIT actual=$ACTUAL_COMMIT" >&2
  exit 40
}

command -v uv >/dev/null 2>&1 || { echo "UV_NOT_FOUND" >&2; exit 40; }
(
  cd "$OLMES_ROOT"
  uv sync --python 3.12 --group gpu
)

"$OLMES_ROOT/.venv/bin/python" - <<'PY'
import sys
import torch, transformers, vllm
print("python", sys.version.split()[0])
print("torch", torch.__version__, "cuda", torch.version.cuda)
print("transformers", transformers.__version__)
print("vllm", vllm.__version__)
PY

cat <<EOF
OLMES_READY=1
OLMES_ROOT=$OLMES_ROOT
OLMES_COMMIT=$ACTUAL_COMMIT
EOF
