#!/usr/bin/env bash
# Run one catan_rl job on a Colab runtime and zip its run directory.
#
#   bash catan/scripts/colab.sh <module> --name <run> [flags]     e.g. colab.sh ppo --name gnn1 --arch gnn ...
#
# Pulls the latest master, builds and installs the engine (Rust is installed on first use; Colab's own
# CUDA torch is kept), runs `python -m catan_rl.<module>` from the repo root, then writes
# /content/<run>.zip containing runs/<run>/. The notebook then downloads it with google.colab.files.
set -euo pipefail
module=$1; shift
name=$(printf '%s\n' "$@" | grep -A1 -x -- --name | tail -1)
[ -n "$name" ] || { echo "colab.sh: --name is required" >&2; exit 2; }

cd "$(dirname "$0")/.."
git pull -q --ff-only
git log --oneline -1
command -v cargo >/dev/null || [ -f ~/.cargo/env ] || curl -sSf https://sh.rustup.rs | sh -s -- -y -q --profile minimal >/dev/null
source ~/.cargo/env
pip -q install "maturin>=1.7,<2" matplotlib wandb
rm -rf /content/wheels
maturin build --release -q -o /content/wheels 2>&1 | tail -1
pip -q install --no-deps --force-reinstall /content/wheels/*.whl
python -c "import torch; print('torch', torch.__version__, 'gpu:', torch.cuda.get_device_name() if torch.cuda.is_available() else None)"

python -m "catan_rl.$module" "$@"

rm -f "/content/$name.zip"
zip -qr "/content/$name.zip" "runs/$name"
echo "wrote /content/$name.zip ($(du -h "/content/$name.zip" | cut -f1))"
