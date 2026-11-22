#!/usr/bin/env bash
# Check out COMMIT (default HEAD) into a separate git worktree for timed campaigns, so that edits in this checkout
# cannot change the code a running campaign imports (lane processes import solver modules lazily). data/, runs/ and
# third_party/<repo>/ are shared with this checkout by symlink. Campaigns must run with PYTHONPATH set to the
# worktree source directory so the package is imported from the selected commit. The rows then record the
# worktree's clean commit as their code version.
#
# Usage: scripts/campaign_worktree.sh [COMMIT] [DIR] (default DIR: <repo>/.worktrees/run; reuse at COMMIT if it exists)
set -euo pipefail
repo="$(cd "$(dirname "$0")/.." && pwd)"
commit="$(git -C "$repo" rev-parse --verify "${1:-HEAD}^{commit}")"
dest="${2:-$repo/.worktrees/run}"
if [ -e "$dest/.git" ]; then
  git -C "$dest" checkout -q --detach "$commit"
else
  git -C "$repo" worktree add -q --detach "$dest" "$commit"
fi
dest="$(cd "$dest" && pwd)"
for link in data runs; do
  [ -e "$dest/$link" ] || ln -s "$repo/$link" "$dest/$link"
done
for sub in "$repo"/third_party/*/; do
  name="$(basename "$sub")"
  [ -e "$dest/third_party/$name" ] || ln -s "$repo/third_party/$name" "$dest/third_party/$name"
done
source_dir="$dest"
if [ -d "$dest/src" ]; then
  source_dir="$dest/src"
fi

# Compile or load the Numba kernels before starting campaign lanes.
(cd "$dest" && PYTHONPATH="$source_dir" OMP_NUM_THREADS=1 NUMBA_NUM_THREADS=1 "$repo/.venv/bin/python" - "$source_dir" <<'PYTHON'
import sys
import tomllib
from importlib import import_module
from pathlib import Path

import numpy as np

project = tomllib.loads(Path("pyproject.toml").read_text())["project"]
package = import_module(project["name"].replace("-", "_"))
source_dir = Path(sys.argv[1])
assert Path(package.__file__).resolve().is_relative_to(source_dir), package.__file__
runtime = import_module(f"{package.__name__}.bench.runtime")
hetero = import_module(f"{package.__name__}.hetero")
alns = import_module(f"{package.__name__}.solvers.alns")
greedy = import_module(f"{package.__name__}.solvers.greedy")

alns._warmup()
rng = np.random.default_rng(0)
toy = hetero.Instance(req=np.eye(2)[[0, 1, 0, 1]], loc=rng.random((4, 2)), dur=rng.random(4),
                      ab=np.eye(2)[[0, 0, 1, 1]], depot=np.zeros((4, 2)), species=[0, 0, 1, 1])
hetero.evaluate(toy, greedy.construct(toy, restarts=2))
print("worktree", Path.cwd(), "at", runtime.code_version())
PYTHON
)
printf 'run campaigns as: cd %q && PYTHONPATH=%q %q scripts/<script>.py ...\n'   "$dest" "$source_dir" "$repo/.venv/bin/python"
