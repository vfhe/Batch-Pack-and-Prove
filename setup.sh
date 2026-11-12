#!/usr/bin/env bash
# Install vfhe at the commit the paper's numbers come from (reproduce.py's
# VFHE_COMMIT), built, next to this repository:
#
#   bash setup.sh
#
# It clones vfhe into vfhe/ (or fetches into an existing clone), checks out
# that commit with its submodules, creates a virtual environment in venv/
# with vfhe's build dependencies and pytest, builds vfhe with `make build`,
# and makes the venv's Python import it. Then:
#
#   venv/bin/python reproduce.py --vfhe vfhe
#   venv/bin/python -m pytest
#
# Needs git, make, a C compiler and Python 3.11 or later with its venv module
# and headers (Debian/Ubuntu: apt-get install build-essential python3-venv
# python3-dev). PYTHON picks the interpreter (default python3), VFHE another
# directory for the clone.
set -euo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
VFHE=${VFHE:-$HERE/vfhe}
VENV=$HERE/venv
URL=https://github.com/vfhe/vfhe.git

commit=$(sed -n 's/^VFHE_COMMIT = "\([0-9a-f]\{40\}\)"$/\1/p' "$HERE/reproduce.py")
[ -n "$commit" ] || { echo "no VFHE_COMMIT in reproduce.py" >&2; exit 1; }

[ -d "$VFHE/.git" ] || git clone --quiet --no-checkout "$URL" "$VFHE"
git -C "$VFHE" fetch --quiet origin "$commit"
git -C "$VFHE" checkout --quiet --detach "$commit"
git -C "$VFHE" submodule update --quiet --init --recursive --depth 1

PY=${PYTHON:-python3}
if ! "$PY" -c 'import sys; sys.exit(sys.version_info < (3, 11))'; then
    echo "vfhe needs Python 3.11 or later; $PY is $("$PY" --version 2>&1) (set PYTHON)" >&2
    exit 1
fi
[ -x "$VENV/bin/python" ] || "$PY" -m venv "$VENV" || { rm -rf "$VENV"; exit 1; }
make -C "$VFHE" deps PYTHON="$VENV/bin/python" DEPENDENCY_GROUPS=compile
# meson builds the extension for the Python it runs under: the venv's.
PATH="$VENV/bin:$PATH" make -C "$VFHE" build

# The venv imports vfhe from the checkout, as PYTHONPATH would; pytest runs
# this repository's tests.
site=$("$VENV/bin/python" -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')
{ echo "$VFHE/build"; echo "$VFHE/build/proto"; ls -d "$VFHE"/modules/*/python/src; } \
    > "$site/vfhe.pth"
"$VENV/bin/python" -m pip install --quiet pytest

echo "vfhe ${commit:0:7} built in $VFHE; run: venv/bin/python reproduce.py --vfhe ${VFHE#"$HERE"/}"
