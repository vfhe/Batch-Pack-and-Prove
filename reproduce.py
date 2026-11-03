"""Run every measurement the paper's tables use with vfhe, then print them.

The measurements, one after the other, single-threaded:

1. ``measure.py``: the cost model's microbenchmarks over ``R_q``, ``2^8 ..
   2^16`` (Fig. pc, GKR, Table res_all);
2. ``measure_range.py``: the range proof over ``Z_p``, ``ell* = 20 .. 24``,
   with the parameters of `RANGE_PARAMETERS` (Table res_range_proof).

Each writes a JSON file into ``--out-dir`` (default ``microbenchmarks/``);
then ``tables.py`` prints the paper's tables from them. Each command is
printed before it runs, so a single one can be run again by hand.

``--smoke`` runs every step at tiny sizes into ``results/smoke/``: every
proof is still verified and the tables print (a ``-`` where a size was not
measured), but the times mean nothing.

Usage: python3.12 reproduce.py --vfhe PATH [--smoke] [--out-dir DIR]
                               [--label LABEL] [--any-commit]

``PATH`` is a vfhe checkout at `VFHE_COMMIT`, built with ``make build``.
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
from pathlib import Path

#: The vfhe commit the paper's numbers were measured with.
VFHE_COMMIT = "83a59f7656f5d3256910dc644830490ae5e9279c"

#: ``(bits of p, sigma, m, kappa)`` of the range proof over ``Z_p``.
RANGE_PARAMETERS = (208, 2**11, 15, 20)

#: ``--sizes`` of each step, measured and smoke.
SIZES = {"rq": "8-16", "range": "20-24"}
SMOKE_SIZES = {"rq": "8", "range": "16"}

HERE = Path(__file__).resolve().parent


def vfhe_path(checkout: Path) -> str:
    """The ``PYTHONPATH`` of a built vfhe checkout, as the README sets it."""
    sources = sorted(str(p) for p in (checkout / "modules").glob("*/python/src"))
    return os.pathsep.join(
        [str(checkout / "build"), str(checkout / "build" / "proto"), *sources]
    )


def checkout_commit(checkout: Path) -> tuple[str, bool]:
    """The checkout's ``HEAD`` and whether tracked files differ from it."""

    def git(*args: str) -> str:
        return subprocess.run(
            ["git", "-C", str(checkout), *args],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()

    head = git("rev-parse", "HEAD")
    return head, bool(git("status", "--porcelain", "--untracked-files=no"))


def steps(out_dir: Path, label: str, smoke: bool) -> list[tuple[list[str], Path]]:
    """Each measurement as ``(arguments, output file)``."""
    sizes = SMOKE_SIZES if smoke else SIZES
    fast = ["--repeat", "1", "--budget", "0"] if smoke else []
    rq = out_dir / f"vfhe-{label}.json"
    bits, sigma, m, kappa = RANGE_PARAMETERS
    zp = out_dir / f"vfhe-{label}-range{bits}-sigma{sigma}-m{m}-kappa{kappa}.json"
    return [
        (["measure.py", "--sizes", sizes["rq"], *fast, "--out", str(rq)], rq),
        (
            [
                *("measure_range.py", "--sizes", sizes["range"]),
                *("--bits", str(bits), "--sigma", str(sigma)),
                *("--m", str(m), "--kappa", str(kappa), "--out", str(zp)),
            ],
            zp,
        ),
    ]


def failures(path: Path) -> list[str]:
    """The sizes or workers a measurement file records as failed."""
    if not path.exists():
        return ["no output"]
    runs = json.loads(path.read_text()).get("runs", {})
    return [key for key, run in runs.items() if "error" in run]


def main() -> None:
    parser = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    parser.add_argument("--vfhe", type=Path, required=True, help="a built checkout")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--out-dir", type=Path)
    parser.add_argument("--label", default="measured", help="for the file names")
    parser.add_argument(
        "--any-commit", action="store_true", help="accept another vfhe commit"
    )
    args = parser.parse_args()

    checkout = args.vfhe.resolve()
    head, dirty = checkout_commit(checkout)
    if (head != VFHE_COMMIT or dirty) and not args.any_commit:
        state = " with local changes" if dirty else ""
        sys.exit(
            f"vfhe is at {head[:7]}{state}, the paper's numbers are from "
            f"{VFHE_COMMIT[:7]}: check that commit out (or pass --any-commit)"
        )
    out_dir = args.out_dir or HERE / (
        "results/smoke" if args.smoke else "microbenchmarks"
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    env = os.environ | {"PYTHONPATH": vfhe_path(checkout), "VFHE_NUM_THREADS": "1"}

    plan = steps(out_dir, args.label, args.smoke)
    failed = []
    for arguments, path in plan:
        # The measurements ignore the user site when this process does.
        command = [sys.executable, *(["-s"] if sys.flags.no_user_site else [])]
        command += arguments
        print(f"== {shlex.join(command)}", flush=True)
        run = subprocess.run(command, cwd=HERE, env=env)
        problems = failures(path)
        if run.returncode != 0 or problems:
            failed.append(f"{path.name}: exit {run.returncode}; {', '.join(problems)}")

    (_, rq), (_, zp) = plan
    tables = [sys.executable, "tables.py", "--microbenchmarks", str(rq)]
    # An empty --range leaves the range table out, not the paper's file in.
    tables += [f"--range={zp if zp.exists() else ''}"]
    print(f"== {shlex.join(tables)}", flush=True)
    subprocess.run(tables, cwd=HERE, check=True)
    if failed:
        sys.exit("failed:\n  " + "\n  ".join(failed))


if __name__ == "__main__":
    main()
