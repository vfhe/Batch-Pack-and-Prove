"""Measure the cost model's microbenchmarks with vfhe.

Times, single-threaded, at the paper's ring (``N = 2^13``, three 45-bit
primes, splitting degree 4) and at each size ``2^v`` ``R_q`` elements:

- ``commit``: `Basefold.commit` of a multilinear polynomial with ``2^v``
  random evaluations (coefficient transform, encoding and Merkle tree);
- ``basefold_eval``: the prover of one `BasefoldEval` claim at a point of
  random ``R_q`` coordinates;
- ``batchfold_batching``: the prover of one `BatchFoldEval` claim at a
  point of exceptional coordinates, where the common-point sumcheck is
  skipped: the random linear combination of the CRT components and one
  Basefold run per prime over the residue code;
- ``sumcheck``: the prover of a `SumcheckProd` over two random tables of
  ``2^v`` ``R_q`` elements;
- ``merkle``: the Merkle tree of ``commit`` alone (`Basefold.merkle_commit`
  on the codeword).

The code is the general foldable code with twists in the residue fields
``F_(p^4)`` (``twist_field="field"``), with ``k0``, ``n0`` and the number of
query repetitions from `basefold_parameters.estimate`, the estimator of the
paper.
Each time is the minimum over ``--repeat`` runs (fewer once a series
has used ``--budget`` seconds), and every proof is verified once.
More series are recorded under ``details``: the parts of ``commit``
(``coefficients``, ``encode``), the trees of the folded codewords that a
Basefold proof commits (``merkle_layers``), the verifiers (one run each),
``batchfold_eval_generic``, the BatchFold prover at the point of
``basefold_eval``, where the common-point sumcheck runs, and
``sumcheck_statement_digest``, the digest of the sumcheck's statement. The
prover binds that digest to its Fiat-Shamir transcript, so ``sumcheck``
includes it: one hash of both whole tables, which a sumcheck over virtual or
committed oracles does not compute (`estimates.Microbenchmarks.load` takes
it out).

Each size runs in its own processes, so the memory of one is returned
before the next. The output is what ``tables.py --microbenchmarks FILE``
reads, and is rewritten after each size.

Usage: python3.12 measure.py [--sizes 8-16] [--repeat 3] [--budget 300]
                             [--machine LABEL] [--out FILE]

Needs vfhe on ``PYTHONPATH``, as ``networks.py`` does.
"""

from __future__ import annotations

import argparse
import json
import platform
import resource
import subprocess
import sys
import time
from collections.abc import Callable
from functools import partial
from pathlib import Path
from typing import TypeVar

import basefold_parameters

T = TypeVar("T")

LOG_N = 13
SPLIT_DEGREE = 4
PRIME_BITS = [45, 45, 45]
TWIST_FIELD = "field"
INSTANTIATION = "general"
SIZES = range(8, 17)

#: The series of the cost model, then Merkle.
SERIES = ("commit", "basefold_eval", "batchfold_batching", "sumcheck", "merkle")

#: The worker processes run at each size, one after the other.
WORKERS = ("pcs", "sumcheck")


def parameters(log_size: int, log_field: int = basefold_parameters.LOG_FIELD) -> dict:
    p = basefold_parameters.estimate(log_size, log_field=log_field)
    return {
        "k0": p.k0,
        "n0": p.n0,
        "c": p.n0 // p.k0,
        "folds": p.folds,
        "repetitions": p.repetitions,
    }


class Timer:
    """The minimum time of an operation over at most ``repeat`` runs (or
    ``runs``), and fewer once they have taken ``budget`` seconds; at least
    one run."""

    def __init__(self, repeat: int, budget: float):
        self.repeat = repeat
        self.budget = budget
        self.samples: dict[str, list[float]] = {}

    def __call__(self, name: str, op: Callable[[], T], runs: int | None = None) -> T:
        samples = self.samples.setdefault(name, [])
        runs = self.repeat if runs is None else runs
        spent = 0.0
        while True:
            start = time.perf_counter_ns()
            result = op()
            elapsed = (time.perf_counter_ns() - start) / 1e9
            samples.append(elapsed * 1000)
            spent += elapsed
            if len(samples) >= runs or spent >= self.budget:
                return result
            # Drop this result before the next run, so two never coexist in memory.
            del result

    def best(self) -> dict[str, float]:
        return {name: min(s) for name, s in self.samples.items()}


def ring():
    from vfhe.arith import Ring

    r = Ring(1 << LOG_N, prime_size=PRIME_BITS, split_degree=SPLIT_DEGREE)
    if r.split_degree != SPLIT_DEGREE:
        raise RuntimeError(f"the ring splits with degree {r.split_degree}")
    return r


def random_mle(rq, variables: list):
    from vfhe.arith import MLE

    entries = [rq.random_element() for _ in range(1 << len(variables))]
    return MLE(ring=rq, variables=variables, evaluations=entries)


def measure_pcs(log_size: int, timer: Timer) -> dict:
    """Basefold commit and evaluation, BatchFold evaluation, Merkle trees."""
    from vfhe.arith import MLE_Variable
    from vfhe.piop import IOP, Relation_Eval, Statement
    from vfhe.polycom import Basefold, BasefoldEval, BatchFoldEval, FoldableRS

    rq = ring()
    p = parameters(log_size)
    code = FoldableRS(
        rq,
        p["k0"],
        p["c"],
        p["folds"],
        instantiation=INSTANTIATION,
        twist_field=TWIST_FIELD,
    )
    scheme = Basefold(code)
    f = random_mle(rq, [MLE_Variable(f"x{i}") for i in range(log_size)])

    def commit():
        scheme.openings.clear()
        return scheme.commit(f)

    oracle, opening = timer("commit", commit)
    root = oracle.commitment.root

    table = timer("coefficients", lambda: f.to_coefficients().table)
    word = timer("encode", partial(code.encode, table))
    assert word == opening.word, "encode differs from the committed codeword"
    del word, table

    tree = timer("merkle", lambda: scheme.merkle_commit(opening.word))
    assert tree.root == root, "merkle_commit differs from the commitment"
    tree = None
    # The trees of the codewords folded to levels folds-1 .. 1, which a
    # Basefold proof commits; a tree's cost depends only on its length.
    levels = range(p["folds"] - 1, 0, -1)
    timer(
        "merkle_layers",
        lambda: [scheme.merkle_commit(opening.word[: code.n0 << lv]) for lv in levels],
    )

    def claim(point) -> Statement:
        value = f.evaluate(point, in_place=False).constant()
        return Statement(Relation_Eval(), oracles=(oracle,), point=point, value=value)

    def iop(protocol) -> IOP:
        iop = IOP(domain=rq, fiat_shamir=True)
        iop.register(Relation_Eval, protocol)
        return iop

    def prove_and_verify(name: str, verify: str, protocol, statement) -> None:
        proof = timer(name, lambda: iop(protocol).prove(statement))
        accepted = timer(verify, lambda: iop(protocol).verify(statement, proof), 1)
        assert accepted, f"{name}: the proof does not verify"

    generic = claim({v: rq.random_element() for v in f.variables})
    exceptional = claim({v: rq.random_exceptional() for v in f.variables})
    reps = p["repetitions"]
    basefold = BasefoldEval(scheme, rep=reps)
    prove_and_verify("basefold_eval", "basefold_verify", basefold, generic)
    batchfold = BatchFoldEval(scheme, rep=reps)
    prove_and_verify("batchfold_batching", "batchfold_verify", batchfold, exceptional)
    prove_and_verify(
        "batchfold_eval_generic", "batchfold_verify_generic", batchfold, generic
    )
    return p


def measure_sumcheck(log_size: int, timer: Timer) -> dict:
    """A degree-2 sumcheck over ``R_q``: two tables, the native kernel."""
    from vfhe.arith import MLE_Variable
    from vfhe.piop import IOP, Statement, TableOracle, element_digest
    from vfhe.piop.protocols import SumcheckProd
    from vfhe.piop.relations import Relation_SumProd

    rq = ring()
    variables = [MLE_Variable(f"x{i}") for i in range(log_size)]
    f, g = random_mle(rq, variables), random_mle(rq, variables)
    value = None
    for a, b in zip(f.table, g.table, strict=True):
        value = a * b if value is None else value + a * b
    statement = Statement(
        Relation_SumProd(), oracles=[TableOracle(f), TableOracle(g)], value=value
    )

    def iop() -> IOP:
        iop = IOP(domain=rq, fiat_shamir=True)
        iop.register(Relation_SumProd, SumcheckProd())
        return iop

    proof = timer("sumcheck", lambda: iop().prove(statement))
    timer("sumcheck_statement_digest", lambda: element_digest(statement))
    accepted = timer("sumcheck_verify", lambda: iop().verify(statement, proof), 1)
    assert accepted, "sumcheck: the proof does not verify"
    return {}


def worker(name: str, log_size: int, repeat: int, budget: float) -> dict:
    from vfhe.util.parallel import num_threads, set_num_threads

    set_num_threads(1)
    assert num_threads() == 1
    timer = Timer(repeat, budget)
    measure = {"pcs": measure_pcs, "sumcheck": measure_sumcheck}[name]
    params = measure(log_size, timer)
    return {
        "times": timer.best(),
        "samples": timer.samples,
        "parameters": params,
        "max_rss_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
    }


# --- The driver -------------------------------------------------------------


def vfhe_version() -> dict:
    import vfhe.arith
    from vfhe.util.bindings import lib

    checkout = Path(vfhe.arith.__path__[0])

    def git(*args: str) -> str:
        return subprocess.run(
            ["git", "-C", str(checkout), *args], capture_output=True, text=True
        ).stdout.strip()

    return {
        "commit": git("rev-parse", "--short=7", "HEAD") or "unknown",
        "branch": git("rev-parse", "--abbrev-ref", "HEAD"),
        "dirty": bool(git("status", "--porcelain", "--untracked-files=no")),
        "engine": getattr(lib, "__name__", "unknown"),
    }


def machine() -> dict:
    info: dict[str, str | int | float] = {
        "python": platform.python_version(),
        "system": platform.platform(),
    }
    try:
        cpuinfo = Path("/proc/cpuinfo").read_text()
        meminfo = Path("/proc/meminfo").read_text()
    except OSError:
        return info
    for line in cpuinfo.splitlines():
        if line.startswith("model name"):
            info["cpu"] = line.split(":", 1)[1].strip()
            break
    info["cpus"] = cpuinfo.count("\nprocessor") + cpuinfo.startswith("processor")
    for line in meminfo.splitlines():
        if line.startswith("MemTotal"):
            info["memory_gib"] = round(int(line.split()[1]) / 2**20, 1)
    return info


def parse_sizes(text: str) -> list[int]:
    if "-" in text:
        lo, hi = map(int, text.split("-"))
        return list(range(lo, hi + 1))
    return [int(s) for s in text.split(",")]


def describe(version: dict, host: dict) -> str:
    return (
        f"Single-threaded vfhe measurements ({host.get('cpu', 'unknown CPU')}, "
        f"engine {version['engine']}, vfhe {version['commit']}"
        f"{' with local changes' if version['dirty'] else ''}), over R_q with "
        "N = 2^13, three 45-bit primes and splitting degree 4. Times in "
        "milliseconds by log2 of the input size in R_q elements, each the "
        "minimum over its runs. Basefold uses the general foldable code with "
        "twists in F_(p^4), and the estimator's k0, n0 and query "
        "repetitions; "
        "commit and every proof include the Merkle hashing, which merkle times "
        "alone for the commitment. basefold_eval is one claim at a point of "
        "random R_q coordinates. batchfold_batching is BatchFold's evaluation "
        "of one claim at a point of exceptional coordinates, where vfhe skips "
        "the common-point sumcheck; sumcheck is a degree-2 sumcheck over R_q "
        "(SumcheckProd, two tables)."
    )


def output(path: Path, sizes: list[int], results: dict, header: dict) -> None:
    """The file: the sizes every series of the cost model has, then the rest
    under ``details``."""
    complete = [v for v in sizes if all(s in results.get(v, {}) for s in SERIES)]
    data = {
        "description": header["description"],
        "unit": "ms",
        "log_sizes": complete,
    }
    for s in SERIES:
        data[s] = [round(results[v][s], 1) for v in complete]
    extra = sorted({s for r in results.values() for s in r} - set(SERIES))
    data["details"] = {
        s: {str(v): round(results[v][s], 1) for v in sizes if s in results.get(v, {})}
        for s in extra
    }
    data["configuration"] = header["configuration"]
    data["vfhe"] = header["vfhe"]
    data["machine"] = header["machine"]
    data["runs"] = header["runs"]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n\n")[0])
    parser.add_argument("--sizes")
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument("--budget", type=float, default=300.0)
    parser.add_argument("--machine", help="label for the file name")
    parser.add_argument("--out", type=Path)
    parser.add_argument("--worker", nargs=2, metavar=("NAME", "LOG_SIZE"))
    args = parser.parse_args()

    if args.worker:
        name, log_size = args.worker[0], int(args.worker[1])
        report = worker(name, log_size, args.repeat, args.budget)
        print(json.dumps(report))
        return

    sizes = parse_sizes(args.sizes or f"{SIZES[0]}-{SIZES[-1]}")
    version, host = vfhe_version(), machine()
    label = args.machine or "".join(
        c if c.isalnum() else "-" for c in host.get("cpu", "unknown").lower()
    ).strip("-")
    out = args.out or Path(__file__).with_name("microbenchmarks") / (
        f"vfhe-{version['commit']}-{label}.json"
    )
    config = {
        "domain": "rq",
        "log_n": LOG_N,
        "prime_bits": PRIME_BITS,
        "split_degree": SPLIT_DEGREE,
        "instantiation": INSTANTIATION,
        "twist_field": TWIST_FIELD,
        "threads": 1,
        "repeat": args.repeat,
        "budget_s": args.budget,
        "parameters": {str(v): parameters(v) for v in sizes},
    }
    header = {
        "description": describe(version, host),
        "configuration": config,
        "vfhe": version,
        "machine": host,
        "runs": {},
    }
    results: dict[int, dict[str, float]] = {}
    for v in sizes:
        for name in WORKERS:
            print(f"== 2^{v} {name}", file=sys.stderr, flush=True)
            run = subprocess.run(
                [
                    sys.executable,
                    # The workers ignore the user site as this process does.
                    *(["-s"] if sys.flags.no_user_site else []),
                    *(__file__, "--worker", name, str(v)),
                    *("--repeat", str(args.repeat), "--budget", str(args.budget)),
                ],
                stdout=subprocess.PIPE,
                text=True,
            )
            key = f"{v}/{name}"
            if run.returncode != 0:
                header["runs"][key] = {"error": f"exit {run.returncode}"}
                print(f"   failed: exit {run.returncode}", file=sys.stderr)
                continue
            report = json.loads(run.stdout.strip().splitlines()[-1])
            results.setdefault(v, {}).update(report.pop("times"))
            header["runs"][key] = report
            for s, ms in sorted(report["samples"].items()):
                print(f"   {s}: {min(ms):.1f} ms ({len(ms)} runs)", file=sys.stderr)
            output(out, sizes, results, header)
    output(out, sizes, results, header)
    print(out)


if __name__ == "__main__":
    main()
