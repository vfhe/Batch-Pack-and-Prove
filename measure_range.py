"""Measure the range proof over Z_p, compiled with Basefold, with vfhe.

The protocol is vfhe's `RangeDecomposition` (the paper's optimized range
proof) reducing to `Lookup` (the lookup argument over ``Z_p``), with every
oracle they send committed by field Basefold over a pseudo-Mersenne prime
(`BasefoldMessages`): one scheme for the ``2^ell*`` tables (``mu'``,
``read_ts``, ``f1_WSmu``, ``f1_RS``) and one for the ``2^b*`` tables
(``final_cts``, ``f1_WSt``, ``f1_S``); their claims are opened in one
batched opening per scheme (common-point sumcheck and random linear
combination). ``v`` is committed with ring Basefold and its claims opened
with `BatchFoldEval`, as the paper's cost model does for ``R_q``
evaluations.

The shape is the paper's for its largest model: ``R_q`` with ``N = 2^13``,
three 45-bit primes and splitting degree 4, ``c = 4`` digits of ``beta =
2^11`` (``B = 2^44``), so a claim on ``v`` with ``ell`` variables has ``2^ell*``
digits, ``ell* = ell + 15``. The sizes are ``ell*`` (default 20-24). The
challenge parameters ``(sigma, m, kappa)`` default to ``(2^11, 15, 20)``;
``m`` is capped at ``ell + 13``, its largest value.

Times, in milliseconds, one prover run per size (the minimum over
``--repeat`` runs), single-threaded, Fiat-Shamir:

- ``digits``: the range prover without its commitment: the digit table,
  the ``kappa`` partial evaluations ``psi`` and the ``kappa`` values of ``v``;
- ``lookup``: the lookup prover without its commitments: the counters, the
  four fingerprint tables and their grand products;
- ``commit_main`` / ``commit_table``: the commitments of the four ``2^ell*``
  and of the three ``2^b*`` oracles; ``commit`` is their sum;
- ``eval_main`` / ``eval_table``: the batched Basefold openings of the claims
  on each scheme's oracles; ``eval`` is their sum;
- ``v_eval``: the BatchFold opening of the ``kappa`` claims on ``v`` (part
  of the AC's batched opening in the paper, so kept apart). From ``ell = 8``
  on, ``v``'s code is ``measure.py``'s: the estimator's ``k0``, ``n0`` and
  query repetitions, twists in ``F_(p^4)``. The estimator starts at ``2^8``,
  so below that a smaller code of the same kind is used
  (``v_code_from_estimator`` false in the output);
- ``virtual_eval``: the openings of the virtual oracles the zerochecks end
  on, down to claims on the committed ones;
- ``zerochecks``: the sumcheck rounds of the lookup's two degree-3
  zerochecks;
- ``prove``: the whole prover run;
- ``verify``: the verifier, once.

The first proof at each size is verified. Before measuring, the driver
also runs the proof at a toy shape (``N = 8``, two variables) and checks
that it verifies and that a tampered commitment does not.

Each size runs in its own process, so the memory of one is returned before
the next; the output is rewritten after each size.

Usage: python3.12 measure_range.py [--bits 208] [--sizes 20-24]
                                   [--sigma 2048] [--m 15] [--kappa 20]
                                   [--repeat 1] [--machine LABEL] [--out FILE]

Needs vfhe on ``PYTHONPATH``, as ``measure.py`` does, whose helpers it
uses.
"""

from __future__ import annotations

import argparse
import json
import random
import resource
import subprocess
import sys
import time
from pathlib import Path

from measure import (
    INSTANTIATION,
    TWIST_FIELD,
    machine,
    parameters,
    parse_sizes,
    vfhe_version,
)

LOG_N = 13
PRIME_BITS = [45, 45, 45]
SPLIT_DEGREE = 4
CHUNKS = 4
LOG_BETA = 11
BITS = 208
SIZES = range(20, 25)
SIGMA, M, KAPPA = 1 << 11, 15, 20

#: The 2-adicity of the pseudo-Mersenne prime, for its base transform.
TWO_ADICITY = 32

#: v's code below the estimator's first size: four times the message length,
#: as the estimator's codes, and at most this many query repetitions.
V_RATE = 4
V_REPETITIONS = 32

SERIES = (
    "digits",
    "lookup",
    "commit_main",
    "commit_table",
    "commit",
    "eval_main",
    "eval_table",
    "eval",
    "v_eval",
    "virtual_eval",
    "zerochecks",
)


def pmf_field(bits: int):
    """The pseudo-Mersenne prime field ``Z_p`` of ``bits`` bits."""
    from vfhe.arith import PseudoMersenneField

    return PseudoMersenneField.generate(bits, two_adicity=TWO_ADICITY)


class Clock:
    """Wall time accumulated per phase, in milliseconds."""

    def __init__(self):
        self.ms: dict[str, float] = {}

    def add(self, phase: str, start_ns: int) -> None:
        elapsed = (time.perf_counter_ns() - start_ns) / 1e6
        self.ms[phase] = self.ms.get(phase, 0.0) + elapsed


def timed_classes(clock: Clock, main_vars: int):
    """Subclasses of the protocols and of `BasefoldMessages` that add the
    time of each prover call to `clock`."""
    from vfhe.piop.protocols import (
        Lookup,
        RangeDecomposition,
        Sumcheck,
        VirtualEval,
    )
    from vfhe.polycom import BasefoldEval, BasefoldMessages, BatchFoldEval

    class Messages(BasefoldMessages):
        def send(self, prover, label, table):
            start = time.perf_counter_ns()
            oracle = super().send(prover, label, table)
            main = table.num_vars == main_vars
            clock.add("commit_main" if main else "commit_table", start)
            if label.endswith("/mu"):
                clock.add("commit_mu", start)
            return oracle

    def timed(base, phase):
        class Timed(base):
            async def prove(self, prover, statements):
                start = time.perf_counter_ns()
                out = await super().prove(prover, statements)
                clock.add(phase, start)
                return out

        Timed.__name__ = base.__name__
        return Timed

    def timed_eval(base):
        class Eval(base):
            async def prove(self, prover, statements):
                start = time.perf_counter_ns()
                out = await super().prove(prover, statements)
                clock.add(self.phase, start)
                return out

        Eval.__name__ = base.__name__
        return Eval

    return {
        "messages": Messages,
        "range": timed(RangeDecomposition, "range"),
        "lookup": timed(Lookup, "lookup_total"),
        "sumcheck": timed(Sumcheck, "zerochecks"),
        "virtual": timed(VirtualEval, "virtual_eval"),
        "eval": timed_eval(BasefoldEval),
        "batchfold": timed_eval(BatchFoldEval),
    }


def build(field, ring, ell: int, sigma: int, m: int, kappa: int, clock):
    """The schemes, the committed `v`, the range claim and an IOP factory."""
    from vfhe.arith import MLE, MLE_Variable, Polynomial
    from vfhe.piop import IOP, OracleKind, Relation_Eval, Statement
    from vfhe.piop.relations import Relation_Lookup, Relation_Range, Relation_Sum
    from vfhe.polycom import Basefold, FieldFoldableRS, FoldableRS

    nu = ring.N.bit_length() - 1
    gamma = CHUNKS.bit_length() - 1
    ell_star, b_star = ell + nu + gamma, LOG_BETA
    bound = 1 << (LOG_BETA * CHUNKS)
    classes = timed_classes(clock, ell_star)

    def field_scheme(log_size: int):
        if log_size >= 8:
            p = parameters(log_size, field.prime.bit_length())
            k0, c, folds, reps = p["k0"], p["c"], p["folds"], p["repetitions"]
        else:  # the toy shape of the check
            k0, c, folds, reps = 4, 4, log_size - 2, 4
        scheme = Basefold(FieldFoldableRS(field, k0, c, folds))
        return scheme, reps

    main, main_reps = field_scheme(ell_star)
    table, table_reps = field_scheme(b_star)
    if ell >= 8:
        p = parameters(ell)
        v_k0, v_rate, v_folds, v_reps = p["k0"], p["c"], p["folds"], p["repetitions"]
    else:
        v_folds = max(1, ell - 2)
        v_k0, v_rate = 1 << (ell - v_folds), V_RATE
        v_reps = min(V_REPETITIONS, V_RATE * v_k0 - 1)
    v_code = FoldableRS(
        ring,
        v_k0,
        v_rate,
        v_folds,
        instantiation=INSTANTIATION,
        twist_field=TWIST_FIELD,
    )
    v_scheme = Basefold(v_code)

    rng = random.Random(ell)
    variables = [MLE_Variable(f"x{i}") for i in range(ell)]
    table_v = MLE(
        ring=ring,
        variables=variables,
        evaluations=[
            Polynomial(ring).from_array([rng.getrandbits(44) for _ in range(ring.N)])
            for _ in range(1 << ell)
        ],
    )
    v, v_opening = v_scheme.commit(table_v)
    statement = Statement(Relation_Range(bound, CHUNKS), oracles=[v])

    def iop(**kw) -> IOP:
        messages = classes["messages"]([main, table])
        iop = IOP(domain=field, **kw)
        iop.register(
            Relation_Range,
            classes["range"](field, m, sigma, kappa, ring, messages=messages),
        )
        iop.register(Relation_Lookup, classes["lookup"](field, messages=messages))
        iop.register(Relation_Sum, classes["sumcheck"]())
        iop.register(Relation_Eval, classes["virtual"](), kind=OracleKind.virtual)
        for scheme, reps, phase, kind in (
            (main, main_reps, "eval_main", "eval"),
            (table, table_reps, "eval_table", "eval"),
            (v_scheme, v_reps, "v_eval", "batchfold"),
        ):
            protocol = classes[kind](scheme, rep=reps)
            protocol.phase = phase
            iop.register(Relation_Eval, protocol)
        iop.prover.witnesses[v] = table_v
        iop.prover.witnesses[v.commitment] = v_opening
        return iop

    config = {
        "ell": ell,
        "ell_star": ell_star,
        "b_star": b_star,
        "m": m,
        "main_code": {"k0": main.code.k0, "c": main.code.c, "d": main.code.d},
        "main_repetitions": main_reps,
        "table_code": {"k0": table.code.k0, "c": table.code.c, "d": table.code.d},
        "table_repetitions": table_reps,
        "v_code": {"k0": v_k0, "c": v_rate, "d": v_folds},
        "v_repetitions": v_reps,
        "v_code_from_estimator": ell >= 8,
    }
    return statement, iop, config


def check() -> None:
    """The compiled proof at a toy shape: it verifies, and a tampered
    commitment does not."""
    from vfhe.arith import Ring

    field = pmf_field(BITS)
    ring = Ring(8, prime_size=[30, 30], split_degree=1)
    statement, iop, _ = build(field, ring, 2, 1 << 8, 5, 2, Clock())
    proof = iop(fiat_shamir=True).prove(statement)
    if not iop(fiat_shamir=True).verify(statement, proof):
        raise SystemExit("check: the compiled range proof does not verify")
    messages = list(proof.messages)
    label, root = messages[0]
    messages[0] = (label, bytes([root[0] ^ 1]) + root[1:])
    if iop(fiat_shamir=True).verify(statement, type(proof)(messages)):
        raise SystemExit("check: a tampered commitment verifies")


def worker(ell_star: int, args) -> dict:
    from vfhe.arith import Ring
    from vfhe.util.parallel import num_threads, set_num_threads

    set_num_threads(1)
    assert num_threads() == 1
    field = pmf_field(args.bits)
    ring = Ring(1 << LOG_N, prime_size=PRIME_BITS, split_degree=SPLIT_DEGREE)
    if ring.split_degree != SPLIT_DEGREE:
        raise RuntimeError(f"the ring splits with degree {ring.split_degree}")
    nu = LOG_N
    ell = ell_star - nu - (CHUNKS.bit_length() - 1)
    m = min(args.m, ell + nu)
    best: dict[str, float] = {}
    samples: dict[str, list[float]] = {}
    config = {}
    for run in range(args.repeat):
        clock = Clock()
        statement, iop, config = build(
            field, ring, ell, args.sigma, m, args.kappa, clock
        )
        start = time.perf_counter_ns()
        proof = iop(fiat_shamir=True).prove(statement)
        clock.add("prove", start)
        ms = clock.ms
        ms["commit"] = ms.get("commit_main", 0.0) + ms.get("commit_table", 0.0)
        ms["eval"] = ms.get("eval_main", 0.0) + ms.get("eval_table", 0.0)
        commit_mu = ms.pop("commit_mu", None)
        ms["digits"] = ms.pop("range") - (commit_mu or 0.0)
        ms["lookup"] = ms.pop("lookup_total") - (ms["commit"] - (commit_mu or 0.0))
        if run == 0:
            start = time.perf_counter_ns()
            if not iop(fiat_shamir=True).verify(statement, proof):
                raise SystemExit(f"2^{ell_star}: the range proof does not verify")
            ms["verify"] = (time.perf_counter_ns() - start) / 1e6
        for phase, value in ms.items():
            samples.setdefault(phase, []).append(value)
            best[phase] = min(best.get(phase, value), value)
        statement = iop = proof = None
    return {
        "times": best,
        "samples": samples,
        "parameters": config,
        "max_rss_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n\n")[0])
    parser.add_argument("--bits", type=int, default=BITS)
    parser.add_argument("--sizes")
    parser.add_argument("--sigma", type=int, default=SIGMA)
    parser.add_argument("--m", type=int, default=M)
    parser.add_argument("--kappa", type=int, default=KAPPA)
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--machine", help="label for the file name")
    parser.add_argument("--out", type=Path)
    parser.add_argument("--worker", type=int, metavar="ELL_STAR")
    args = parser.parse_args()

    if args.worker is not None:
        print(json.dumps(worker(args.worker, args)))
        return

    check()
    sizes = parse_sizes(args.sizes or f"{SIZES[0]}-{SIZES[-1]}")
    version, host = vfhe_version(), machine()
    label = args.machine or "".join(
        c if c.isalnum() else "-" for c in host.get("cpu", "unknown").lower()
    ).strip("-")
    out = args.out or Path(__file__).with_name("microbenchmarks") / (
        f"vfhe-{version['commit']}-{label}-range{args.bits}.json"
    )
    data = {
        "description": (
            "Single-threaded prover times of vfhe's range proof over Z_p "
            f"({args.bits}-bit pseudo-Mersenne prime), compiled with Basefold, "
            f"on {host.get('cpu', 'an unknown CPU')}, engine {version['engine']}, "
            f"vfhe {version['commit']}"
            f"{' with local changes' if version['dirty'] else ''}. Milliseconds "
            "by ell* (log2 of the digits per range class); see measure_range.py "
            "for each series."
        ),
        "unit": "ms",
        "log_sizes": [],
        "configuration": {
            "prime_bits": args.bits,
            "prime": str(pmf_field(args.bits).prime),
            "log_n": LOG_N,
            "prime_bits_rq": PRIME_BITS,
            "split_degree": SPLIT_DEGREE,
            "chunks": CHUNKS,
            "log_beta": LOG_BETA,
            "sigma": args.sigma,
            "m": args.m,
            "kappa": args.kappa,
            "zerochecks": True,
            "threads": 1,
            "repeat": args.repeat,
            "parameters": {},
        },
        "vfhe": version,
        "machine": host,
        "runs": {},
    }
    for ell_star in sizes:
        print(f"== 2^{ell_star} digits", file=sys.stderr, flush=True)
        run = subprocess.run(
            [
                sys.executable,
                *(["-s"] if sys.flags.no_user_site else []),
                __file__,
                *("--worker", str(ell_star), "--bits", str(args.bits)),
                *("--sigma", str(args.sigma), "--m", str(args.m)),
                *("--kappa", str(args.kappa), "--repeat", str(args.repeat)),
            ],
            stdout=subprocess.PIPE,
            text=True,
        )
        if run.returncode != 0:
            data["runs"][str(ell_star)] = {"error": f"exit {run.returncode}"}
            print(f"   failed: exit {run.returncode}", file=sys.stderr)
        else:
            report = json.loads(run.stdout)
            data["runs"][str(ell_star)] = {
                "samples": report["samples"],
                "max_rss_mib": report["max_rss_mib"],
            }
            data["configuration"]["parameters"][str(ell_star)] = report["parameters"]
            data["log_sizes"].append(ell_star)
            for series in (*SERIES, "prove", "verify"):
                if series in report["times"]:
                    data.setdefault(series, {})[str(ell_star)] = round(
                        report["times"][series], 1
                    )
            print(
                "   " + ", ".join(f"{k} {v:.0f}" for k, v in report["times"].items()),
                file=sys.stderr,
            )
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(data, indent=2) + "\n")
    print(out)


if __name__ == "__main__":
    main()
