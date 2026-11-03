"""The tables of [CCCFGS26], printed as the paper prints them.

Usage: python3.12 tables.py [--microbenchmarks FILE] [--range FILE]

``--microbenchmarks`` is a ``measure.py`` file and ``--range`` a
``measure_range.py`` one; both default to the paper's measurements in
``microbenchmarks/`` (an empty ``--range`` prints no range table). A cell
whose microbenchmark is not in the file prints as ``-``.
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from decimal import ROUND_HALF_UP, Decimal

import estimates
from estimates import Circuit, Microbenchmarks, ZpRange

#: The paper's three networks, ``(n0, n1)``.
MODELS = ((128, 64), (64, 16), (32, 16))


#: A cell whose microbenchmark was not measured.
MISSING = "-"


def fmt_time(seconds: float | None) -> str:
    """``5m54s``, ``45s``, ``1m``: minutes, then seconds rounded half up;
    `MISSING` for None or NaN."""
    if seconds is None or math.isnan(seconds):
        return MISSING
    minutes, rest = divmod(math.floor(seconds + 0.5), 60)
    if minutes == 0:
        return f"{rest}s"
    return f"{minutes}m" if rest == 0 else f"{minutes}m{rest}s"


def fmt_decimal(x: float, digits: int = 1) -> str:
    """``x`` with ``digits`` decimals, rounded half up as the paper's
    figures are (``1062.25`` is ``1062.3``)."""
    return str(Decimal(repr(x)).quantize(Decimal(10) ** -digits, ROUND_HALF_UP))


def fmt_size(mib: float) -> str:
    """``867.9 MiB``, ``3.8 GiB``, ``24.7 KiB``."""
    if mib >= 1024:
        return f"{fmt_decimal(mib / 1024)} GiB"
    if mib < 1:
        return f"{fmt_decimal(mib * 1024)} KiB"
    return f"{fmt_decimal(mib)} MiB"


@dataclass(frozen=True)
class Model:
    """Everything the paper reports for one network."""

    n0: int
    n1: int
    circuit: Circuit
    cells: dict[str, int]
    gkr: float
    sparsepack: estimates.SparsePackCost
    spark: estimates.SparkCost


def model(
    n0: int, n1: int, bench: Microbenchmarks, circuit: Circuit | None = None
) -> Model:
    """The estimates for ``(n0, n1)``, from ``circuit`` (the closed form
    when None)."""
    if circuit is None:
        circuit = Circuit.network(n0, n1)
    ell, m = estimates.sparse_parameters(circuit)
    return Model(
        n0,
        n1,
        circuit,
        estimates.circuit_cells(circuit),
        estimates.gkr_time(circuit, bench),
        estimates.sparsepack(ell, m, bench),
        estimates.spark(ell, m, bench),
    )


def res_all_row(m: Model) -> list[str]:
    """A row of Table ``tab:res_all``."""
    s, p = m.spark, m.sparsepack
    return [
        fmt_size(s.sumcheck.size),
        fmt_time(s.sumcheck.time),
        fmt_time(s.commit_time),
        fmt_size(s.eval.size),
        fmt_time(s.eval.time),
        fmt_size(p.sumcheck.size),
        fmt_time(p.sumcheck.time),
        fmt_size(p.eval.size),
        fmt_time(p.eval.time),
    ]


RANGE_COLUMNS = ("SumCheck", "Com", "Eval", "Other", "Total")


def _print_table(title: str, header: list[str], rows: list[list[str]]) -> None:
    widths = [max(len(r[i]) for r in [header, *rows]) for i in range(len(header))]
    print(title)
    for r in [header, *rows]:
        print("  " + "  ".join(c.rjust(w) for c, w in zip(r, widths, strict=True)))
    print()


def print_zp_range(path: str) -> None:
    """The range proofs over ``Z_p`` measured in ``path``: one proof by
    ``ell*``, then each model's four (Table ``tab:res_range_proof``)."""
    zp = ZpRange.load(path)
    c = zp.configuration
    sigma = int(c["sigma"]).bit_length() - 1
    title = (
        f"{c['prime_bits']}-bit p, sigma = 2^{sigma}, m = {c['m']}, "
        f"kappa = {c['kappa']}"
    )
    _print_table(
        f"Range proof over Z_p, one proof by ell* ({title})",
        ["ell*", *RANGE_COLUMNS],
        [
            [str(v), *(fmt_time(row[k]) for k in RANGE_COLUMNS)]
            for v, row in sorted(zp.proofs.items())
        ],
    )
    rows = []
    for n0, n1 in MODELS:
        try:
            times = estimates.zp_range_times(n0, n1, zp)
        except estimates.MissingBenchmark:
            times = None
        rows.append(
            [
                f"({n0}, {n1})",
                *(fmt_time(None if times is None else times[k]) for k in RANGE_COLUMNS),
            ]
        )
    _print_table(
        "Table res_range_proof: range proofs over Z_p (measured)",
        ["model", *RANGE_COLUMNS],
        rows,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    parser.add_argument("--microbenchmarks", default=estimates.DEFAULT_MICROBENCHMARKS)
    parser.add_argument("--range", default=estimates.DEFAULT_RANGE, metavar="FILE")
    args = parser.parse_args()
    bench = Microbenchmarks.load(args.microbenchmarks, missing_as_nan=True)

    points = estimates.pc_sizes(bench)
    _print_table(
        "Fig. pc: Basefold and BatchFold over R_q (ms, MiB)",
        [
            "log2 size",
            "commit",
            "Basefold eval",
            "BatchFold eval",
            "Basefold",
            "BatchFold",
        ],
        [
            [
                str(p.variables),
                f"{p.commit:.0f}",
                f"{p.basefold_eval:.0f}",
                f"{p.batchfold_eval:.0f}",
                fmt_decimal(p.basefold_size),
                fmt_decimal(p.batchfold_size),
            ]
            for p in points
        ],
    )
    r = estimates.pc_ratios(points)
    print(
        f"  eval up to {r['eval_speedup']:.1f}x faster, proofs up to "
        f"{r['size_reduction']:.1f}x smaller, {r['brakedown']:.0f}x over Brakedown\n"
    )

    models = [model(n0, n1, bench) for n0, n1 in MODELS]
    names = [f"({m.n0}, {m.n1})" for m in models]
    keys = list(models[0].cells)
    _print_table(
        "Table full-results: circuit sizes",
        ["model", *keys],
        [
            [n, *(str(m.cells[k]) for k in keys)]
            for n, m in zip(names, models, strict=True)
        ],
    )
    print(
        "GKR: "
        + ", ".join(MISSING if math.isnan(m.gkr) else f"{m.gkr:.1f} s" for m in models)
        + "\n"
    )
    _print_table(
        "Table res_all: Spark | SparsePack",
        [
            "model",
            "SC size",
            "SC time",
            "Comm.",
            "Eval size",
            "Eval time",
            "SC size",
            "SC time",
            "Eval size",
            "Eval time",
        ],
        [[n, *res_all_row(m)] for n, m in zip(names, models, strict=True)],
    )
    if args.range:
        print_zp_range(args.range)


if __name__ == "__main__":
    main()
