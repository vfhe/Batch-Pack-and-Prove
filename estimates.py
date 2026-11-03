"""The cost model behind the tables of [CCCFGS26] (Batch, Pack and Prove).

Every time but the range proofs' is an operation count times
single-threaded microbenchmarks, and every proof size a formula. This
module holds the counts and the formulas; the microbenchmarks are data
(`Microbenchmarks.load`), measured with vfhe by ``measure.py``. The range
proofs over ``Z_p`` are measured end to end by ``measure_range.py``.

Sizes and times:

- `pc_sizes`, `pc_ratios`: Basefold and BatchFold over ``R_q``
  (Fig. ``fig:results:pc``);
- `circuit_cells`: the circuit sizes of Table ``tab:full-results``;
- `gkr_time`: the GKR evaluation of the arithmetic circuit;
- `sparsepack`, `spark`: Table ``tab:res_all``;
- `ZpRange`, `zp_range_times`: Table ``tab:res_range_proof``, from
  ``measure_range.py``'s measurements.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from functools import cache
from pathlib import Path

import basefold_parameters

#: Ring degree ``N = 2^nu``, primes ``L + 1``, splitting degree ``d``.
NU = 13
N = 2**NU
PRIMES = 3
LOG_PRIME = 45
LOG_Q = LOG_PRIME * PRIMES
SPLIT_DEGREE = 4

MIB = 2**20

#: Bytes of an element as the proof-size formulas store it: an ``R_q``
#: element (``N`` coefficients of 32 bytes), a hash, a point of the
#: residue field ``F_(p^4)`` (64 bits per coordinate).
RQ_BYTES = N * 32
HASH_BYTES = 32
FIELD_BYTES = 64 * 4 // 8

#: Bits of an ``R_q`` element in the sumcheck communication counts, and of
#: an ``H_q`` element (``d`` coefficients); ``R_q`` holds ``2^HQ_SHIFT``
#: times as many coefficients.
RQ_BITS = N * LOG_Q
HQ_BITS = SPLIT_DEGREE * LOG_Q
HQ_SHIFT = (N // SPLIT_DEGREE).bit_length() - 1

MICROBENCHMARKS = Path(__file__).with_name("microbenchmarks")
DEFAULT_MICROBENCHMARKS = MICROBENCHMARKS / "vfhe-83a59f7-r7i.json"
DEFAULT_RANGE = MICROBENCHMARKS / "vfhe-83a59f7-r7i-range208-sigma2048-m15-kappa20.json"


def clog2(x: float) -> int:
    """``ceil(log2 x)``, 0 for ``x <= 1``."""
    return 0 if x <= 1 else math.ceil(math.log2(x))


def rq_elements(count: float) -> float:
    """MiB of ``count`` ``R_q`` elements of ``N log q`` bits."""
    return count * RQ_BITS / 8 / MIB


def hq_elements(count: float) -> float:
    """MiB of ``count`` ``H_q`` elements of ``d log q`` bits."""
    return count * HQ_BITS / 8 / MIB


# --- Microbenchmarks --------------------------------------------------------


class MissingBenchmark(ValueError):
    """A size the estimate needs is not in the microbenchmark file."""


@dataclass(frozen=True)
class Microbenchmarks:
    """Single-threaded times in milliseconds by ``log2`` of the input size in
    ``R_q`` elements: Basefold ``commit`` and ``basefold_eval``, BatchFold's
    evaluation at a diagonal point (``batchfold_batching``), and a degree-2
    ``sumcheck`` over ``R_q``. The methods return seconds; a size that was
    not measured raises `MissingBenchmark`, or with ``missing_as_nan`` is a
    NaN, which every estimate built on it carries."""

    commit_ms: dict[int, float]
    basefold_eval_ms: dict[int, float]
    batchfold_batching_ms: dict[int, float]
    sumcheck_ms: dict[int, float]
    missing_as_nan: bool = False

    @classmethod
    def load(
        cls, path: str | Path = DEFAULT_MICROBENCHMARKS, missing_as_nan: bool = False
    ) -> Microbenchmarks:
        """The file at ``path``, as ``measure.py`` writes it.

        ``measure.py`` times its sumcheck's statement digest apart
        (``details.sumcheck_statement_digest``): one hash of both tables that
        its ``sumcheck`` includes and the model's sumchecks, over virtual or
        committed oracles, do not compute. It is subtracted from ``sumcheck``
        at each size that has it."""
        data = json.loads(Path(path).read_text())
        sizes = data["log_sizes"]
        digest = data.get("details", {}).get("sumcheck_statement_digest", {})
        for i, v in enumerate(sizes):
            data["sumcheck"][i] -= digest.get(str(v), 0)

        def table(key: str) -> dict[int, float]:
            return dict(zip(sizes, map(float, data[key]), strict=True))

        return cls(
            table("commit"),
            table("basefold_eval"),
            table("batchfold_batching"),
            table("sumcheck"),
            missing_as_nan,
        )

    def _at(self, table: dict[int, float], log_size: int, what: str) -> float:
        if log_size not in table:
            if self.missing_as_nan:
                return math.nan
            raise MissingBenchmark(f"no {what} benchmark at 2^{log_size}")
        return table[log_size] / 1000

    def commit(self, log_size: int) -> float:
        return self._at(self.commit_ms, log_size, "commit")

    def basefold_eval(self, log_size: int) -> float:
        return self._at(self.basefold_eval_ms, log_size, "Basefold eval")

    def batchfold_batching(self, log_size: int) -> float:
        """BatchFold's evaluation proof at a diagonal point (one residue-field
        point per prime, as the series is measured), which needs no
        common-point sumcheck."""
        return self._at(self.batchfold_batching_ms, log_size, "BatchFold")

    def batchfold_eval(self, log_size: int) -> float:
        """BatchFold's evaluation proof at a general point: its batching,
        then one ``R_q`` sumcheck of the same size."""
        return self.batchfold_batching(log_size) + self.sumcheck(log_size)

    def sumcheck(self, log_size: int) -> float:
        """A degree-2 sumcheck over ``2^log_size`` ``R_q`` elements; past the
        largest benchmark it doubles per variable."""
        top = max(self.sumcheck_ms)
        if log_size > top:
            return self._at(self.sumcheck_ms, top, "sumcheck") * 2 ** (log_size - top)
        return self._at(self.sumcheck_ms, log_size, "sumcheck")

    def hq_sumcheck(self, log_size: int) -> float:
        """A degree-2 sumcheck over ``2^log_size`` ``H_q`` elements, charged
        as one over the ``R_q`` elements holding as many coefficients, at
        least the smallest benchmark."""
        return self.sumcheck(max(log_size - HQ_SHIFT, min(self.sumcheck_ms)))


# --- Polynomial commitments (Fig. pc) ---------------------------------------


@cache
def pc_parameters(variables: int) -> basefold_parameters.Parameters:
    return basefold_parameters.estimate(variables)


def basefold_size(variables: int) -> float:
    """MiB of a Basefold evaluation proof over ``R_q``: two points per
    query and fold, three sumcheck points per fold, the last codeword, and
    a Merkle path per query and fold."""
    p = pc_parameters(variables)
    folds, reps = p.folds, p.repetitions
    size = (
        2 * reps * folds * RQ_BYTES
        + 3 * folds * RQ_BYTES
        + p.n0 * RQ_BYTES
        + HASH_BYTES * (variables + 2) * folds * reps
    )
    return size / MIB


def batchfold_size(variables: int) -> float:
    """MiB of a BatchFold evaluation proof: two ``R_q`` points per query on
    the first fold, the batching and folding sumchecks and the last
    codeword over the three residue fields, field points per query on the
    other folds, and the Merkle paths."""
    p = pc_parameters(variables)
    folds, reps = p.folds, p.repetitions
    log_components = clog2(N // SPLIT_DEGREE)
    size = (
        2 * reps * RQ_BYTES
        + 3 * PRIMES * FIELD_BYTES * (variables + 2 + log_components)
        + 3 * PRIMES * FIELD_BYTES * folds
        + PRIMES * p.n0 * FIELD_BYTES
        + PRIMES * 2 * reps * folds * FIELD_BYTES
        + HASH_BYTES * variables * folds * reps
    )
    return size / MIB


@dataclass(frozen=True)
class PCPoint:
    """One size of Fig. ``fig:results:pc``: times in ms, proof sizes in
    MiB."""

    variables: int
    commit: float
    basefold_eval: float
    batchfold_eval: float
    basefold_size: float
    batchfold_size: float


def pc_sizes(bench: Microbenchmarks) -> list[PCPoint]:
    return [
        PCPoint(
            v,
            bench.commit(v) * 1000,
            bench.basefold_eval(v) * 1000,
            bench.batchfold_eval(v) * 1000,
            basefold_size(v),
            batchfold_size(v),
        )
        for v in sorted(bench.commit_ms)
    ]


#: Brakedown in [CCCFGS25]: seconds per CRT component at ``2^16``.
BRAKEDOWN_2_16 = 8.5


def pc_ratios(points: list[PCPoint]) -> dict[str, float]:
    """The claims of Sec. ``sec:batchfold``: the best evaluation speedup and
    proof-size reduction of BatchFold over Basefold, and the gain over
    Brakedown (per CRT component, ``N/4`` of them) at the largest size."""
    last = points[-1]
    return {
        "eval_speedup": max(p.basefold_eval / p.batchfold_eval for p in points),
        "size_reduction": max(p.basefold_size / p.batchfold_size for p in points),
        "brakedown": BRAKEDOWN_2_16
        * (N // SPLIT_DEGREE)
        / (last.batchfold_eval / 1000),
    }


# --- The arithmetic circuit (Table tab:full-results) -------------------------


@dataclass(frozen=True)
class Circuit:
    """The counts of an HE computation that the estimates use.

    ``multiplications`` maps a level to its ciphertext multiplications;
    ``copycon_inputs`` and ``copycon_quotients`` are the nonzeros of
    ``copycon`` whose term reads an input or a rescale's quotient; ``det``
    is ``V_3,det``, ``copy`` is ``V_3,copy``, ``D`` the levels and
    ``inputs`` the input ciphertexts.
    """

    multiplications: dict[int, int]
    copycon_inputs: int
    copycon_quotients: int
    det: int
    copy: int
    D: int
    inputs: int

    @classmethod
    def network(cls, n0: int, n1: int, inputs: int = 30, outputs: int = 2) -> Circuit:
        """``inputs -> FC(n0) -> x^2 -> FC(n1) -> x^2 -> FC(outputs)`` in
        closed form. The first layer multiplies at the top level ``2``, the
        second at level ``1``."""
        return cls(
            multiplications={2: n0, 1: n1},
            copycon_inputs=2 * inputs * n0,
            copycon_quotients=2 * (n0 * n1 + outputs * n1),
            det=inputs * n0 + n0 * n1 + outputs * n1,
            copy=2 * (n0 + n1 + outputs),
            D=3,
            inputs=inputs,
        )

    @property
    def M(self) -> int:
        """The most multiplications at one level."""
        return max(self.multiplications.values())

    @property
    def delta(self) -> int:
        return clog2(self.D)

    @property
    def mu(self) -> int:
        """``mu = ceil(log2 2M) + delta``."""
        return clog2(2 * self.M) + self.delta


def circuit_cells(circuit: Circuit) -> dict[str, int]:
    """The row of Table ``tab:full-results``. The copycon variables are those
    of the committed ``copycon'``, ``#copy + m + m'`` with ``m = max(2 delta,
    d)`` and ``m' = 1 + max(mu, #in)``, where ``d``, ``#in`` and ``#copy`` are
    the variables of ``V_3,det``, ``V_in`` and ``V_3,copy``: ``copycon =
    eq(T, (0,1)) copycon'``, and the verifier evaluates the first factor
    itself."""
    mu, delta = circuit.mu, circuit.delta
    xmult = 4 * sum(circuit.multiplications.values())
    copycon = circuit.copycon_inputs + circuit.copycon_quotients
    return {
        "Xmult": xmult,
        "copycon": copycon,
        "V_3,det": circuit.det,
        "V_3,copy": circuit.copy,
        "log D": delta,
        "V_3,BD": delta + mu - 1,
        "V_3,quot": mu,
        "Xmult nonzeros": clog2(xmult),
        "copycon nonzeros": clog2(copycon),
        "copycon variables": clog2(circuit.copy)
        + max(2 * delta, clog2(circuit.det))
        + 1
        + max(mu, clog2(2 * circuit.inputs)),
    }


def gkr_time(circuit: Circuit, bench: Microbenchmarks) -> float:
    """Seconds of the GKR evaluation of the circuit: two ``R_q`` sumchecks
    over ``m' = 1 + max(mu, #in)`` variables, three over ``m = max(2 delta,
    d)`` and one over ``s' = ceil(log2 V_3,copy)``. A sumcheck below the
    smallest benchmark, ``2^8``, is charged as one of ``2^8``."""
    n_in = clog2(2 * circuit.inputs)
    m_prime = 1 + max(circuit.mu, n_in)
    m = max(2 * circuit.delta, clog2(circuit.det))
    s_prime = clog2(circuit.copy)
    return sum(
        count * bench.sumcheck(max(v, 8))
        for count, v in ((2, m_prime), (3, m), (1, s_prime))
    )


# --- Sparse polynomials (Table tab:res_all) ----------------------------------


@dataclass(frozen=True)
class Cost:
    """Proof size (MiB) and prover time (seconds)."""

    size: float
    time: float


@dataclass(frozen=True)
class SparsePackCost:
    sumcheck: Cost
    eval: Cost
    variables: int  # of the packed polynomial, at least 8


@dataclass(frozen=True)
class SparkCost:
    sumcheck: Cost
    commit_time: float
    eval: Cost


def sparse_parameters(circuit: Circuit) -> tuple[int, int]:
    """``(ell, m)`` of the sparse polynomials of a circuit: ``2^ell``
    nonzeros (those of ``copycon``) over ``m`` variables (``copycon``'s)."""
    cells = circuit_cells(circuit)
    return cells["copycon nonzeros"], cells["copycon variables"]


def _iptree_degree(eta: int, mu: int) -> int:
    """The degree of the ``eta``-th product-tree sumcheck of ``Pi_iptree``
    (App. ``app:sparse_perf``)."""
    return 4 if eta == 0 else 5 if eta == mu - 2 else 3


def ueval_e_table_time(n: int, repetitions: int, bench: Microbenchmarks) -> float:
    """Seconds to build ``E_(sigma,tau)`` for ``Pi_ueval`` with challenges
    from the diagonal copy of ``H_q`` in ``R_q``.

    The table has ``2^(n + nu)`` entries. Per repetition each entry costs
    ``a (L+1) d`` multiplications (``a`` points, ``L+1`` primes, a
    ``d``-term linear form per prime); once per point and prime, the table
    ``eq(b, x)`` over ``F_(p^d)`` costs ``d`` times as much. One repetition
    is charged what one repetition's sumcheck costs (the same operation
    count, not measured), so the build is ``repetitions + d`` sumchecks
    over ``2^n`` ``R_q`` elements; the number of points scales the sumcheck
    and the table alike.
    """
    return (repetitions + SPLIT_DEGREE) * bench.sumcheck(n)


def sparsepack(
    ell: int, m: int, bench: Microbenchmarks, a: int = 2, kappa: int = 3
) -> SparsePackCost:
    """SparsePack for ``a`` sparse polynomials with ``2^ell`` nonzeros over
    ``m`` variables (App. ``app:sparse_perf``), with challenges from the
    diagonal copy of ``H_q`` in ``R_q``.

    The sumcheck column is ``Pi_iptree``: ``mu - 1`` sumchecks over
    ``2^(ell + alpha + eta)`` entries, each charged ``degree / 2`` degree-2
    sumchecks, and ``4 ell (mu-1) + (4 mu - 1) alpha + 2 mu^2 - 2 mu + 2``
    elements of communication. The points the circuit's predicates are
    evaluated at, and SparsePack's own challenges, are diagonal, so every
    table of ``Pi_iptree`` holds ``H_q`` values: its sumchecks run over
    ``H_q`` and send ``H_q`` elements.

    The eval column is the BatchFold proof of the packed polynomial, ``n =
    ell + alpha + mu + 2 - nu`` variables (at least 8), at a diagonal point
    (no common-point sumcheck), and ``Pi_ueval`` repeated ``kappa (n +
    nu)`` times, each repetition one ``R_q`` sumcheck over ``2^n``, plus
    building its table ``E`` and one more sumcheck batching the claims.
    """
    alpha, mu = clog2(a), clog2(m)
    n = max(ell + alpha + mu + 2 - NU, 8)
    elements = 4 * ell * (mu - 1) + (4 * mu - 1) * alpha + 2 * mu**2 - 2 * mu + 2
    iptree_time = sum(
        bench.hq_sumcheck(ell + alpha + eta) * _iptree_degree(eta, mu) / 2
        for eta in range(mu - 1)
    )
    repetitions = kappa * (n + NU)
    ueval_time = repetitions * bench.sumcheck(n) + ueval_e_table_time(
        n, repetitions, bench
    )
    ueval_size = 3 * kappa * (n + alpha + mu) ** 2 * LOG_Q / 8 / MIB
    batching_size = rq_elements(3 * n)
    return SparsePackCost(
        sumcheck=Cost(hq_elements(elements), iptree_time),
        eval=Cost(
            batchfold_size(n) + ueval_size + batching_size,
            bench.batchfold_batching(n) + ueval_time + bench.sumcheck(n),
        ),
        variables=n,
    )


#: Adding one committed polynomial to a batched Basefold opening: its
#: rate-1/4 codeword enters the random linear combination, ``4 x 2^ell``
#: ``R_q`` multiply-adds, charged as this fraction of a degree-2 sumcheck
#: over ``2^ell`` (about ``5 x 2^ell`` products over all rounds). An
#: operation count, not a measurement.
_BATCHED_OPENING_COMBINATION = 0.8


def _first_layer_size(variables: int) -> float:
    """MiB one more committed polynomial adds to a batched Basefold
    opening: its two codeword entries and a Merkle path per query on the
    first layer."""
    p = pc_parameters(variables)
    return p.repetitions * (2 * RQ_BYTES + HASH_BYTES * (variables + 2)) / MIB


def spark(ell: int, m: int, bench: Microbenchmarks, a: int = 2) -> SparkCost:
    """Spark for the same polynomials, ``c = ceil(m / ell)``, as [Setty20,
    §7] specifies it, over ``R_q`` as [CCCFGS25] commits (no batching across
    the ``R_q`` components), with the counters committed in preprocessing.

    The product trees are one data-parallel layered circuit, one sumcheck
    per layer for all of them: ``(c+2) ell + 2(ceil(log c) + alpha + 1) ell
    (ell - 1) + 2 ell`` ``R_q`` elements, and the prover's work of ``(c + 1)
    + 4ac`` sumchecks over ``2^ell``. The prover commits ``ac`` polynomials
    of ``2^ell``. The ``4ac`` polynomials are queried at two points,
    ``r_ops`` and ``r_mem``, and opened by one batched plain-Basefold proof
    each: a random linear combination of the committed codewords, one
    Basefold run, and the first-layer answers of every polynomial."""
    c = math.ceil(m / ell)
    evals = 4 * a * c
    elements = (c + 2) * ell + 2 * (clog2(c) + clog2(a) + 1) * ell * (ell - 1) + 2 * ell
    added = evals - 2
    return SparkCost(
        sumcheck=Cost(
            rq_elements(elements), ((c + 1) + 4 * c * a) * bench.sumcheck(ell)
        ),
        commit_time=a * c * bench.commit(ell),
        eval=Cost(
            2 * basefold_size(ell) + added * _first_layer_size(ell),
            2 * bench.basefold_eval(ell)
            + added * _BATCHED_OPENING_COMBINATION * bench.sumcheck(ell),
        ),
    )


# --- Range proofs over Z_p, measured (Table tab:res_range_proof) -------------


def range_classes(
    n0: int, n1: int, log_bounds: tuple[int, int, int] = (45, 90, 45)
) -> list[tuple[int, int, int]]:
    """The four range proofs of a two-layer network, as ``(ell, gamma,
    table bits)``: the key-switch digits (``3 n0 + 2 n1`` values), the
    rescale remainders (``2(n0 + n1)``), and the quotients of each layer
    (``2 n0``, ``2 n1``), ``2^ell`` of them after padding. ``log_bounds``
    are the bits of the digits and remainders, of the first-layer quotients
    and of the second-layer ones. Each proof splits a coefficient into
    ``c = 2^gamma`` chunks of ``b / c`` bits, ``c = 4`` (``8`` for the
    first-layer quotients), so it looks up ``2^(ell + gamma + nu)`` chunks
    in a table of ``2^(b / c)``."""
    L = PRIMES
    b1, b2, b3 = log_bounds
    c, c_quot = 4, 8
    gamma, gamma_quot = clog2(c), clog2(c_quot)
    return [
        (clog2(L * n0 + (L - 1) * n1), gamma, b1 // c),
        (clog2(2 * (n0 + n1)), gamma, b1 // c),
        (clog2(2 * n0), gamma_quot, b2 // c_quot),
        (clog2(2 * n1), gamma, b3 // c),
    ]


@dataclass(frozen=True)
class ZpRange:
    """The range proof over ``Z_p`` as ``measure_range.py`` times it: the
    prover's seconds by ``ell*`` (``2^ell*`` chunks), in the columns of
    Table ``tab:res_range_proof``:

    - ``SumCheck``: the lookup's two zerochecks;
    - ``Com``: the commitments of its seven oracles;
    - ``Eval``: the openings of its claims (the batched Basefold openings,
      the virtual oracles' and ``v``'s BatchFold opening);
    - ``Other``: the rest of the prover (the chunks, the ``kappa`` partial
      evaluations, the counters and the fingerprint tables);
    - ``Total``: the whole prover.

    ``configuration`` is the file's (the prime, ``sigma``, ``m``,
    ``kappa``)."""

    proofs: dict[int, dict[str, float]]
    configuration: dict

    @classmethod
    def load(cls, path: str | Path = DEFAULT_RANGE) -> ZpRange:
        data = json.loads(Path(path).read_text())
        if not data["configuration"].get("zerochecks"):
            raise ValueError(
                f"{path}: measured without the zerochecks, so the prover is incomplete"
            )

        def ms(series: str, v: str) -> float:
            return data.get(series, {}).get(v, 0.0) / 1000

        proofs = {}
        for v in map(str, data["log_sizes"]):
            row = {
                "SumCheck": ms("zerochecks", v),
                "Com": ms("commit", v),
                "Eval": ms("eval", v) + ms("virtual_eval", v) + ms("v_eval", v),
                "Total": ms("prove", v),
            }
            row["Other"] = row["Total"] - row["SumCheck"] - row["Com"] - row["Eval"]
            proofs[int(v)] = row
        return cls(proofs, data["configuration"])

    def proof(self, ell_star: int) -> dict[str, float]:
        if ell_star not in self.proofs:
            raise MissingBenchmark(f"no range proof measured at ell* = {ell_star}")
        return self.proofs[ell_star]


def zp_range_times(n0: int, n1: int, measured: ZpRange) -> dict[str, float]:
    """Seconds of the four range proofs of `range_classes` over ``Z_p``,
    by column: each is the measured proof with the same ``ell* = ell + gamma
    + nu``. The measurement splits a coefficient into ``c = 4`` chunks of
    ``2^11``; the first-layer quotients split into 8, which looks up as many
    chunks in the same table, with one variable fewer in ``v``."""
    out: dict[str, float] = dict.fromkeys(
        ("SumCheck", "Com", "Eval", "Other", "Total"), 0.0
    )
    for ell, gamma, _ in range_classes(n0, n1):
        for column, seconds in measured.proof(ell + gamma + NU).items():
            out[column] += seconds
    return out
