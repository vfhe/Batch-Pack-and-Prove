"""Basefold parameters and proof-size estimate.

For a polynomial of ``variables`` variables over a field of ``log_field``
bits, search the base code (dimension ``k0``, length ``n0``) and the gap
that defines ``gamma``, and return the parameters of the smallest proof:
the number of query repetitions comes from the Johnson bound applied to
the improved minimum-distance estimate of the folded code, and the cost is
a proof-size formula in bits.

Usage: python3.12 basefold_parameters.py VARIABLES [LOG_FIELD]
"""

from __future__ import annotations

import math
import sys
from dataclasses import dataclass

SECURITY = 128
LOG_FIELD = 45 * 4  # a residue field F_(p^4) of R_q with 45-bit primes
OPENINGS = 1  # M, the number of batched openings
KAPPA = 256  # bits of a hash
MAX_LOG_K0 = 8


@dataclass(frozen=True)
class Parameters:
    """``k0``, ``n0``: the base code; ``folds = variables - log2 k0``;
    ``repetitions``: the queries; ``delta``: the proximity parameter;
    ``gamma``, ``gap``: the batching parameter and the gap that set it;
    ``distance``: the relative minimum distance of the last code;
    ``bits``: the proof-size estimate."""

    k0: int
    n0: int
    folds: int
    repetitions: int
    delta: float
    gamma: float
    gap: float
    distance: float
    bits: float


def distance(k0: int, folds: int, c: float, lam: int, log_field: int) -> float:
    """Relative minimum distance of the code after ``folds`` foldings of a
    base code of dimension ``k0`` and length ``c * k0``."""
    n = c * k0
    t: float = k0
    for _ in range(folds):
        n *= 2
        t = 2 * t + (lam + 2.002 * t + 0.585 * n) / (log_field - 1.001)
    return 1 - t / n


def johnson(gamma: float, x: float) -> float:
    return 1 - math.sqrt(1 - x * (1 - gamma))


def estimate(
    variables: int,
    log_field: int = LOG_FIELD,
    security: int = SECURITY,
    openings: int = OPENINGS,
    kappa: int = KAPPA,
) -> Parameters:
    """The parameters with the smallest proof-size estimate."""
    best: Parameters | None = None
    for i in range(4, 16):
        gap = 10 ** (-i / 2)
        for log_k0 in range(MAX_LOG_K0 + 1):
            for log_n0 in range(max(2, log_k0 + 1), log_k0 + 3):
                k0, n0 = 2**log_k0, 2**log_n0
                d = variables - log_k0
                reldis = distance(k0, d, n0 / k0, security, log_field)
                gamma = (2 * d / (2 ** (log_field - security) - openings + 1)) ** (
                    1 / 3
                ) + gap
                delta = min(
                    (reldis + d * gamma) / 3,
                    johnson(gamma, johnson(gamma, reldis)),
                )
                base = 1 - delta + d * gamma
                term = (
                    2.0**-security
                    - (openings - 1) / 2**log_field
                    - 2 * d / (gamma**3 * 2**log_field)
                )
                if not (0 < base < 1 and term > 0):
                    continue
                reps = math.ceil(math.log(term) / math.log(base))
                if reps <= 0:
                    continue
                bits = (
                    3 * (d - 1) * log_field
                    + k0 * log_field
                    + 2 * reps * openings * log_field
                    + 2 * reps * (d - 1) * log_field
                    + 2 * reps * kappa * log_n0 * (openings + 1)
                    + 2 * reps * openings * d * kappa
                    + kappa * reps * (d - 1) * d
                )
                if best is None or bits < best.bits:
                    best = Parameters(k0, n0, d, reps, delta, gamma, gap, reldis, bits)
    if best is None:
        raise ValueError(f"no parameters for {variables} variables")
    return best


if __name__ == "__main__":
    v = int(sys.argv[1])
    f = int(sys.argv[2]) if len(sys.argv) > 2 else LOG_FIELD
    p = estimate(v, f)
    print(
        f"k0=2^{p.k0.bit_length() - 1} n0=2^{p.n0.bit_length() - 1} "
        f"folds={p.folds} repetitions={p.repetitions} "
        f"size={p.bits / 8e6:.3f} MB"
    )
