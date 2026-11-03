"""The paper's networks, evaluated in CKKS with vfhe and traced.

``inputs -> FC(n0) -> x^2 -> FC(n1) -> x^2 -> FC(outputs)``, with 30 inputs
and 2 outputs, each input a ciphertext of ``N/2`` slots; no rotations. The
paper's parameters are ``N = 2^13``, three 45-bit primes, splitting degree
4 and a Gaussian secret with ``sigma = 3.2``. The counts do not depend on
``N``, the splitting degree or the values, so a smaller ring gives the
same circuit.

Usage: python3.12 networks.py N0 N1 [--log-n 13] [--split-degree 4] [--check]

Prints the circuit sizes of the traced network, its row of Table
``tab:full-results`` and the closed forms they must satisfy. Needs vfhe on
``PYTHONPATH``.
"""

from __future__ import annotations

import argparse
import random
import time

from vfhe.arith import Polynomial, Ring
from vfhe.fhe import CKKS_Scheme
from vfhe.mlwe import MLWE_Key
from vfhe.util import trace
from vfhe.vfhe import cccfgs26, witness
from vfhe.vfhe.computation import Computation

import circuit_sizes
import estimates

INPUTS = 30
OUTPUTS = 2
LOG_N = 13
SPLIT_DEGREE = 4
PRIME_BITS = [45, 45, 45]
SCALE = 2**40
SIGMA = 3.2


def gaussian_key(scheme: CKKS_Scheme, sigma: float = SIGMA) -> MLWE_Key:
    """A secret key with discrete-Gaussian coefficients of width ``sigma``,
    and error of the same width."""
    s = Polynomial(scheme.rings[0]).sample_gaussian(sigma, ntt=False)
    return MLWE_Key([s.get_polynomial(signed=True)], sigma, scheme)


def evaluate(
    n0: int,
    n1: int,
    log_n: int = LOG_N,
    split_degree: int = SPLIT_DEGREE,
    inputs: int = INPUTS,
    outputs: int = OUTPUTS,
    seed: int = 1,
) -> Computation:
    """Encrypt ``inputs`` random ciphertexts, evaluate the network on them
    with small integer weights, and return the traced computation."""
    N = 1 << log_n
    scheme = CKKS_Scheme(
        Ring(N, prime_size=PRIME_BITS, split_degree=split_degree), scaling_factor=SCALE
    )
    key = gaussian_key(scheme)
    s = key.poly[0]
    scheme.rlk = scheme.gen_rlk(key, [-(s * s)])
    rng = random.Random(seed)
    slots = N // 2
    xs = [
        scheme.encrypt(scheme.encode([rng.uniform(-1, 1)] * slots), key)
        for _ in range(inputs)
    ]

    def weights(rows: int, columns: int, lvl: int) -> list[list]:
        ring = scheme.rings[lvl]
        return [
            [
                scheme.encode([rng.choice([1, 2, 3])], ring=ring, scale=1)
                for _ in range(columns)
            ]
            for _ in range(rows)
        ]

    w1, w2, w3 = weights(n0, inputs, 0), weights(n1, n0, 1), weights(outputs, n1, 2)
    with trace.recording() as t:
        h = scheme.linear_combinations(xs, w1)
        h = [scheme.rescale(scheme.multiply(y, y, scheme.rlk)) for y in h]
        h = scheme.linear_combinations(h, w2)
        h = [scheme.rescale(scheme.multiply(y, y, scheme.rlk)) for y in h]
        scheme.linear_combinations(h, w3)
    return Computation(t.operations())


def closed_forms(
    n0: int, n1: int, inputs: int = INPUTS, outputs: int = OUTPUTS
) -> dict:
    """What the traced sizes must be."""
    return {
        "Xmult": {"c,c": 4 * (n0 + n1)},
        "copycon": {"i": 2 * inputs * n0, "q": 2 * (n0 * n1 + outputs * n1)},
        "V_3,det": inputs * n0 + n0 * n1 + outputs * n1,
        "V_3,copy": 2 * (n0 + n1 + outputs),
        "multiplications": {1: n1, 2: n0},
        "D": 3,
        "inputs": inputs,
    }


def traced(sizes: circuit_sizes.Sizes) -> dict:
    """The quantities of `closed_forms`, as traced."""
    return {
        "Xmult": sizes.xmult,
        "copycon": sizes.copycon,
        "V_3,det": sizes.det,
        "V_3,copy": sizes.copy,
        "multiplications": sizes.multiplications,
        "D": sizes.D,
        "inputs": sizes.inputs,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    parser.add_argument("n0", type=int)
    parser.add_argument("n1", type=int)
    parser.add_argument("--log-n", type=int, default=LOG_N)
    parser.add_argument("--split-degree", type=int, default=SPLIT_DEGREE)
    parser.add_argument("--check", action="store_true", help="check the witness")
    args = parser.parse_args()
    n0, n1 = args.n0, args.n1

    t0 = time.time()
    computation = evaluate(n0, n1, args.log_n, args.split_degree)
    sizes = circuit_sizes.sizes(computation)
    print(
        f"({n0}, {n1}), N = 2^{args.log_n}, split degree {args.split_degree}: "
        f"traced and counted in {time.time() - t0:.0f} s"
    )
    if computation.unread:
        print(f"  not read as steps: {computation.unread}")

    ok = True
    got, expected = traced(sizes), closed_forms(n0, n1)
    for name in expected:
        match = got[name] == expected[name]
        ok &= match
        verdict = "ok" if match else f"expected {expected[name]}"
        print(f"  {name:16} {got[name]!s:28} {verdict}")
    print(f"  D = {sizes.D}, delta = {sizes.delta}, M = {sizes.M}, mu = {sizes.mu}")
    for name, o in sizes.oracles.items():
        print(f"  {name:9} {o.dense:6} values, {o.variables} variables")
    for name, p in sizes.predicates.items():
        print(f"  {name:12} {p.nonzeros:6} nonzeros, {p.variables} variables")
    for r in sizes.range_classes:
        print(
            f"  range {r.name:7} l = {r.level}: {r.count:5} values below "
            f"2^{r.bound.bit_length() - 1}, {r.variables} dense variables, "
            f"{r.padded} in {r.oracle}"
        )

    cells = estimates.circuit_cells(circuit_sizes.circuit(sizes))
    print("  tab:full-results: " + ", ".join(f"{k} {v}" for k, v in cells.items()))

    if args.check:
        w = witness.Witness(computation, cccfgs26)
        w.check()
        print("  the witness checks")
    print("  closed forms hold" if ok else "  CLOSED FORMS FAIL")
    if not ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
