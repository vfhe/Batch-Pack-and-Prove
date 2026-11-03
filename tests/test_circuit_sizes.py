"""The sizes of the circuit of a small network against the closed forms:
k inputs -> FC(n0) -> x^2 -> FC(n1) -> x^2 -> FC(2). Needs vfhe."""

import pytest

pytest.importorskip("vfhe.vfhe.computation")

from vfhe.arith import Ring
from vfhe.fhe import CKKS_Scheme
from vfhe.util import trace
from vfhe.vfhe import cccfgs26, witness
from vfhe.vfhe.computation import Computation

import circuit_sizes

N = 256


def test_the_sizes_of_a_small_network_follow_the_closed_forms():
    # k inputs -> FC(n0) -> x^2 -> FC(n1) -> x^2 -> FC(2), without rotations.
    k, n0, n1 = 3, 4, 2
    scheme = CKKS_Scheme(
        Ring(N, prime_size=[45, 45, 45], split_degree=1), scaling_factor=2**40
    )
    key = scheme.key_gen_sparse(N // 8, 3.2)
    secret = key.poly[0]
    scheme.rlk = scheme.gen_rlk(key, [-(secret * secret)])
    xs = [scheme.encrypt(scheme.encode([0.5]), key) for _ in range(k)]

    def weights(rows, columns, lvl):
        ring = scheme.rings[lvl]
        return [
            [scheme.encode([1.0], ring=ring, scale=1) for _ in range(columns)]
            for _ in range(rows)
        ]

    w1, w2, w3 = weights(n0, k, 0), weights(n1, n0, 1), weights(2, n1, 2)
    with trace.recording() as t:
        h = scheme.linear_combinations(xs, w1)
        h = [scheme.rescale(scheme.multiply(y, y, scheme.rlk)) for y in h]
        h = scheme.linear_combinations(h, w2)
        h = [scheme.rescale(scheme.multiply(y, y, scheme.rlk)) for y in h]
        scheme.linear_combinations(h, w3)
    computation = Computation(t.operations())
    sizes = circuit_sizes.sizes(computation)
    assert sizes.xmult == {"c,c": 4 * (n0 + n1)}
    assert sizes.det == k * n0 + n0 * n1 + 2 * n1
    assert sizes.copycon == {"i": 2 * k * n0, "q": 2 * (n0 * n1 + 2 * n1)}
    assert sizes.copy == 2 * (n0 + n1 + 2)
    assert sizes.multiplications == {1: n1, 2: n0}
    assert (sizes.depth, sizes.D, sizes.delta, sizes.M, sizes.mu) == (2, 3, 2, 4, 5)
    assert sizes.inputs == k
    assert sizes.oracles["V_3,BD"] == circuit_sizes.Oracle(3 * n0 + 2 * n1, 6)
    assert sizes.oracles["V_3,rc3"] == circuit_sizes.Oracle(4 * (n0 + n1), 6)
    assert sizes.predicates["copycon"] == circuit_sizes.Predicate(
        2 * sizes.det, 4 + 5 + 6
    )
    primes = scheme.ring.primes
    classes = {(c.name, c.level): c for c in sizes.range_classes}
    assert sorted(classes) == [
        ("w_ks", 1),
        ("w_ks", 2),
        ("w_quot", 1),
        ("w_quot", 2),
        ("w_rmd", 1),
        ("w_rmd", 2),
    ]
    assert classes["w_ks", 2].count == 3 * n0
    assert classes["w_quot", 2].bound == primes[0] * primes[1]
    assert (classes["w_rmd", 1].count, classes["w_rmd", 1].variables) == (2 * n1, 2)
    assert classes["w_ks", 1].oracle == "V_3,BD" and classes["w_ks", 1].padded == 6
    w = witness.Witness(computation, cccfgs26)
    w.check()
    digits = witness.coefficients(w, "w_ks", ell=3)
    assert len(digits) == 3 * n0 and all(len(v) == N for v in digits)
    assert digits[0] == w.query("w_ks")[0].value.get_polynomial(signed=True)
    assert len(witness.coefficients(w, "w_ks")) == 3 * n0 + 2 * n1
