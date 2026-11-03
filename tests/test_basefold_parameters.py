"""The Basefold parameter search against the parameters of the paper's
proof sizes: the base-code dimension, the folds and the repetitions at 8
to 16 variables over ``F_(p^4)`` with 45-bit primes."""

import pytest

import basefold_parameters

PAPER = {
    # variables: (log2 k0, folds, repetitions)
    8: (7, 1, 312),
    9: (8, 1, 311),
    10: (8, 2, 314),
    11: (8, 3, 316),
    12: (8, 4, 319),
    13: (8, 5, 321),
    14: (8, 6, 324),
    15: (8, 7, 326),
    16: (8, 8, 329),
}


@pytest.mark.parametrize("variables", PAPER)
def test_paper_parameters(variables):
    p = basefold_parameters.estimate(variables)
    assert (p.k0.bit_length() - 1, p.folds, p.repetitions) == PAPER[variables]
    assert p.n0 == 4 * p.k0
