"""The paper's three networks, traced with vfhe on a small ring, against the
closed forms and Table tab:full-results. Needs vfhe."""

import pytest

pytest.importorskip("vfhe.vfhe.computation")

import circuit_sizes
import estimates
import networks
from tests.test_estimates import FULL_RESULTS


@pytest.fixture(scope="module", params=list(FULL_RESULTS), ids=str)
def traced(request):
    n0, n1 = request.param
    sizes = circuit_sizes.sizes(networks.evaluate(n0, n1, log_n=6))
    return (n0, n1), sizes


def test_closed_forms(traced):
    (n0, n1), sizes = traced
    assert networks.traced(sizes) == networks.closed_forms(n0, n1)


def test_full_results(traced):
    model, sizes = traced
    cells = estimates.circuit_cells(circuit_sizes.circuit(sizes))
    assert list(cells.values()) == FULL_RESULTS[model]
    # The copycon variables are those of the traced predicate, copycon'.
    assert cells["copycon variables"] == sizes.predicates["copycon"].variables


def test_the_cost_model_reads_the_traced_circuit(traced):
    # The closed form the tables use is the traced circuit.
    model, sizes = traced
    assert circuit_sizes.circuit(sizes) == estimates.Circuit.network(*model)
