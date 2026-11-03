"""The circuit of [CCCFGS26, Sec. 5] for a traced computation: its gate
counts, the sizes of its oracles (Table 1), the sparse predicates and the
range-check classes, in the paper's names.

It reads vfhe's public API only: `vfhe.vfhe.computation.Computation`
(``linears``, ``multiplications``, ``rescales``, ``inputs``, ``origin``,
``written_by``, each step's ``ell`` and ``layer``) and `vfhe.vfhe.witness`
(``Witness``, ``range_classes``).

A plaintext-linear step is a ``copycon`` gate. The paper's level of a step
over ``ell`` primes is ``l = ell - 1``, and ``D`` is one more than the
highest level, so that levels ``0..D-1`` index the ``l + 1`` primes of a
level. ``M`` is the most multiplications at one level, as the text
defines it. For a class's
values as coefficient vectors, use `vfhe.vfhe.witness.coefficients` with
``ell = level + 1``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from vfhe.vfhe import cccfgs26, witness
from vfhe.vfhe.computation import Computation, Division, Linear

import estimates

if TYPE_CHECKING:
    from collections.abc import Iterable

__all__ = [
    "Oracle",
    "Predicate",
    "RangeClass",
    "Sizes",
    "circuit",
    "sizes",
    "wire_class",
]

#: The wire classes of a ciphertext by the step that wrote it: a rescale's
#: quotient, a copy.
_CLASSES: dict[type, str] = {Division: "q", Linear: "c"}

#: The order of the classes in a predicate's name; ``x`` is a ciphertext the
#: circuit has no wire for, such as a key switch's output not rescaled.
_ORDER = "iqcx"

#: The AC oracle each range-checked witness lives in.
_ORACLE_OF = {"w_ks": "V_3,BD", "w_quot": "V_3,rc3", "w_rmd": "V_3,rc3"}


@dataclass
class Oracle:
    """An oracle of the circuit: ``dense`` values, laid out over
    ``variables`` variables."""

    dense: int
    variables: int


@dataclass
class Predicate:
    """A sparse wiring predicate: ``nonzeros`` points of a hypercube of
    ``variables`` variables."""

    nonzeros: int
    variables: int


@dataclass
class RangeClass:
    """A range class of `vfhe.vfhe.witness.range_classes` in the paper's
    terms: the witnesses named ``name`` at level ``level = ell - 1``,
    ``count`` values of `R_q`, each within ``bound``. ``variables`` is
    ``ceil(log2 count)``, the size of the values concatenated (the dense
    count the range-proof estimates use); ``oracle`` names the AC oracle
    holding them and ``padded`` its number of variables, both None for a
    name the circuit has no oracle for."""

    name: str
    level: int
    count: int
    bound: int
    variables: int
    oracle: str | None
    padded: int | None


@dataclass
class Sizes:
    """The circuit of [CCCFGS26] for one computation.

    ``multiplications`` counts the multiplications by level, ``depth`` is
    the longest chain of them; ``D``, ``delta = ceil(log2 D)``, ``M`` (the
    most multiplications at one level) and ``mu = ceil(log2 2M) + delta``
    are Table 1's. ``inputs`` counts the input ciphertexts the circuit
    reads. ``xmult`` counts the products of the ``Xmult`` gates by the
    classes of their operands (``"c,c"``: two copies; ``i``, ``q``, ``c``
    for an input, a quotient, a copy, ``x`` for none of them), four per
    multiplication; ``copycon`` the nonzeros of the ``copycon`` predicate
    by the class of the ciphertext a term multiplies, two per term. ``det``
    is the number of distinct plaintexts, ``V_3,det``, a ciphertext added
    as is counting the plaintext 1; ``copy`` the number of copied
    components, ``V_3,copy``. ``oracles`` are the AC oracles of Table 1
    (and ``V_in``, ``V_3,copy``, ``V_3,ptxt``, as dense as they come),
    ``predicates`` the sparse ones, ``Xmult_(a,b)`` and ``copycon``.
    """

    multiplications: dict[int, int]
    depth: int
    D: int
    delta: int
    M: int
    mu: int
    inputs: int
    xmult: dict[str, int]
    copycon: dict[str, int]
    det: int
    copy: int
    oracles: dict[str, Oracle]
    predicates: dict[str, Predicate]
    range_classes: list[RangeClass]


def _bits(n: int) -> int:
    """``ceil(log2 n)``, 0 for ``n <= 1``."""
    return max(n - 1, 0).bit_length()


def wire_class(computation: Computation, versions: Iterable[int]) -> str:
    """The wire class of the ciphertext whose components are ``versions``:
    ``"i"`` (an input of the computation), ``"q"`` (a rescale's quotient),
    ``"c"`` (a plaintext-linear step's copy) or ``"x"`` (anything else)."""

    def of(version: int) -> str:
        if computation.origin.get(version, version) in computation.inputs:
            return "i"
        return _CLASSES.get(type(computation.written_by(version)), "x")

    found = {of(v) for v in versions}
    return found.pop() if len(found) == 1 else "x"


def sizes(computation: Computation) -> Sizes:
    """The gate counts, oracle sizes, sparse predicates and range classes
    of the circuit for ``computation``.

    The variables follow Table 1 and the layer equations of Sec. 5: the
    ``Xmult`` predicates take ``V_2``'s ``mu + 1`` and twice
    ``s' = max(#in, mu, #copy)``. ``copycon`` is nonzero only at the gate
    type ``T = (0,1)``: it is ``eq(T, (0,1)) * copycon'(Z, X, Y)``, and the
    verifier evaluates the first factor itself. The predicate given is
    ``copycon'``, over ``#copy + m + m'`` variables, ``m = max(2 delta, d)``
    and ``m' = 1 + max(mu, #in)``, with ``d``, ``#in`` and ``#copy`` the
    variables of ``V_3,ptxt``, ``V_in`` and ``V_3,copy``.
    """
    multiplications = computation.multiplications
    by_level: dict[int, int] = {}
    for m in multiplications:
        by_level[m.ell - 1] = by_level.get(m.ell - 1, 0) + 1
    steps = [*multiplications, *computation.rescales]
    D = max((s.ell for s in steps), default=1)
    delta = _bits(D)
    M = max(by_level.values(), default=0)
    mu = _bits(2 * M) + delta
    inputs: set[tuple[int, ...]] = set()
    xmult: dict[str, int] = {}
    for m in multiplications:
        classes = sorted(
            (wire_class(computation, v) for v in m.versions), key=_ORDER.index
        )
        name = ",".join(classes)
        xmult[name] = xmult.get(name, 0) + 4
        inputs |= {tuple(v) for v in m.versions if wire_class(computation, v) == "i"}
    copycon: dict[str, int] = {}
    plaintexts: set[int | None] = set()
    for step in computation.linears:
        for t in step.terms:
            kind = wire_class(computation, t.versions)
            copycon[kind] = copycon.get(kind, 0) + 2
            plaintexts.add(t.plaintext_version)
            if kind == "i":
                inputs.add(tuple(t.versions))
    det = len(plaintexts)
    copy = 2 * len(computation.linears)
    n_in, n_copy, d = _bits(2 * len(inputs)), _bits(copy), _bits(det)
    count = len(multiplications)
    digits = sum(len(m.key_switch.digits) for m in multiplications)
    rescales = len(computation.rescales)  # one per component
    oracles = {
        "V_in": Oracle(2 * len(inputs), n_in),
        "V_1": Oracle(2 * count, mu),
        "V_2,d01": Oracle(2 * count, mu),
        "V_2,d2": Oracle(count, mu - 1),
        "V_2": Oracle(3 * count, mu + 1),
        "V_3,BD": Oracle(digits, mu - 1 + delta),
        "V_3,quot": Oracle(rescales, mu),
        "V_3,rc3": Oracle(2 * rescales, mu + 1),
        "V_3,copy": Oracle(copy, n_copy),
        "V_3,ptxt": Oracle(det, d),
    }
    s_prime = max(n_in, mu, n_copy)
    predicates = {
        f"Xmult_({name})": Predicate(n, mu + 1 + 2 * s_prime)
        for name, n in xmult.items()
    }
    predicates["copycon"] = Predicate(
        sum(copycon.values()),
        n_copy + max(2 * delta, d) + 1 + max(mu, n_in),
    )
    classes = []
    for c in witness.range_classes(witness.Witness(computation, cccfgs26)):
        oracle = _ORACLE_OF.get(c.name)
        classes.append(
            RangeClass(
                c.name,
                c.ell - 1,
                c.count,
                c.bound,
                _bits(c.count),
                oracle,
                None if oracle is None else oracles[oracle].variables,
            )
        )
    return Sizes(
        multiplications=dict(sorted(by_level.items())),
        depth=max((m.layer + 1 for m in multiplications), default=0),
        D=D,
        delta=delta,
        M=M,
        mu=mu,
        inputs=len(inputs),
        xmult=xmult,
        copycon=copycon,
        det=det,
        copy=copy,
        oracles=oracles,
        predicates=predicates,
        range_classes=classes,
    )


def circuit(sizes: Sizes) -> estimates.Circuit:
    """The counts of ``sizes`` that the cost model reads. Its ``copycon``
    terms must read inputs or quotients only."""
    others = set(sizes.copycon) - {"i", "q"}
    if others:
        raise ValueError(f"copycon terms on wires of class {sorted(others)}")
    return estimates.Circuit(
        multiplications=dict(sizes.multiplications),
        copycon_inputs=sizes.copycon.get("i", 0),
        copycon_quotients=sizes.copycon.get("q", 0),
        det=sizes.det,
        copy=sizes.copy,
        D=sizes.D,
        inputs=sizes.inputs,
    )
