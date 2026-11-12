# Batch-Pack-and-Prove

Reproduction of the experimental results of the full version of *Batch,
Pack and Prove* (ASIACRYPT 2026), on top of the
[vfhe](https://github.com/vfhe) library.

How the paper's numbers are obtained:

- the range proofs over `Z_p` are measured end to end with vfhe: the whole
  single-threaded prover, compiled with Basefold, hashing and Fiat-Shamir
  included;
- Basefold and BatchFold (Fig. `fig:results:pc`), the GKR evaluation and
  SparsePack against Spark (Table `tab:res_all`) are operation counts
  times four single-threaded vfhe microbenchmarks over `R_q` (Basefold
  commit and evaluation, BatchFold evaluation and a degree-2 sumcheck,
  Merkle hashing included), with proof sizes from formulas;
- the circuit sizes (Table `tab:full-results`) are closed forms, checked
  against vfhe's execution trace of the three networks.

## What reproduces what

| Paper | Code | Test |
|---|---|---|
| Fig. `fig:results:pc` (Basefold vs BatchFold), the 2.8x / 9.5x / 302x claims | `estimates.pc_sizes`, `estimates.pc_ratios` | `tests/test_estimates.py` |
| Table `tab:full-results` (circuit sizes) | `estimates.circuit_cells`; traced: `networks.py`, `circuit_sizes.py` | `tests/test_estimates.py`, `tests/test_networks.py` |
| GKR evaluation, 21.9 / 5.9 / 3.0 s | `estimates.gkr_time` | `tests/test_estimates.py` |
| Table `tab:res_all` (SparsePack vs Spark) | `estimates.sparsepack`, `estimates.spark` | `tests/test_estimates.py` |
| Table `tab:res_range_proof` (range proofs over `Z_p`) | `measure_range.py`, `estimates.ZpRange`, `estimates.zp_range_times` | `tests/test_estimates.py` |
| Basefold parameters (repetitions, base code) | `basefold_parameters.py` | `tests/test_basefold_parameters.py` |

## Running

Python 3.11 or later, with pytest for the tests. Printing the tables needs
nothing else:

```bash
python3.12 tables.py
```

prints every table as the paper prints them, from the paper's measurements
in `microbenchmarks/`. `--microbenchmarks FILE` (a `measure.py` file) and
`--range FILE` (a `measure_range.py` file) take other measurements.

Everything else needs vfhe at the commit the paper's numbers come from,
`reproduce.VFHE_COMMIT`. `setup.sh` clones it into `vfhe/`, builds it, and
creates a virtual environment in `venv/` whose Python imports it (with
pytest). It needs git, make, a C compiler and Python 3.11 or later with its
`venv` module and headers (Debian/Ubuntu: `build-essential python3-venv
python3-dev`); `PYTHON=python3.12` picks the interpreter:

```bash
bash setup.sh
venv/bin/python networks.py 128 64 --log-n 6
venv/bin/python -m pytest
```

A vfhe checkout of your own works too: build it with `make build` and put
`build`, `build/proto` and each `modules/*/python/src` on `PYTHONPATH`.

`networks.py` defaults to the paper's parameters (`N = 2^13`, three 45-bit
primes, splitting degree 4, Gaussian secret with `sigma = 3.2`). The counts
do not depend on the ring degree, so `--log-n 6` gives the same circuit in
a fraction of the time and memory. The tests that need vfhe are skipped
without it.

## Reproducing the measurements

`reproduce.py` runs every measurement the tables use with vfhe, one after
the other and single-threaded, writes the results into `microbenchmarks/`
and prints the tables from them:

1. `measure.py`: Basefold commit and evaluation, BatchFold evaluation and a
   degree-2 sumcheck over `R_q` at the paper's ring, `2^8 .. 2^16`;
2. `measure_range.py`: the range proof over `Z_p` (208-bit `p`, `sigma =
   2^11`, `m = 15`, `kappa = 20`), compiled with Basefold, at `ell* = 20 ..
   24`.

Every proof is verified. The script stops on a vfhe commit other than
`reproduce.VFHE_COMMIT` (or on local changes) unless given `--any-commit`.
The files in `microbenchmarks/` record vfhe `83a59f7`, a commit with the
same library code as `VFHE_COMMIT` (the two differ only in CI files and
development tool versions).
After `setup.sh`:

```bash
venv/bin/python reproduce.py --vfhe vfhe
```

The paper's numbers were measured on an AWS `r7i.metal-48xl` (Intel Xeon
Platinum 8488C) with vfhe's `avx512ifma` engine, which vfhe picks by itself
on a CPU with AVX-512 IFMA. The machine needs **at least 384 GiB of
memory**: the `R_q` commitment at `2^16` peaks near 285 GiB (the range
proof stays below 60 GiB). The run takes about an hour and a half there,
70 minutes of it for step 1. Each command is printed before it runs, so
one step can be run again by hand.

`--smoke` runs the same pipeline at tiny sizes into `results/smoke/` in a
few minutes and a few GiB: every proof still verifies and the tables print,
with `-` for the cells whose sizes were not measured. Its times mean
nothing.

## Notes on the model

- Two charges are operation counts, not measurements: building the table
  `E` of `Pi_ueval` (`estimates.ueval_e_table_time`), and adding one more
  polynomial to a batched Basefold opening in Spark
  (`estimates._BATCHED_OPENING_COMBINATION`).
- The challenges come from the diagonal copy of `H_q` in `R_q`, so
  SparsePack's product trees run over `H_q`; Spark is costed as [Setty20]
  specifies it, over `R_q` as [CCCFGS25] commits (no batching across the
  `R_q` components).

## Files

| File | Content |
|---|---|
| `estimates.py` | the cost model |
| `microbenchmarks/vfhe-83a59f7-r7i.json` | the `R_q` microbenchmarks the tables use (`measure.py`) |
| `microbenchmarks/vfhe-83a59f7-r7i-range208-sigma2048-m15-kappa20.json` | the range proof over `Z_p` (`measure_range.py`) |
| `measure.py` | measures the microbenchmarks with vfhe |
| `measure_range.py` | measures the range proof over `Z_p` with vfhe |
| `reproduce.py` | runs every measurement, then prints the tables |
| `setup.sh` | installs vfhe at `reproduce.VFHE_COMMIT` for the rest |
| `basefold_parameters.py` | Basefold parameter search and proof-size estimate |
| `tables.py` | the paper's tables, printed |
| `circuit_sizes.py` | the circuit of Sec. 5 for a traced computation (needs vfhe) |
| `networks.py` | the three networks, evaluated and traced (needs vfhe) |
| `tests/` | the paper's numbers as tests |

## License

Apache-2.0, see `LICENSE`.
