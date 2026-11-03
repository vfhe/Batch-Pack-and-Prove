"""The cost model against the numbers of [CCCFGS26]'s tables, as printed,
from the paper's measurements in ``microbenchmarks/``."""

import json
import math

import pytest

import estimates
import tables
from estimates import Circuit, Microbenchmarks

BENCH = Microbenchmarks.load()
MODELS = {(n0, n1): tables.model(n0, n1, BENCH) for n0, n1 in tables.MODELS}


def approx(x):
    return pytest.approx(x, rel=1e-9, abs=1e-9)


# --- Fig. pc -----------------------------------------------------------------

# Milliseconds and MiB at 8..16 variables, as Fig. pc plots them.
COMMIT = [400, 886, 1999, 4462, 9698, 21401, 45205, 96324, 208248]
BASEFOLD_EVAL = [273, 579, 1761, 4246, 9224, 19220, 39688, 80273, 160148]
BATCHFOLD_EVAL = [291, 522, 992, 1891, 3673, 7249, 14394, 29158, 57685]
BASEFOLD_SIZE = [284.8, 412.4, 571.7, 732.6, 897.5, 1063.0, 1233.4, 1403.4, 1579.4]
BATCHFOLD_SIZE = [156.2, 155.7, 157.4, 158.6, 160.3, 161.5, 163.3, 164.6, 166.4]


def test_fig_pc():
    points = estimates.pc_sizes(BENCH)
    assert [p.variables for p in points] == list(range(8, 17))
    assert [round(p.commit) for p in points] == COMMIT
    assert [round(p.basefold_eval) for p in points] == BASEFOLD_EVAL
    assert [round(p.batchfold_eval) for p in points] == BATCHFOLD_EVAL
    assert [float(tables.fmt_decimal(p.basefold_size)) for p in points] == (
        BASEFOLD_SIZE
    )
    assert [float(tables.fmt_decimal(p.batchfold_size)) for p in points] == (
        BATCHFOLD_SIZE
    )


def test_fig_pc_claims():
    # "up to 2.8x", "up to 9.5x", "up to 302x" over Brakedown, all at 2^16.
    points = estimates.pc_sizes(BENCH)
    r = estimates.pc_ratios(points)
    assert round(r["eval_speedup"], 1) == 2.8
    assert round(r["size_reduction"], 1) == 9.5
    assert round(r["brakedown"]) == 302
    last = points[-1]
    assert r["eval_speedup"] == approx(last.basefold_eval / last.batchfold_eval)
    assert r["brakedown"] == approx(8.5 * 2**11 / (last.batchfold_eval / 1000))


def test_batchfold_eval_adds_a_sumcheck():
    assert BENCH.batchfold_eval(12) == approx(
        BENCH.batchfold_batching(12) + BENCH.sumcheck(12)
    )


# --- Table tab:full-results --------------------------------------------------

# Xmult, copycon, V_3,det, V_3,copy, log D, V_3,BD, V_3,quot, nonzeros of
# Xmult and copycon (log2), copycon variables.
FULL_RESULTS = {
    (128, 64): [768, 24320, 12160, 388, 2, 11, 10, 10, 15, 34],
    (64, 16): [320, 5952, 2976, 164, 2, 10, 9, 9, 13, 30],
    (32, 16): [192, 3008, 1504, 100, 2, 9, 8, 8, 12, 27],
}


@pytest.mark.parametrize("model", FULL_RESULTS)
def test_full_results(model):
    assert list(MODELS[model].cells.values()) == FULL_RESULTS[model]


def test_mu_takes_the_largest_layer():
    # M is the most multiplications at one level (the first layer's n0).
    c = Circuit.network(128, 64)
    assert c.M == 128
    assert c.mu == math.ceil(math.log2(2 * 128)) + 2


# --- GKR ---------------------------------------------------------------------------

GKR = {(128, 64): 21.9, (64, 16): 5.9, (32, 16): 3.0}


@pytest.mark.parametrize("model", GKR)
def test_gkr(model):
    assert round(MODELS[model].gkr, 1) == GKR[model]


def test_gkr_sumchecks():
    # (32, 16): m' = 1 + max(mu, #in) = 9, m = max(2 delta, d) = 11, and
    # s' = ceil(log2 100) = 7, charged at the smallest benchmark, 2^8.
    t = BENCH.sumcheck
    assert MODELS[32, 16].gkr == approx(2 * t(9) + 3 * t(11) + t(8))


# --- Table tab:res_all -------------------------------------------------------

# Spark: SumCheck size and time, Comm., Eval size and time; SparsePack:
# SumCheck size and time, Eval size and time.
RES_ALL = {
    (128, 64): ["235.3 MiB", "6m14s", "9m38s", "6.2 GiB", "6m35s",
                "25.4 KiB", "1s", "163.0 MiB", "2m6s"],
    (64, 16): ["176.5 MiB", "1m33s", "2m8s", "5.5 GiB", "1m37s",
               "17.7 KiB", "1s", "159.4 MiB", "13s"],
    (32, 16): ["150.3 MiB", "47s", "58s", "5.2 GiB", "48s",
               "16.7 KiB", "1s", "159.4 MiB", "13s"],
}  # fmt: skip


@pytest.mark.parametrize("model", RES_ALL)
def test_res_all(model):
    assert tables.res_all_row(MODELS[model]) == RES_ALL[model]


def test_sparse_parameters():
    # (ell, m): copycon's nonzeros and variables; the packed polynomial has
    # ell + alpha + mu + 2 - nu variables, at least 8.
    got = {
        model: (*estimates.sparse_parameters(m.circuit), m.sparsepack.variables)
        for model, m in MODELS.items()
    }
    assert got == {
        (128, 64): (15, 34, 11),
        (64, 16): (13, 30, 8),
        (32, 16): (12, 27, 8),
    }


def test_sparsepack():
    # (128, 64): alpha = 1, mu = 6, n = 11; Pi_iptree's five sumchecks over
    # H_q (degrees 4, 3, 3, 3, 5) and its closed-form communication.
    t, hq = BENCH.sumcheck, BENCH.hq_sumcheck
    s = MODELS[128, 64].sparsepack
    assert s.sumcheck.time == approx(
        (4 * hq(16) + 3 * hq(17) + 3 * hq(18) + 3 * hq(19) + 5 * hq(20)) / 2
    )
    elements = 4 * 15 * 5 + (4 * 6 - 1) * 1 + 2 * 36 - 2 * 6 + 2
    assert s.sumcheck.size == approx(estimates.hq_elements(elements))
    # BatchFold at a diagonal point, Pi_ueval's 3 (11 + 13) repetitions, the
    # table E (as many sumchecks plus d), and the batching sumcheck.
    reps = 3 * (11 + 13)
    assert s.eval.time == approx(
        BENCH.batchfold_batching(11) + (2 * reps + 4) * t(11) + t(11)
    )


def test_spark():
    # (128, 64): c = ceil(34 / 15) = 3, a = 2: six commitments of 2^15, 24
    # polynomials opened by two batched Basefold proofs.
    t = BENCH.sumcheck
    s = MODELS[128, 64].spark
    assert s.commit_time == approx(6 * BENCH.commit(15))
    elements = 5 * 15 + 2 * (2 + 1 + 1) * 15 * 14 + 2 * 15
    assert s.sumcheck.size == approx(estimates.rq_elements(elements))
    assert s.sumcheck.time == approx((4 + 24) * t(15))
    assert s.eval.time == approx(2 * BENCH.basefold_eval(15) + 22 * 0.8 * t(15))


def test_sparsepack_against_spark():
    # The claims of Sec. 7: 41x / 37x / 34x shorter, 10.6x / 23.5x / 11.2x
    # faster (Spark's SumCheck + Comm. + Eval against SparsePack's two).
    shorter, faster = [], []
    for m in MODELS.values():
        s, p = m.spark, m.sparsepack
        shorter.append(
            (s.sumcheck.size + s.eval.size) / (p.sumcheck.size + p.eval.size)
        )
        faster.append(
            (s.sumcheck.time + s.commit_time + s.eval.time)
            / (p.sumcheck.time + p.eval.time)
        )
    assert [round(x) for x in shorter] == [41, 37, 34]
    assert [round(x, 1) for x in faster] == [10.6, 23.5, 11.2]


# --- Table tab:res_range_proof -----------------------------------------------------

# SumCheck, Com, Eval, Other, Total, over Z_p.
RANGE = {
    (128, 64): ["13s", "3m47s", "1m30s", "24s", "5m54s"],
    (64, 16): ["6s", "1m47s", "43s", "11s", "2m47s"],
    (32, 16): ["3s", "52s", "22s", "6s", "1m24s"],
}


@pytest.mark.parametrize("model", RANGE)
def test_range(model):
    n0, n1 = model
    times = estimates.zp_range_times(n0, n1, estimates.ZpRange.load())
    assert [tables.fmt_time(times[k]) for k in tables.RANGE_COLUMNS] == RANGE[model]


def range_file(tmp_path, sizes=(20, 21, 22, 23, 24), zerochecks=True):
    # measure_range.py's format; at 2^v every series is v ms, prove 10 v.
    keys = [str(v) for v in sizes]
    data = {
        "log_sizes": list(sizes),
        "configuration": {
            "prime_bits": 208,
            "sigma": 2048,
            "m": 15,
            "kappa": 20,
            "zerochecks": zerochecks,
        },
    }
    for series in ("zerochecks", "commit", "eval", "virtual_eval", "v_eval"):
        data[series] = {k: float(k) for k in keys}
    data["prove"] = {k: 10.0 * int(k) for k in keys}
    path = tmp_path / "range.json"
    path.write_text(json.dumps(data))
    return path


def test_zp_range(tmp_path):
    zp = estimates.ZpRange.load(range_file(tmp_path))
    assert zp.proof(20) == approx(
        {"SumCheck": 0.02, "Com": 0.02, "Eval": 0.06, "Total": 0.2, "Other": 0.1}
    )
    # (128, 64) has the proofs ell* = 24, 24, 24, 22.
    times = estimates.zp_range_times(128, 64, zp)
    assert times["Total"] == approx((3 * 240 + 220) / 1000)
    assert times["Other"] == approx(times["Total"] / 2)
    assert [e + g + estimates.NU for e, g, _ in estimates.range_classes(64, 16)] == [
        23,
        23,
        23,
        20,
    ]


def test_zp_range_needs_the_zerochecks(tmp_path):
    with pytest.raises(ValueError, match="zerochecks"):
        estimates.ZpRange.load(range_file(tmp_path, zerochecks=False))


# --- Loading and printing ----------------------------------------------------------


def test_statement_digest_left_out(tmp_path):
    # A measured file's sumcheck less its statement digest, at the sizes
    # that list one.
    data = json.loads(estimates.DEFAULT_MICROBENCHMARKS.read_text())
    raw = dict(zip(data["log_sizes"], data["sumcheck"], strict=True))
    data["details"] = {"sumcheck_statement_digest": {"10": 100.0}}
    path = tmp_path / "digest.json"
    path.write_text(json.dumps(data))
    bench = Microbenchmarks.load(path)
    assert bench.sumcheck(10) == approx((raw[10] - 100) / 1000)
    assert bench.sumcheck(11) == approx(raw[11] / 1000)
    assert bench.commit(10) == BENCH.commit(10)


@pytest.mark.parametrize(
    "seconds, text",
    [
        (60.227, "1m"),
        (44.8, "45s"),
        (1256.646, "20m57s"),
        (6.53, "7s"),
        (12.388, "12s"),
        (119.6, "2m"),
        (59.5, "1m"),
    ],
)
def test_fmt_time(seconds, text):
    assert tables.fmt_time(seconds) == text


def test_fmt_size():
    assert tables.fmt_size(0.0241) == "24.7 KiB"
    assert tables.fmt_size(867.94) == "867.9 MiB"
    assert tables.fmt_size(3891.2) == "3.8 GiB"


def test_tables_print_what_was_measured(tmp_path, monkeypatch, capsys):
    # A file without 2^24 leaves the (128, 64) range row unpriced, and
    # microbenchmarks at 2^8 only leave every estimate that needs another
    # size as "-".
    data = json.loads(estimates.DEFAULT_MICROBENCHMARKS.read_text())
    for key in ("commit", "basefold_eval", "batchfold_batching", "sumcheck"):
        data[key] = data[key][:1]
    data["log_sizes"] = [8]
    bench = tmp_path / "bench.json"
    bench.write_text(json.dumps(data))
    zp = range_file(tmp_path, sizes=(20, 22, 23))
    argv = ["tables.py", "--microbenchmarks", str(bench), "--range", str(zp)]
    monkeypatch.setattr("sys.argv", argv)
    tables.main()
    out = capsys.readouterr().out
    assert "Range proof over Z_p" in out
    rows = out.split("Table res_range_proof")[1].splitlines()
    row = next(line for line in rows if "(128, 64)" in line)
    assert row.split()[2:] == ["-"] * 5
    row = next(line for line in rows if "(64, 16)" in line)
    assert "-" not in row.split()[2:]
    with pytest.raises(estimates.MissingBenchmark):
        tables.model(128, 64, estimates.Microbenchmarks.load(bench))


def test_tables_without_range(monkeypatch, capsys):
    monkeypatch.setattr("sys.argv", ["tables.py", "--range", ""])
    tables.main()
    out = capsys.readouterr().out
    assert "Table res_all" in out
    assert "res_range_proof" not in out
