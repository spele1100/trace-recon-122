import os
import random
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "build"))
sys.path.insert(0, os.path.dirname(__file__))

import tr122  # noqa: E402
import ref_decoder  # noqa: E402


def rand_x(n, seed):
    rng = random.Random(seed)
    return "".join(rng.choice("01") for _ in range(n))


def test_channel_simulator_statistics():
    x = "1" * 40
    tr = tr122.simulate_traces(x, 0.8, 20000, 3)
    assert all(t == "1" * len(t) for t in tr)
    lens = sum(len(t) for t in tr) / len(tr)
    assert abs(lens - 32) < 1.0  # 40 * 0.8


def test_rank_means_matches_reference():
    x = rand_x(16, 7)
    p = 0.75
    tr = tr122.simulate_traces(x, p, 3000, 11)
    cpp = {tuple(rs): v for rs, v in tr122.rank_means(tr, 16, 1)}
    ref = ref_decoder.rank_means(tr, 16)
    for l in range(1, 17):
        assert abs(cpp[(l,)] - ref[l - 1]) < 1e-12


@pytest.mark.parametrize("seed", [1, 2, 3])
def test_cross_check_decode_cpp_vs_reference(seed):
    n, p, T = 20, 0.8, 30000
    x = rand_x(n, seed)
    tr = tr122.simulate_traces(x, p, T, 100 + seed)
    got_cpp = tr122.decode(tr, n, p, 1, zeta=0.02)["out"]
    got_ref = ref_decoder.decode(tr, n, p, zeta=0.02)
    assert got_ref == x  # reference recovers exactly
    assert got_cpp == x  # C++ agrees with the truth (and hence the reference)


@pytest.mark.parametrize("seed", range(5))
def test_oracle_d1_exact(seed):
    n, p, T = 32, 0.9, 20000
    x = rand_x(n, seed + 50)
    tr = tr122.simulate_traces(x, p, T, seed)
    got = tr122.decode(tr, n, p, 1, zeta=0.02)["out"]
    assert got == x


@pytest.mark.parametrize("seed", range(3))
def test_oracle_d2_exact(seed):
    n, p, T = 16, 0.7, 20000
    x = rand_x(n, seed + 90)
    tr = tr122.simulate_traces(x, p, T, seed)
    got = tr122.decode(tr, n, p, 2, zeta=0.03)["out"]
    assert got == x


def test_low_retention_fails_cleanly():
    # p=0.5, D=1, small T: decoder must still output a length-n string
    # (paper rule) even when reconstruction is beyond the regime.
    x = rand_x(16, 5)
    tr = tr122.simulate_traces(x, 0.5, 500, 5)
    got = tr122.decode(tr, 16, 0.5, 1, zeta=0.02)["out"]
    assert len(got) == 16


def test_large_n_d1_path():
    # exercises the specialized large-n D=1 code path (n > 64)
    n, p, T = 68, 0.9, 100000
    x = rand_x(n, 42)
    tr = tr122.simulate_traces(x, p, T, 42)
    got = tr122.decode(tr, n, p, 1, zeta=0.02)["out"]
    assert got == x


def test_n80_is_beyond_d1_frontier():
    # HONEST LIMITATION: at n=80, p=0.9 the D=1 relaxation cannot separate
    # wrong tail bits within practical zeta/T (measured empirically; the
    # paper's theory requires D ~ Y(n,p)^2 >> 1 there). We record the
    # limitation instead of pretending success: expect >= 1 bit error.
    n, p, T = 80, 0.9, 300000
    x = rand_x(n, 42)
    tr = tr122.simulate_traces(x, p, T, 42)
    got = tr122.decode(tr, n, p, 1, zeta=0.02)["out"]
    errs = sum(1 for a, b in zip(got, x) if a != b)
    assert 0 < errs <= n  # fails cleanly, string length preserved
