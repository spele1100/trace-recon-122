# trace-recon-122

C++17 + pybind11 implementation of the uniform quasipolynomial-time trace
reconstruction decoder from OpenAI's /math release
("Uniform quasipolynomial-time-trace-reconstruction", paper #122 of our
survey — see `papers/math-survey-2026-10.md`). Given independent deletion traces of an unknown binary string
(each position retained independently with probability p), it recovers the
string **exactly** by greedy prefix extension with a moment-relaxation
feasibility test per bit.

Input format is deliberately simple for the nVLC cross-trace merge use
case: a list of binary trace strings. Pure computation — no vision or
hardware dependencies.

## Build

```bash
sudo apt-get install -y libeigen3-dev          # once
pip install pybind11
cmake -S . -B build -Dpybind11_DIR=$(python3 -m pybind11 --cmakedir) -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
```

## Use (Python)

```python
import sys; sys.path.insert(0, "build")
import tr122

x = "1011010010110100"                      # ground truth (unknown in practice)
traces = tr122.simulate_traces(x, p=0.7, T=20000, seed=1)
res = tr122.decode(traces, n=len(x), p=0.7, D=2, zeta=0.03)
res["out"]        # decoded string
res["bits"]       # per-bit feasibility debug (accept/violation/eigenvalue)
```

API: `simulate_traces(x, p, T, seed)`, `rank_means(traces, n, D)`,
`prefix_feasible(traces, n, p, prefix, D, zeta, ...)` (single-prefix test),
`decode(traces, n, p, D, zeta, ...)`.

- `D` = max rank-set size (paper: D ≳ Y(n,p)²; empirically D=1 or 2).
- `zeta` = statistical slack (0.02–0.05 works; must exceed estimation noise
  and stay below the wrong-prefix separation margin).
- Generic moment system supports any D but n ≤ 64; D=1 supports n ≤ 512
  via a specialized path.

## Tests

```bash
./build/tr122_selftest          # C++ oracle + numerics
python3 -m pytest tests -q      # incl. numpy/scipy reference cross-check
```

## Benchmarks

```bash
python3 bench/bench.py          # writes bench/results.csv
```

Frontier (3 random strings per cell, exact reconstruction; full data in
`bench/results.csv`, 325 rows):

D=1, zeta=0.02 — 100%-exact region (T >= 10000):

| retention p | max n (100% exact) | notes |
|---|---|---|
| 0.95 | 64 | n=96: 2/3, n=128: 1/3 at T=50k; n>=192 fails |
| 0.90 | 48 | n=64: 2/3; n>=96 fails (tail separation collapse) |
| 0.80 | 32 | n>=48 fails |
| 0.70 | 24 | marginal even there |
| 0.60 | 16 | n>=24 fails |

D=2, zeta=0.03 — T=50000: p=0.8 and 0.7 exact at n=32 (2/2); p=0.6
marginal (0-1/2); p=0.5 fails. D=2 clearly extends the frontier vs D=1
(e.g. n=32, p=0.7: D=1 1/3 -> D=2 2/2) exactly as the paper's
D ~ Y(n,p)^2 = (1 + ln n / (1 + ln 1/q))^2 scaling predicts.

Runtime: D=1 ~1-25 s per decode (n=16..256); D=2 ~1-2 min per decode at
n=32 (tail exhaustive search is exact; head relies on annealing).

## Layout

- `src/channel.hpp` — deletion-channel simulator
- `src/rank_stats.hpp` — rank observables, empirical means, exact channel
  coefficient polynomial (eq. from `decoding.tex`)
- `src/moment_generic.hpp` — moment system + feasibility, any D, n ≤ 64
- `src/moment_d1.hpp` — D=1 specialized path, n ≤ 512
- `src/feas_solver.hpp` — preconditioned FISTA + coordinate descent +
  Chebyshev reweighting for the affine feasibility subproblem
- `src/decoder.hpp` — greedy prefix decoder (paper rule incl. 0-fallback)
- `tests/` — C++ selftest, pytest oracle suite, independent Python
  reference decoder
- `bench/` — frontier benchmarks → CSV

Honest substitutions of the paper's exact-rational SDP feasibility search
are documented in SPEC.md:
- **FISTA affine solver** instead of exact-rational SDP feasibility
  (preconditioned FISTA + coordinate descent + Chebyshev reweighting);
- **D=1 LP-slab** specialized path (n ≤ 512) instead of the generic
  moment system;
- **D≥2 restricted slack-family search** for the tail exhaustive step
  (head relies on annealing) instead of the full SDP search.

## nVLC cross-trace merge use-case
Target application: merging ~10⁴ independently corrupted copies of the
same transmitted packet (deletion channel, p ≈ 0.96). Verdict from the
frontier data: **offline batch only, not realtime** — p=0.95 n≤64 sits
inside the D=1 100%-exact zone, but each decode takes seconds-to-minutes
and the trace budget (T ≥ 10k per packet) rules out streaming use.

---
Part of the paper-to-artifact series: reverse-matching → exact-dft →
scs-2approx → editdist-121 → trace-recon-122.
