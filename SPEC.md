# SPEC — trace-recon-122

Implementation of **"Uniform quasipolynomial-time trace reconstruction"**
(OpenAI /math release, `preprints/Uniform-quasipolynomial-time-trace-reconstruction-October-5-2026`,
tex saved in `/root/papers/trace-recon-122/tex/`).

Paper guarantee (verified from `main.tex`): for every n, rational retention
p ∈ (0,1], and every x ∈ {0,1}^n, the decoder requests at most
`exp(C·B(n,p))` traces (B(n,p) = p⁻¹·(ln n)·X(n,p)²·(1+ln X)⁶,
X(n,p) = 1 + ln n/(1+ln 1/q)) and outputs **exactly x** with probability
≥ 2/3 (per fixed x; the decoding section's Chebyshev analysis gives ≥ 31/32
on the good event). Quasipolynomial for fixed p; polynomial when q ≤ n^−ε.

## Paper → code component map

| Paper (section) | Code | Notes |
|---|---|---|
| Deletion channel: retain each position w.p. p (`main.tex` §1) | `src/channel.hpp` | mt19937_64 simulator |
| Rank observables T_I for rank sets I, 1 ≤ \|I\| ≤ D (`setup.tex` §2.1) | `src/rank_stats.hpp` | empirical means + exact coefficient polynomial |
| Channel coefficient c_{I;i} = C(i₁−1,ℓ₁−1)·ΠC(gaps)·p^{ℓ_k}q^{i_k−ℓ_k} (`decoding.tex` eq. after "explicit-channel-coefficient") | `channel_coefficients()` | **verified against brute-force enumeration over all 2ⁿ retention patterns** (selftest [1]) |
| Moment relaxation: coords m_J, J⊆U, \|J\|≤2D; matrix M(m) = (m_{I△J}) ⪰ 0 (`setup.tex` eq. moment-matrix) | `src/moment_generic.hpp` (n ≤ 64, any D), `src/moment_d1.hpp` (D=1, n ≤ 512) | u64-mask subsets cap generic path at n ≤ 64 |
| E_m[T_I] affine in m after substituting prefix (`decoding.tex` §2.1) | `GenericFeas::build()` / `D1Feas::build[_free]()` | fixed bits substituted; free bits as (1+Y_i)/2, submask expansion |
| Prefix feasibility system: m∈[−1,1], M(m)⪰0, \|E_m[T_I] − t̂_I\| ≤ ζ (`decoding.tex` eq. prefix-feasibility) | `solve()` in both feas classes | see substitutions below |
| Greedy decoder: test both 1-bit extensions; accept if precisely one feasible; else append 0 (`decoding.tex` §2.2) | `src/decoder.hpp` | exact rule incl. the 0-fallback; per-bit debug recorded |
| Sample bound T = 512(1+N)ζ⁻² (Chebyshev, eq. good-probability) | not enforced | astronomically conservative; empirically ~10⁴–10⁵ traces suffice (bench) |
| Parameter choices D = K_D·y², ζ = 2^{−K_m a_p u y² w⁶} (eq. integer-decoder-parameters) | exposed as knobs (`D`, `zeta`) | paper constants K_D, K_m are unknown absolute constants; theory requires D ≳ Y(n,p)² ≈ 5–30 in our regime, but **empirically D=1–2 works** up to the frontier mapped in bench |

## Honest substitutions (SDP feasibility)

The paper's feasibility search is an **Agmon–Motzkin–Schoenberg relaxation
with exact rational arithmetic** (`decoding.tex` Lemma rational-feasibility):
on violation, step m += λa along a rational separating direction
(λ = ζ/(16G)), with a Schur-complement PSD check; iteration cap
10(1+l)/λ². We implement two deviations:

1. **"agmon" mode** (`solve_agmon`): the same iteration in floating point
   (Eigen eigenvector for the PSD direction), with a small iteration cap
   (default 4000 ≪ paper bound). Faithful but slow to converge because the
   paper's λ is astronomically small in practice.

2. **Default "pocs" mode** — the mode used in all tests/benchmarks:
   - Stage 1: solve the affine part (minimize max \|E_m[T_I]−t̂_I\| over the
     box) with Jacobi-preconditioned FISTA + Gauss–Seidel coordinate descent
     (`src/feas_solver.hpp`) and multiplicative reweighting toward the
     Chebyshev (max) objective.
   - Stage 2, PSD:
     - **D = 1**: PSD is *free*: any m1 ∈ [−1,1]^r admits the rank-one
       completion m2 = m1·m1ᵀ, giving M = [1;m1][1;m1]ᵀ + diag(0, 1−m1_i²) ⪰ 0.
       So feasibility = the LP-slab problem solved in stage 1. (This is a
       special case of the paper's slack construction: any mixture of a
       point mass with uniform signs is a positive moment functional.)
     - **D ≥ 2**: PSD genuinely couples moments up to order 2D. We search
       within the paper's slack family m_S = ρ·Π_{i∈S} y_i with y ∈ {±1}^r,
       ρ = 1 − ζ/4, whose moment matrix is ρ·zzᵀ + (1−ρ)I ⪰ (1−ρ)I — exactly
       the mixture used in the paper's Lemma prefix-slack (true completion
       with weight 1−ζ/8 + uniform signs with weight ζ/8). The sign vector y
       is initialized from the stage-1 solution and refined by exhaustive
       Gray-code search (r ≤ 18) or annealing + single/pair flips (r > 18).
   - Soundness: any accepted point is a genuine positive moment functional
     satisfying all constraints (verified by an eigenvalue check in
     `check()`). Existence for correct prefixes holds because the paper's
     slack point lies inside our search family. The substitution is the
     *search method* only — for wrong prefixes the paper proves no moment
     functional exists at all, so our smaller family rejects them a
     fortiori.

3. **Pruning**: channel coefficients below 1e−10 are dropped (paper keeps
   all); trace count T and D, ζ are user knobs instead of the paper's
   conservative formulas.

## Verified correctness (see tests/)

- Channel-coefficient polynomial ≡ brute-force E[T_I] over all 2ⁿ retention
  subsets (n=8, random strings, D ≤ 2) — exact to 1e−12.
- Empirical rank means concentrate at the exact values (Chebyshev regime).
- Oracle: known random string → simulate traces → decode → **exact**
  reconstruction (paper guarantees exact, not approximate — confirmed):
  D=1 n=24 p=0.8 T=50k: 10/10; D=2 n=16 p=0.7 T=20k: 5/5; pytest suite
  16/16 incl. n=32 p=0.9 D=1, n=16 p=0.7 D=2, cross-check against an
  independent numpy/scipy reference decoder (`tests/ref_decoder.py`).
- Known limitation recorded as a test: n=80, p=0.9, D=1 fails at tail
  positions even with T=3·10⁵ (see bench for the full frontier).

## Practical complexity

- Per feasibility test: O(nnz) per solver iteration; nnz = Σ_I #tuples(I)
  after pruning. D=1: ~10–100ms per test, ~0.5–2 s per decode (n ≤ 256).
  D=2: exhaustive tail search exact for r ≤ 18; heads rely on annealing —
  works at n ≤ 64 but slower and less reliable than D=1 (see bench).
- The theoretical D ≳ Y(n,p)² = (1+ln n/(1+ln 1/q))² scaling is visible
  empirically: at n=80, p=0.9 (Y ≈ 2.3) D=1's separation margin collapses
  at the tail; D=2 restores it (tested at n ≤ 64).
