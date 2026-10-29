"""Pure-Python (numpy) reference decoder for cross-checking the C++ library.

Implements the D=1 special case of the paper's decoder:
- rank observables for single ranks, empirical means
- per-prefix feasibility: exists m1 in [-1,1]^r with
      |a0 + A m1 - t_hat| <= zeta   for every rank
  (PSD is free at D=1 via the rank-one completion, see SPEC.md)
  solved with scipy.optimize.lsq_linear (bounded least squares) +
  multiplicative reweighting toward the max violation.
- greedy bit-by-bit extension, appending 0 when both/none accept (paper rule).
"""

import itertools
from math import comb

import numpy as np
from scipy.optimize import lsq_linear


def binom(n, k):
    return comb(n, k) if 0 <= k <= n else 0


def rank_means(traces, n):
    """Empirical mean of the bit at each trace rank 1..n."""
    means = np.zeros(n)
    for tr in traces:
        L = len(tr)
        for l in range(1, min(L, n) + 1):
            means[l - 1] += int(tr[l - 1])
    return means / max(len(traces), 1)


def coeff_rows(n, p, d, u):
    """Affine form of E_m[T_l] for each rank l, given prefix u of length d.

    Returns A (n x r), a0 (n), where r = n - d; E = a0 + A m1.
    """
    q = 1 - p
    r = n - d
    A = np.zeros((n, r))
    a0 = np.zeros(n)
    for l in range(1, n + 1):
        for i in range(n):
            c = binom(i, l - 1) * p**l * q ** (i + 1 - l)
            if c < 1e-12:
                continue
            if i < d:
                if u[i] == 1:
                    a0[l - 1] += c
            else:
                a0[l - 1] += 0.5 * c
                A[l - 1, i - d] = 0.5 * c
    return A, a0


def feasible(A, a0, b, zeta, outer=10):
    """Bounded LS + reweighting to target max |a0 + A m - b| <= zeta."""
    r = A.shape[1]
    if r == 0:
        return float(np.max(np.abs(a0 - b))) <= zeta, float(np.max(np.abs(a0 - b)))
    w = np.ones(len(b))
    best_mv = np.inf
    m = np.zeros(r)
    for _ in range(outer):
        W = w[:, None]
        res = lsq_linear(A * W, (b - a0) * w, bounds=(-1, 1), tol=1e-10, max_iter=200)
        m = res.x
        viol = np.abs(a0 + A @ m - b)
        best_mv = viol.max()
        if best_mv <= 0.5 * zeta:
            break
        w *= 1 + 4 * viol / zeta
    return best_mv <= zeta, best_mv


def decode(traces, n, p, zeta=0.02):
    t = rank_means(traces, n)
    u = []
    for d in range(n):
        cands = []
        for bit in (0, 1):
            uu = u + [bit]
            A, a0 = coeff_rows(n, p, d + 1, uu)
            ok, mv = feasible(A, a0, t, zeta)
            cands.append((ok, mv))
        if cands[0][0] and not cands[1][0]:
            u.append(0)
        elif cands[1][0] and not cands[0][0]:
            u.append(1)
        else:
            u.append(0)
    return "".join(str(b) for b in u)
