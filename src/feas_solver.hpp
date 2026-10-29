// feas_solver.hpp — FISTA (accelerated projected gradient) bounded least squares
// for the affine moment constraints  a0_s + A_s . m = b_s  over m in [-1,1]^l.
// Rows are internally normalized; Lipschitz constant estimated by power
// iteration on the Hessian. Used inside POCS; robust on the ill-conditioned
// (geometrically decaying) constraint system where Kaczmarz stalls.
#pragma once
#include <cmath>
#include <vector>

namespace tr122 {

inline double max_violation_of(const std::vector<std::vector<std::pair<int, double>>>& A,
                               const std::vector<double>& a0, const std::vector<double>& b,
                               const std::vector<double>& m) {
    double mv = 0.0;
    for (size_t s = 0; s < A.size(); ++s) {
        double r = a0[s];
        for (auto& p : A[s]) r += p.second * m[p.first];
        mv = std::max(mv, std::fabs(r - b[s]));
    }
    return mv;
}

// Minimize sum_s w_s (a0_s + A_s.m - b_s)^2 / 2, w_s = 1/||A_s||^2, box [-1,1].
inline void fista_solve(const std::vector<std::vector<std::pair<int, double>>>& A,
                        const std::vector<double>& a0, const std::vector<double>& b,
                        std::vector<double>& m, int iters = 500,
                        const std::vector<double>* wts = nullptr) {
    size_t l = m.size();
    size_t N = A.size();
    std::vector<double> nrm2(N, 0.0), colsq(l, 0.0);
    for (size_t s = 0; s < N; ++s)
        for (auto& p : A[s]) {
            nrm2[s] += p.second * p.second;
            double w = wts ? (*wts)[s] : 1.0;
            colsq[p.first] += w * p.second * p.second;  // diag of weighted Hessian
        }
    for (auto& v : colsq) v = std::sqrt(std::max(v, 1e-300));
    // u-space: m_i = u_i * s_i with s_i = 1/sqrt(colsq): unit-ish diagonal Hessian
    std::vector<double> sc(l);
    for (size_t i = 0; i < l; ++i) sc[i] = 1.0 / colsq[i];
    auto Hv = [&](const std::vector<double>& v) {
        std::vector<double> u(l, 0.0);
        for (size_t s = 0; s < N; ++s) {
            if (nrm2[s] < 1e-30) continue;
            double w = wts ? (*wts)[s] / nrm2[s] : 1.0 / nrm2[s];
            double av = 0.0;
            for (auto& p : A[s]) av += p.second * sc[p.first] * v[p.first];
            for (auto& p : A[s]) u[p.first] += sc[p.first] * p.second * av * w;
        }
        return u;
    };
    std::vector<double> v(l, 0.0);
    for (size_t i = 0; i < l; ++i) v[i] = std::sin(1.0 + i);
    double L = 1.0;
    for (int it = 0; it < 30; ++it) {
        auto u = Hv(v);
        double nv = 0.0;
        for (size_t i = 0; i < l; ++i) nv += u[i] * u[i];
        nv = std::sqrt(nv);
        if (nv < 1e-30) break;
        for (size_t i = 0; i < l; ++i) v[i] = u[i] / nv;
        L = nv;
    }
    L = std::max(L, 1e-12) * 1.05;
    double invL = 1.0 / L;
    auto grad = [&](const std::vector<double>& uu, std::vector<double>& g) {
        std::fill(g.begin(), g.end(), 0.0);
        for (size_t s = 0; s < N; ++s) {
            if (nrm2[s] < 1e-30) continue;
            double w = wts ? (*wts)[s] / nrm2[s] : 1.0 / nrm2[s];
            double r = a0[s];
            for (auto& p : A[s]) r += p.second * uu[p.first] * sc[p.first];
            double rc = (r - b[s]) * w;
            for (auto& p : A[s]) g[p.first] += rc * p.second * sc[p.first];
        }
    };
    auto proj = [&](std::vector<double>& uu) {
        for (size_t i = 0; i < l; ++i)
            uu[i] = std::min(sc[i], std::max(-sc[i], uu[i]));
    };
    std::vector<double> u(l), y(l), g(l), un(l);
    for (size_t i = 0; i < l; ++i) u[i] = std::min(sc[i], std::max(-sc[i], m[i] * colsq[i]));
    y = u;
    double tk = 1.0;
    for (int it = 0; it < iters; ++it) {
        grad(y, g);
        for (size_t i = 0; i < l; ++i) un[i] = y[i] - invL * g[i];
        proj(un);
        double tn = (1.0 + std::sqrt(1.0 + 4.0 * tk * tk)) / 2.0;
        for (size_t i = 0; i < l; ++i) y[i] = un[i] + ((tk - 1.0) / tn) * (un[i] - u[i]);
        proj(y);
        u = un;
        tk = tn;
    }
    for (size_t i = 0; i < l; ++i) m[i] = u[i] * sc[i];
}

}  // namespace tr122

namespace tr122 {
// Coordinate-descent (Gauss-Seidel) bounded least squares: exact 1-D
// minimization per coordinate. Very effective on the near-triangular
// moment-constraint system where gradient methods stall.
inline void cd_solve(const std::vector<std::vector<std::pair<int, double>>>& A,
                     const std::vector<double>& a0, const std::vector<double>& b,
                     std::vector<double>& m, int sweeps = 60,
                     const std::vector<double>* wts = nullptr) {
    size_t l = m.size();
    std::vector<double> nrm2(A.size(), 0.0);
    for (size_t s = 0; s < A.size(); ++s)
        for (auto& p : A[s]) nrm2[s] += p.second * p.second;
    std::vector<double> res(A.size());
    for (size_t s = 0; s < A.size(); ++s) {
        double v = a0[s];
        for (auto& p : A[s]) v += p.second * m[p.first];
        res[s] = v - b[s];
    }
    // reverse column index: coord -> constraints containing it
    std::vector<std::vector<std::pair<size_t, double>>> col(l);
    for (size_t s = 0; s < A.size(); ++s)
        for (auto& p : A[s]) col[p.first].emplace_back(s, p.second);
    for (int sw = 0; sw < sweeps; ++sw) {
        double tot = 0.0;
        for (size_t i = 0; i < l; ++i) {
            double num = 0.0, den = 0.0;
            for (auto& pr : col[i]) {
                double w = wts ? (*wts)[pr.first] : 1.0;
                num += w * pr.second * res[pr.first];
                den += w * pr.second * pr.second;
            }
            if (den < 1e-30) continue;
            double delta = -num / den;
            double nv = m[i] + delta;
            if (nv > 1.0) delta = 1.0 - m[i];
            if (nv < -1.0) delta = -1.0 - m[i];
            if (delta == 0.0) continue;
            m[i] += delta;
            for (auto& pr : col[i]) res[pr.first] += pr.second * delta;
        }
        for (double v : res) tot += v * v;
        if (tot < 1e-18) break;
    }
}

// Feasibility wrapper: weighted-LS FISTA with multiplicative reweighting
// toward minimizing the MAX absolute violation (Chebyshev-style).
inline double maxviolation_solve(
    const std::vector<std::vector<std::pair<int, double>>>& A,
    const std::vector<double>& a0, const std::vector<double>& b,
    std::vector<double>& m, double zeta, int outer = 12, int iters = 500) {
    double prev = 1e18;
    std::vector<double> w(A.size(), 1.0);
    double mv = 1e18;
    for (int o = 0; o < outer; ++o) {
        fista_solve(A, a0, b, m, iters, &w);
        cd_solve(A, a0, b, m, 40, &w);
        std::vector<double> r(A.size());
        double mo = 0.0;
        for (size_t s = 0; s < A.size(); ++s) {
            double v = a0[s];
            for (auto& p : A[s]) v += p.second * m[p.first];
            r[s] = std::fabs(v - b[s]);
            mo = std::max(mo, r[s]);
        }
        mv = mo;
        if (mo <= 0.5 * zeta) break;
        if (mo > 0.97 * prev) { if (o >= 2) break; prev = mo; continue; }
        prev = mo;
        bool changed = false;
        for (size_t s = 0; s < A.size(); ++s)
            if (r[s] > 0.25 * zeta) { w[s] *= 1.0 + 4.0 * r[s] / zeta; changed = true; }
        if (!changed) break;
    }
    return mv;
}
}  // namespace tr122
