// tr122/moment_generic.hpp — generic moment system + feasibility (n <= 64, any D).
//
// Paper mapping:
//  - moment coords m_J for nonempty J ⊆ U, |J| <= 2D  (setup.tex eq. moment-matrix)
//  - moment matrix M(m)[I][J] = m_{I^J}  (rows I,J with |I|,|J| <= D)
//  - constraints |E_m[T_I] - t̂_I| <= ζ   (decoding.tex eq. prefix-feasibility)
//  - feasibility search: paper uses an Agmon/Motzkin–Schoenberg relaxation
//    (decoding.tex Lemma rational-feasibility) with exact rational arithmetic.
//    SUBSTITUTION: we implement (a) "agmon" mode: the same iteration in floating
//    point with an iteration cap, and (b) "pocs" mode: alternating projections
//    (Kaczmarz slab projections + PSD-cone projection via eigendecomposition),
//    which empirically finds the slack point fast. Both accept only points that
//    satisfy the exact constraints within tolerance. See SPEC.md.
#pragma once
#include <Eigen/Dense>
#include <algorithm>
#include <numeric>
#include <cmath>
#include <cstdint>
#include <vector>

#include "feas_solver.hpp"
#include "feas_types.hpp"
#include "rank_stats.hpp"

namespace tr122 {


// Coordinate list: masks over U with popcount in [1, min(K, r)] sorted.
inline std::vector<Mask> enumerate_coord_masks(int r, int K) {
    std::vector<Mask> out;
    for (int k = 1; k <= std::min(K, r); ++k) {
        std::vector<int> cur(k);
        for (int i = 0; i < k; ++i) cur[i] = i;
        while (true) {
            Mask m = 0;
            for (int i : cur) m |= (Mask(1) << i);
            out.push_back(m);
            int i = k - 1;
            while (i >= 0 && cur[i] == r - k + i) --i;
            if (i < 0) break;
            ++cur[i];
            for (int j = i + 1; j < k; ++j) cur[j] = cur[j - 1] + 1;
        }
    }
    std::sort(out.begin(), out.end());
    return out;
}

class GenericFeas {
public:
    int n, d, r, D;  // string length, prefix length (positions 0..d-1 fixed), r = n-d
    std::vector<Mask> rows_;    // |I| <= D masks over U (local), sorted; row 0 is empty
    std::vector<Mask> coords_;  // 1 <= |J| <= 2D masks over U (local), sorted
    std::vector<int> row_index_;  // maps row mask -> index via lookup helper
    // constraints: for each rank set, sparse affine form a0 + sum a_c * m_c
    std::vector<double> a0_, b_;
    std::vector<std::vector<std::pair<int, double>>> A_;
    std::vector<double> m_;      // current moment vector (parallel to coords_)
    std::vector<int> row_of_mask_storage_;

    GenericFeas(int n_, int d_, int D_) : n(n_), d(d_), r(n_ - d_), D(D_) {
        rows_ = enumerate_coord_masks(r, D);
        // rows_ from enumerate_coord_masks excludes empty mask; prepend it
        rows_.insert(rows_.begin(), 0);
        coords_ = enumerate_coord_masks(r, 2 * D);
        m_.assign(coords_.size(), 0.0);
    }

    int coord_index(Mask s) const {
        auto it = std::lower_bound(coords_.begin(), coords_.end(), s);
        return (int)(it - coords_.begin());  // caller must ensure existence
    }
    int row_index(Mask s) const {
        auto it = std::lower_bound(rows_.begin(), rows_.end(), s);
        return (int)(it - rows_.begin());
    }

    double getm(Mask s) const {  // s nonempty local mask (must exist if |s|<=2D)
        auto it = std::lower_bound(coords_.begin(), coords_.end(), s);
        if (it == coords_.end() || *it != s) return 0.0;
        return m_[(int)(it - coords_.begin())];
    }
    void setm(Mask s, double v) {
        auto it = std::lower_bound(coords_.begin(), coords_.end(), s);
        if (it != coords_.end() && *it == s) m_[(int)(it - coords_.begin())] = v;
    }

    // Build constraints from precomputed coefficient lists + empirical means.
    // u: prefix bits (length d). bits: 0/1
    void build(const std::vector<CoeffList>& cls, const std::vector<double>& tmeans,
               const std::vector<int>& u) {
        A_.clear(); a0_.clear(); b_.clear();
        A_.resize(cls.size()); a0_.resize(cls.size(), 0.0); b_ = tmeans;
        // accumulate into map-like buffer
        std::vector<std::pair<int, double>> buf;
        for (size_t s = 0; s < cls.size(); ++s) {
            buf.clear();
            const CoeffList& cl = cls[s];
            for (size_t t = 0; t < cl.tuples.size(); ++t) {
                Mask A = cl.tuples[t];
                Mask fixed = A & ((d >= 64) ? ~Mask(0) : ((Mask(1) << d) - 1));
                Mask unfixed = A >> d;  // local mask over U (positions d..n-1 -> 0..)
                // fixed bits must all be 1, else monomial dies
                bool alive = true;
                for (int i = 0; i < d; ++i)
                    if ((fixed >> i) & 1) {
                        if (u[i] == 0) { alive = false; break; }
                    }
                if (!alive) continue;
                double w = cl.coef[t] * std::pow(0.5, (double)__builtin_popcountll(unfixed));
                // enumerate submasks of unfixed (including empty -> a0)
                Mask sub = unfixed;
                while (true) {
                    if (sub == 0) {
                        a0_[s] += w;
                    } else {
                        buf.emplace_back(coord_index(sub), w);
                    }
                    if (sub == 0) break;
                    sub = (sub - 1) & unfixed;
                }
            }
            // merge duplicates
            std::sort(buf.begin(), buf.end(),
                      [](auto& a, auto& b) { return a.first < b.first; });
            for (auto& p : buf) {
                if (!A_[s].empty() && A_[s].back().first == p.first)
                    A_[s].back().second += p.second;
                else
                    A_[s].push_back(p);
            }
            // drop exact zeros
            A_[s].erase(std::remove_if(A_[s].begin(), A_[s].end(),
                                       [](auto& p) { return p.second == 0.0; }),
                        A_[s].end());
        }
    }

    Eigen::MatrixXd moment_matrix() const {
        int G = (int)rows_.size();
        Eigen::MatrixXd M(G, G);
        for (int i = 0; i < G; ++i)
            for (int j = i; j < G; ++j) {
                Mask s = rows_[i] ^ rows_[j];
                double v = (s == 0) ? 1.0 : getm(s);
                M(i, j) = v;
                M(j, i) = v;
            }
        return M;
    }

    std::pair<double, double> check() const {  // returns (max_violation, min_eig)
        double mv = 0.0;
        for (size_t s = 0; s < A_.size(); ++s) {
            double v = a0_[s];
            for (auto& p : A_[s]) v += p.second * m_[p.first];
            mv = std::max(mv, std::fabs(v - b_[s]));
        }
        Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es(moment_matrix());
        double le = es.eigenvalues().size() ? es.eigenvalues()[0] : 1.0;
        return {mv, le};
    }

    void kaczmarz_slab(double zeta_inner, std::mt19937_64& rng) {
        {
            std::vector<int> order(A_.size());
            std::iota(order.begin(), order.end(), 0);
            std::shuffle(order.begin(), order.end(), rng);
            for (int s : order) {
                double v = a0_[s];
                for (auto& p : A_[s]) v += p.second * m_[p.first];
                double res = v - b_[s];
                if (std::fabs(res) <= zeta_inner) continue;
                double nrm2 = 0.0;
                for (auto& p : A_[s]) nrm2 += p.second * p.second;
                if (nrm2 < 1e-30) continue;  // constant constraint; nothing to project
                double over = std::fabs(res) - zeta_inner;
                double step = over / nrm2 * (res > 0 ? 1.0 : -1.0);
                for (auto& p : A_[s]) m_[p.first] -= step * p.second;
            }
        }
    }

    void clip_box() {
        for (auto& v : m_) v = std::min(1.0, std::max(-1.0, v));
    }

    // Project onto the structured PSD cone {M(m) >= 0, diag = 1}.
    // D == 1: rank-one completion m2_ij = m1_i m1_j makes
    //   M = [1;m1][1;m1]^T + diag(0, 1 - m1_i^2) >= 0 for ANY boxed m1,
    // so no eigendecomposition is needed (PSD is free at D=1).
    // D >= 2: Higham-style alternating projection (eigenvalue clamp + reset
    // unit diagonal), then a final shrink if a small negative eigenvalue
    // remains.
    void psd_project() {
        if (D == 1) {
            for (Mask s : coords_) {
                if (__builtin_popcountll(s) == 2) {
                    int i = __builtin_ctzll(s);
                    int j = 63 - __builtin_clzll(s);
                    setm(s, getm(Mask(1) << i) * getm(Mask(1) << j));
                }
            }
            clip_box();
            return;
        }
        Eigen::MatrixXd M = moment_matrix();
        for (int hi = 0; hi < 25; ++hi) {
            Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es(M);
            double lmin = es.eigenvalues()[0];
            if (lmin >= -1e-13) break;
            Eigen::VectorXd ev = es.eigenvalues();
            for (int i = 0; i < ev.size(); ++i) ev[i] = std::max(ev[i], 0.0);
            M = es.eigenvectors() * ev.asDiagonal() * es.eigenvectors().transpose();
            M.diagonal().setOnes();  // m_{I^I} = 1 structurally
        }
        int G = (int)rows_.size();
        for (Mask s : coords_) {
            double v;
            int k = __builtin_popcountll(s);
            if (k <= D) {
                v = M(0, row_index(s));
            } else {
                Mask I = 0;
                Mask tmp = s;
                for (int c = 0; c < D && tmp; ++c) {
                    Mask low = tmp & (~tmp + 1);
                    I |= low;
                    tmp ^= low;
                }
                Mask J = s & ~I;
                v = M(row_index(I), row_index(J));
            }
            setm(s, v);
        }
        clip_box();
        ensure_psd_shrink();
    }

    void ensure_psd_shrink() {
        Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es(moment_matrix());
        double l = es.eigenvalues()[0];
        if (l < 0.0) {
            double rho = 1.0 / (1.0 - l);
            for (auto& v : m_) v *= rho;
        }
    }


    // Unified feasibility:
    //  Stage 1: FISTA (weighted-LS) on the affine constraints.
    //  Stage 2a (D == 1): fractional m1 is PSD-completable (rank-one +
    //    nonneg diagonal), so accept iff max violation <= zeta.
    //  Stage 2b (D >= 2): PSD must hold for coords up to 2D. We search
    //    within the paper's slack family m_S = rho * prod_{i in S} y_i
    //    (y in {+-1}^r, rho = 1 - zeta/4), whose moment matrix is
    //    M = rho*z*z^T + (1-rho)*I >= (1-rho)I: PSD with explicit slack,
    //    exactly the mixture used in decoding.tex Lemma prefix-slack.
    //    y is initialized from the FISTA solution and refined by greedy
    //    sign flips. SUBSTITUTION of the general SDP feasibility search;
    //    both soundness (accepted points are valid moment functionals)
    //    and existence (the slack point lies in this family) hold.
    double eval_viol_y(const std::vector<int>& y, double rho) const {
        double mv = 0.0;
        for (size_t s = 0; s < A_.size(); ++s) {
            double v = a0_[s];
            for (auto& p : A_[s]) {
                Mask sub = coords_[p.first];
                double mon = 1.0;
                Mask t = sub;
                while (t) {
                    int i = __builtin_ctzll(t);
                    if (y[i] < 0) mon = -mon;
                    t &= t - 1;
                }
                v += p.second * rho * mon;
            }
            mv = std::max(mv, std::fabs(v - b_[s]));
        }
        return mv;
    }
    void set_m_y(const std::vector<int>& y, double rho) {
        for (size_t c = 0; c < coords_.size(); ++c) {
            double mon = 1.0;
            Mask t = coords_[c];
            while (t) {
                int i = __builtin_ctzll(t);
                if (y[i] < 0) mon = -mon;
                t &= t - 1;
            }
            m_[c] = rho * mon;
        }
    }
    // residual of constraint s at sign vector y with mixing rho
    double res_s(size_t s, const std::vector<int>& y, double rho) const {
        double v = a0_[s];
        for (auto& p : A_[s]) {
            Mask t = coords_[p.first];
            double mon = 1.0;
            while (t) {
                int i = __builtin_ctzll(t);
                if (y[i] < 0) mon = -mon;
                t &= t - 1;
            }
            v += p.second * rho * mon;
        }
        return v - b_[s];
    }
    // c_{s,k} = sum over coords S containing k of w_S * prod_{S\{k}} y
    double c_sk(size_t s, int k, const std::vector<int>& y) const {
        double c = 0.0;
        for (auto& p : A_[s]) {
            Mask t = coords_[p.first];
            if (!((t >> k) & 1)) continue;
            double mon = 1.0;
            Mask u = t & ~(Mask(1) << k);
            while (u) {
                int i = __builtin_ctzll(u);
                if (y[i] < 0) mon = -mon;
                u &= u - 1;
            }
            c += p.second * mon;
        }
        return c;
    }
    // delta of flipping bit k: res_s -= 2*rho*c_sk; returns affected (s, delta) and d(sum sq)
    // Exact search over y in {+-1}^r via Gray code when r <= 18; otherwise
    // annealing + single/pair flips from several restarts.
    double obj_y(const std::vector<int>& y, double rho) const {
        double o = 0.0;
        for (size_t s = 0; s < A_.size(); ++s) {
            double v = res_s(s, y, rho);
            o += v * v;
        }
        return o;
    }
    void linear_search_y(std::vector<int>& y, double rho, std::mt19937_64& rng) {
        std::vector<int> best_y = y;
        double best = 1e18;
        // column index: coord -> constraints containing it (singleton coords only)
        std::vector<std::vector<std::pair<size_t, int>>> col(r);
        for (size_t s = 0; s < A_.size(); ++s)
            for (auto& p : A_[s]) {
                Mask cm = coords_[p.first];
                if (__builtin_popcountll(cm) == 1)
                    col[__builtin_ctzll(cm)].emplace_back(s, p.first);
            }
        for (int rs = 0; rs < 4; ++rs) {
            std::vector<int> yc = (rs == 0) ? y : std::vector<int>(r);
            if (rs > 0) for (int i = 0; i < r; ++i) yc[i] = (rng() & 1) ? 1 : -1;
            std::vector<double> resid(A_.size());
            double cur = 0.0;
            for (size_t s = 0; s < A_.size(); ++s) {
                resid[s] = res_s(s, yc, rho);
                cur += resid[s] * resid[s];
            }
            for (int sweep = 0; sweep < 12; ++sweep) {
                bool improved = false;
                for (int k = 0; k < r; ++k) {
                    if (col[k].empty()) continue;
                    double delta = 0.0;
                    for (auto& pr : col[k]) {
                        double w = A_[pr.first][0].second;  // unused
                        (void)w;
                        auto& pair = pr;
                        // coefficient of y_k in constraint s: coord value
                        // find the entry
                        double ak = 0.0;
                        for (auto& pp : A_[pair.first])
                            if (pp.first == pair.second) ak = pp.second;
                        double rn = resid[pair.first] - 2.0 * rho * ak * yc[k];
                        delta += rn * rn - resid[pair.first] * resid[pair.first];
                    }
                    if (delta < -1e-12) {
                        for (auto& pr : col[k]) {
                            double ak = 0.0;
                            for (auto& pp : A_[pr.first])
                                if (pp.first == pr.second) ak = pp.second;
                            resid[pr.first] -= 2.0 * rho * ak * yc[k];
                        }
                        yc[k] = -yc[k];
                        cur += delta;
                        improved = true;
                    }
                }
                if (!improved) break;
            }
            if (cur < best) { best = cur; best_y = yc; }
        }
        y = best_y;
    }
    void search_y(std::vector<int>& y, double rho, std::mt19937_64& rng) {
        if (r == 0) return;
        if (r <= 18) {  // exhaustive Gray-code search (exact)
            std::vector<int> yb(r, 1);
            double best = obj_y(yb, rho);
            std::vector<int> best_y = yb;
            std::vector<double> resid(A_.size());
            for (size_t s = 0; s < A_.size(); ++s) resid[s] = res_s(s, yb, rho);
            for (uint32_t g = 1; g < (1u << r); ++g) {
                int k = __builtin_ctzll(g & (~g + 1));  // flipped bit in Gray sequence
                // correct bit index: Gray code flips bit ctz(g)
                k = __builtin_ctz(g);
                double delta = 0.0;
                int yk = yb[k];
                for (size_t s = 0; s < A_.size(); ++s) {
                    double ck = c_sk(s, k, yb) * yk;
                    if (ck == 0.0) continue;
                    double rn = resid[s] - 2.0 * rho * ck;
                    delta += rn * rn - resid[s] * resid[s];
                    resid[s] = rn;
                }
                yb[k] = -yb[k];
                (void)delta;
                double o = 0.0;
                for (double v : resid) o += v * v;
                if (o < best) { best = o; best_y = yb; }
            }
            y = best_y;
            return;
        }
        // annealing + polish, multi-restart
        std::vector<int> best_y = y;
        double best = obj_y(y, rho);
        for (int rs = 0; rs < 4; ++rs) {
            std::vector<int> yc = (rs == 0) ? y : std::vector<int>(r);
            if (rs > 0) for (int i = 0; i < r; ++i) yc[i] = (rng() & 1) ? 1 : -1;
            double cur = obj_y(yc, rho);
            double T = 0.02;
            for (int it = 0; it < 60 * r; ++it) {
                int k = rng() % r;
                std::vector<int> yn = yc; yn[k] = -yn[k];
                double nv = obj_y(yn, rho);  // O(nnz): acceptable for moderate n
                if (nv <= cur || std::uniform_real_distribution<double>(0, 1)(rng) <
                                     std::exp(-(nv - cur) / (T * cur + 1e-9))) {
                    yc = yn; cur = nv;
                }
                T *= 0.9995;
            }
            // pair-flip polish
            for (int sweep = 0; sweep < 8; ++sweep) {
                bool improved = false;
                for (int a = 0; a < r; ++a) {
                    double bestd = 0; int bestk = -1;
                    std::vector<int> yn = yc;
                    yn[a] = -yn[a];
                    double base = obj_y(yn, rho);
                    if (base < cur) { cur = base; yc = yn; improved = true; continue; }
                    for (int b = 0; b < r && b != a; ++b) {
                        std::vector<int> y2 = yn; y2[b] = -y2[b];
                        double v = obj_y(y2, rho);
                        if (v < cur + bestd) { bestd = v - cur; bestk = b; }
                    }
                    if (bestk >= 0) {
                        yc[a] = -yc[a]; yc[bestk] = -yc[bestk];
                        cur += bestd; improved = true;
                    }
                }
                if (!improved) break;
            }
            if (cur < best) { best = cur; best_y = yc; }
        }
        y = best_y;
    }
    FeasResult solve(const FeasOptions& opt) {
        FeasResult res;
        if (opt.method == "agmon") return solve_agmon(opt);
        maxviolation_solve(A_, a0_, b_, m_, opt.zeta);
        if (r > 0) {
            double rho = (D == 1) ? 1.0 : 1.0 - opt.zeta / 4.0;
            std::vector<int> y(r, 1);
            for (Mask s : coords_)
                if (__builtin_popcountll(s) == 1) {
                    int i = __builtin_ctzll(s);
                    y[i] = (getm(s) >= 0) ? 1 : -1;
                }
            std::mt19937_64 rng(0xC0FFEE);
            search_y(y, rho, rng);
            set_m_y(y, rho);
        }
        auto [mv, le] = check();
        res.max_violation = mv; res.min_eig = le;
        res.accept = (mv <= opt.zeta && le >= -opt.psd_tol);
        res.rounds = 1;
        return res;
    }

    // Paper-faithful Agmon relaxation (floating point, capped).
    FeasResult solve_agmon(const FeasOptions& opt) {
        FeasResult res;
        int G = (int)rows_.size();
        double lam = opt.agmon_lambda_scale * opt.zeta / (16.0 * std::max(G, 1));
        for (int it = 0; it < opt.agmon_iters; ++it) {
            // violated affine constraint?
            int vs = -1; double vres = 0;
            for (size_t s = 0; s < A_.size(); ++s) {
                double v = a0_[s];
                for (auto& p : A_[s]) v += p.second * m_[p.first];
                if (std::fabs(v - b_[s]) > opt.zeta &&
                    std::fabs(v - b_[s]) > std::fabs(vres)) {
                    vs = (int)s; vres = v - b_[s];
                }
            }
            if (vs >= 0) {
                for (auto& p : A_[vs]) m_[p.first] -= lam * p.second * (vres > 0 ? 1 : -1);
                clip_box();
                continue;
            }
            Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es(moment_matrix());
            double le = es.eigenvalues()[0];
            if (le < -opt.psd_tol) {
                Eigen::VectorXd v = es.eigenvectors().col(0);
                // a = h / (G * v^T v), h_S = sum_{I^J=S} v_I v_J
                std::vector<double> a(m_.size(), 0.0);
                for (int i = 0; i < G; ++i)
                    for (int j = 0; j < G; ++j) {
                        Mask s = rows_[i] ^ rows_[j];
                        if (s == 0 || i > j) continue;
                        double c = (i == j) ? v[i] * v[i] : 2.0 * v[i] * v[j];
                        auto it = std::lower_bound(coords_.begin(), coords_.end(), s);
                        if (it != coords_.end() && *it == s) a[(int)(it - coords_.begin())] += c;
                    }
                double vv = v.squaredNorm();
                for (size_t c = 0; c < m_.size(); ++c) m_[c] += lam * a[c] / (G * vv);
                clip_box();
                continue;
            }
            auto [mv, le2] = check();
            res.accept = true; res.max_violation = mv; res.min_eig = le2; res.rounds = it;
            return res;
        }
        auto [mv, le] = check();
        res.max_violation = mv; res.min_eig = le;
        res.accept = (mv <= opt.zeta && le >= -opt.psd_tol);
        return res;
    }
};

}  // namespace tr122
