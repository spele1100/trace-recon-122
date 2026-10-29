// tr122/moment_d1.hpp — D=1 moment system, specialized for large n (up to 512).
// Rank observables: single ranks only. Moment coords: m1[i], m2[i][j].
// Moment matrix (size r+1): [[1, m1^T],[m1, M2]] with M2[i][i]=1, M2[i][j]=m2[i][j].
// Same POCS/Agmon feasibility as the generic path.
#pragma once
#include <Eigen/Dense>
#include <algorithm>
#include <cmath>
#include <random>
#include <vector>

#include "feas_solver.hpp"

#include "feas_types.hpp"
#include "rank_stats.hpp"

namespace tr122 {

class D1Feas {
public:
    int n, d, r;
    std::vector<double> m1, m2;  // m2 flattened lower-triangular: idx(i,j), i<j
    // constraints per rank l (1..n): const part + linear form over m1
    std::vector<double> a0_, b_;
    std::vector<std::vector<std::pair<int, double>>> A_;  // only m1 coords

    explicit D1Feas(int n_, int d_) : n(n_), d(d_), r(n_ - d_) {
        m1.assign(r, 0.0);
        m2.assign((size_t)r * (r - 1) / 2, 0.0);
    }
    int idx2(int i, int j) const {  // i<j
        if (i > j) std::swap(i, j);
        return i * r - (i * (i + 1)) / 2 + (j - i - 1);
    }

    // Mask-free build for large n: c_{l,i} = C(i,l-1) p^l q^{i+1-l} (0-indexed i).
    void build_free(int n_, double p, double q, const std::vector<double>& tmeans,
                    const std::vector<int>& u) {
        n = n_; r = n - d;
        m1.assign(r, 0.0); m2.assign((size_t)r * (r - 1) / 2, 0.0);
        A_.assign(n_, {}); a0_.assign(n_, 0.0); b_ = tmeans;
        std::vector<double> powp(n_ + 1, 1.0), powq(n_ + 1, 1.0);
        for (int i = 1; i <= n_; ++i) { powp[i] = powp[i - 1] * p; powq[i] = powq[i - 1] * q; }
        for (int l = 1; l <= n_; ++l) {
            for (int i = l - 1; i < n_; ++i) {
                double c = binom(i, l - 1) * powp[l] * powq[i + 1 - l];
                if (c < 1e-12) { if (i > 2 * l) break; continue; }
                if (i < d) {
                    if (u[i] == 1) a0_[l - 1] += c;
                } else {
                    a0_[l - 1] += 0.5 * c;
                    A_[l - 1].emplace_back(i - d, 0.5 * c);
                }
            }
        }
    }
    // cls: coefficient lists for single-rank sets (k=1); u: prefix bits
    void build(const std::vector<CoeffList>& cls, const std::vector<double>& tmeans,
               const std::vector<int>& u) {
        A_.assign(cls.size(), {});
        a0_.assign(cls.size(), 0.0);
        b_ = tmeans;
        for (size_t s = 0; s < cls.size(); ++s) {
            for (size_t t = 0; t < cls[s].tuples.size(); ++t) {
                int i = __builtin_ctzll(cls[s].tuples[t]);  // 0-indexed position
                double c = cls[s].coef[t];
                if (i < d) {
                    if (u[i] == 1) a0_[s] += c;
                } else {
                    a0_[s] += 0.5 * c;
                    A_[s].emplace_back(i - d, 0.5 * c);
                }
            }
        }
    }

    Eigen::MatrixXd moment_matrix() const {
        int G = r + 1;
        Eigen::MatrixXd M = Eigen::MatrixXd::Identity(G, G);
        for (int i = 0; i < r; ++i) M(0, 1 + i) = M(1 + i, 0) = m1[i];
        for (int i = 0; i < r; ++i)
            for (int j = i + 1; j < r; ++j) {
                M(1 + i, 1 + j) = M(1 + j, 1 + i) = m2[idx2(i, j)];
            }
        return M;
    }

    std::pair<double, double> check() const {
        double mv = 0.0;
        for (size_t s = 0; s < A_.size(); ++s) {
            double v = a0_[s];
            for (auto& p : A_[s]) v += p.second * m1[p.first];
            mv = std::max(mv, std::fabs(v - b_[s]));
        }
        Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es(moment_matrix());
        return {mv, es.eigenvalues()[0]};
    }

    void kaczmarz_slab(double zi) {
        for (size_t s = 0; s < A_.size(); ++s) {
            double v = a0_[s];
            for (auto& p : A_[s]) v += p.second * m1[p.first];
            double res = v - b_[s];
            if (std::fabs(res) <= zi) continue;
            double nrm2 = 0.0;
            for (auto& p : A_[s]) nrm2 += p.second * p.second;
            if (nrm2 < 1e-30) continue;
            double over = std::fabs(res) - zi;
            double step = over / nrm2 * (res > 0 ? 1.0 : -1.0);
            for (auto& p : A_[s]) m1[p.first] -= step * p.second;
        }
    }
    void clip_box() {
        for (auto& v : m1) v = std::min(1.0, std::max(-1.0, v));
        for (auto& v : m2) v = std::min(1.0, std::max(-1.0, v));
    }
    // D=1: rank-one completion makes PSD automatic (see GenericFeas note).
    void psd_project() {
        for (int i = 0; i < r; ++i)
            for (int j = i + 1; j < r; ++j) m2[idx2(i, j)] = m1[i] * m1[j];
        clip_box();
    }

    // Greedy sign flips on sum-of-squares residuals (constraints are linear
    // in m1 at D=1); mirrors GenericFeas::search_y for the linear case.
    double obj_y(const std::vector<int>& y) const {
        double o = 0.0;
        for (size_t s = 0; s < A_.size(); ++s) {
            double v = a0_[s];
            for (auto& p : A_[s]) v += p.second * y[p.first];
            double r = v - b_[s];
            o += r * r;
        }
        return o;
    }
    void search_y(std::vector<int>& y, std::mt19937_64& rng) {
        if (r == 0) return;
        std::vector<int> best_y = y;
        double best = obj_y(y);
        for (int rs = 0; rs < 4; ++rs) {
            std::vector<int> yc = (rs == 0) ? y : std::vector<int>(r);
            if (rs > 0) for (int i = 0; i < r; ++i) yc[i] = (rng() & 1) ? 1 : -1;
            std::vector<double> resid(A_.size());
            double cur = 0.0;
            for (size_t s = 0; s < A_.size(); ++s) {
                double v = a0_[s];
                for (auto& p : A_[s]) v += p.second * yc[p.first];
                resid[s] = v - b_[s];
                cur += resid[s] * resid[s];
            }
            for (int sweep = 0; sweep < 15; ++sweep) {
                bool improved = false;
                for (int k = 0; k < r; ++k) {
                    double delta = 0.0;
                    for (size_t s = 0; s < A_.size(); ++s) {
                        double ak = 0.0;
                        for (auto& p : A_[s])
                            if (p.first == k) ak = p.second;
                        if (ak == 0.0) continue;
                        double rn = resid[s] - 2.0 * ak * yc[k];
                        delta += rn * rn - resid[s] * resid[s];
                    }
                    if (delta < -1e-12) {
                        for (size_t s = 0; s < A_.size(); ++s) {
                            double ak = 0.0;
                            for (auto& p : A_[s])
                                if (p.first == k) ak = p.second;
                            if (ak == 0.0) continue;
                            resid[s] -= 2.0 * ak * yc[k];
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
    FeasResult solve(const FeasOptions& opt) {
        FeasResult res;
        maxviolation_solve(A_, a0_, b_, m1, opt.zeta);
        // NOTE: at D=1 the fractional m1 in [-1,1] is already a valid moment
        // functional (rank-one completion M = [1;m1][1;m1]^T + diag(1-m1_i^2)
        // >= 0), so no sign rounding is needed; accept iff viol <= zeta.
        psd_project();  // rank-one completion: PSD automatic (see note)
        auto [mv, le] = check();
        res.max_violation = mv; res.min_eig = le;
        res.accept = (mv <= opt.zeta && le >= -opt.psd_tol);
        res.rounds = 1;
        return res;
    }
};

}  // namespace tr122
