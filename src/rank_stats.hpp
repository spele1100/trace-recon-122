// tr122/rank_stats.hpp — rank observables and their channel coefficients.
//
// Paper (setup.tex §2.1, decoding.tex eq. explicit-channel-coefficient):
// For rank set I = {l1<...<lk}, the trace observable is the product of trace
// bits at ranks l1..lk (0 if the trace is shorter than lk). Its channel
// expectation T_I(X) = sum_{i1<...<ik} c_{I;i} X_i1...X_ik with
//   c = C(i1-1, l1-1) * prod_j C(i_{j+1}-i_j-1, l_{j+1}-l_j-1) * p^{lk} q^{ik-lk}
#pragma once
#include <cmath>
#include <cstdint>
#include <map>
#include <vector>

namespace tr122 {

using Mask = uint64_t;  // bitset over [n], requires n <= 64 for the generic path

inline double binom(int n, int k) {
    if (k < 0 || k > n) return 0.0;
    if (k > n - k) k = n - k;
    double r = 1.0;
    for (int i = 0; i < k; ++i) r = r * (n - i) / (i + 1);
    return r;
}

// A rank set is a sorted vector of ranks (1-indexed). We identify rank sets
// with bitmasks over ranks [n] as well; both helpers provided.
struct RankSet {
    std::vector<int> ranks;  // sorted, 1-indexed
};

// Enumerate all rank sets of size 1..D over [n]. Returns masks.
inline std::vector<Mask> enumerate_rank_sets(int n, int D) {
    std::vector<Mask> out;
    for (int k = 1; k <= D; ++k) {
        // iterate masks with popcount k
        Mask m = 0;
        std::vector<int> idx(k);
        for (int i = 0; i < k; ++i) idx[i] = i;  // 0-based positions
        auto emit = [&]() {
            Mask mm = 0;
            for (int i : idx) mm |= (Mask(1) << i);
            out.push_back(mm);
        };
        if (k > n) break;
        emit();
        while (true) {
            int i = k - 1;
            while (i >= 0 && idx[i] == n - k + i) --i;
            if (i < 0) break;
            ++idx[i];
            for (int j = i + 1; j < k; ++j) idx[j] = idx[j - 1] + 1;
            emit();
        }
    }
    return out;
}

inline std::vector<int> mask_to_ranks(Mask m) {
    std::vector<int> r;
    int i = 0;
    while (m) {
        if (m & 1) r.push_back(i + 1);
        m >>= 1;
        ++i;
    }
    return r;
}

inline Mask ranks_to_mask(const std::vector<int>& ranks) {
    Mask m = 0;
    for (int l : ranks) m |= (Mask(1) << (l - 1));
    return m;
}

// Empirical mean of the observable for every rank set (mask over ranks).
// traces: strings of '0'/'1'.
inline std::vector<double> estimate_rank_means(const std::vector<std::string>& traces,
                                               const std::vector<Mask>& rank_sets) {
    std::vector<double> sum(rank_sets.size(), 0.0);
    for (const auto& tr : traces) {
        int L = (int)tr.size();
        for (size_t s = 0; s < rank_sets.size(); ++s) {
            Mask m = rank_sets[s];
            int last = 63 - __builtin_clzll(m);  // highest rank-1 index
            if (last >= L) continue;             // observable = 0
            double v = 1.0;
            Mask mm = m;
            int i = 0;
            while (mm) {
                if (mm & 1) v *= (tr[i] - '0');
                mm >>= 1;
                ++i;
            }
            sum[s] += v;
        }
    }
    double T = (double)std::max<size_t>(traces.size(), 1);
    for (auto& v : sum) v /= T;
    return sum;
}

// Precomputed channel coefficients for one rank set:
// list of (input-position mask, coefficient), pruned to coeff >= eps.
struct CoeffList {
    std::vector<Mask> tuples;     // mask over input positions [n] (0-indexed bits)
    std::vector<double> coef;
};

inline CoeffList channel_coefficients(const std::vector<int>& ranks, int n, double p,
                                      double q, double eps = 1e-12,
                                      const double* powp = nullptr,
                                      const double* powq = nullptr) {
    static std::vector<double> pp, qq;
    if (!powp) {  // build power tables once per (p,q); cheap enough to rebuild
        pp.assign(n + 1, 1.0); qq.assign(n + 1, 1.0);
        for (int i = 1; i <= n; ++i) { pp[i] = pp[i - 1] * p; qq[i] = qq[i - 1] * q; }
        powp = pp.data(); powq = qq.data();
    }
    (void)0;
    CoeffList cl;
    int k = (int)ranks.size();
    // enumerate i1<...<ik over [n]
    std::vector<int> idx(k);
    for (int i = 0; i < k; ++i) idx[i] = ranks[i] - 1 >= i ? i : i;  // start 0..k-1
    // simple full enumeration
    std::vector<int> cur(k);
    for (int i = 0; i < k; ++i) cur[i] = i;
    auto coef_of = [&](const std::vector<int>& pos) {
        // pos 0-indexed input positions ascending
        double c = binom(pos[0], ranks[0] - 1);
        for (int j = 0; j + 1 < k; ++j)
            c *= binom(pos[j + 1] - pos[j] - 1, ranks[j + 1] - ranks[j] - 1);
        c *= powp[ranks[k - 1]] * powq[pos[k - 1] + 1 - ranks[k - 1]];
        return c;
    };
    if (k > n) return cl;
    while (true) {
        double c = coef_of(cur);
        if (c >= eps) {
            Mask m = 0;
            for (int i : cur) m |= (Mask(1) << i);
            cl.tuples.push_back(m);
            cl.coef.push_back(c);
        }
        // next k-combination
        int i = k - 1;
        while (i >= 0 && cur[i] == n - k + i) --i;
        if (i < 0) break;
        ++cur[i];
        for (int j = i + 1; j < k; ++j) cur[j] = cur[j - 1] + 1;
    }
    return cl;
}

}  // namespace tr122
