// tr122/decoder.hpp — greedy prefix decoder (decoding.tex §2.2).
// Start with empty prefix; at each step test feasibility of both one-bit
// extensions; if exactly one accepts append it, otherwise append 0 (paper rule).
#pragma once
#include <cmath>
#include <string>
#include <vector>

#include "moment_d1.hpp"
#include "moment_generic.hpp"
#include "rank_stats.hpp"

namespace tr122 {

struct DecodeOptions {
    int D = 1;
    double zeta = 0.1;
    double psd_tol = 1e-9;
    int pocs_rounds = 60;
    int kaczmarz_sweeps = 3;
    int agmon_iters = 4000;
    std::string method = "pocs";
    double prune_eps = 1e-10;  // prune channel coefficients below this
};

struct BitDebug {
    int d = 0;         // position being decided (0-indexed)
    int chosen = 0;
    bool feas0 = false, feas1 = false;
    double viol0 = 0, viol1 = 0, eig0 = 0, eig1 = 0;
    int rounds0 = 0, rounds1 = 0;
};

struct DecodeResult {
    std::string out;
    std::vector<BitDebug> bits;
};

inline DecodeResult decode(const std::vector<std::string>& traces, int n, double p,
                           const DecodeOptions& opt) {
    double q = 1.0 - p;
    DecodeResult result;
    std::vector<int> rank_set_sizes_note;
    // 1. rank sets + empirical means + coefficient lists (all independent of prefix)
    std::vector<Mask> rsets;
    std::vector<double> tmeans;
    if (opt.D == 1 && n > 64) {
        tmeans.assign(n, 0.0);
        for (const auto& tr : traces)
            for (size_t i = 0; i < tr.size() && i < (size_t)n; ++i)
                tmeans[i] += tr[i] - '0';
        double TT = (double)std::max<size_t>(traces.size(), 1);
        for (auto& v : tmeans) v /= TT;
    } else {
        rsets = enumerate_rank_sets(n, opt.D);
        tmeans = estimate_rank_means(traces, rsets);
    }
    std::vector<CoeffList> cls(rsets.size());
    for (size_t s = 0; s < rsets.size(); ++s)
        cls[s] = channel_coefficients(mask_to_ranks(rsets[s]), n, p, q, opt.prune_eps);

    FeasOptions fopt;
    fopt.method = opt.method;
    fopt.zeta = opt.zeta;
    fopt.psd_tol = opt.psd_tol;
    fopt.pocs_rounds = opt.pocs_rounds;
    fopt.kaczmarz_sweeps = opt.kaczmarz_sweeps;
    fopt.agmon_iters = opt.agmon_iters;

    std::vector<int> u;  // prefix bits
    u.reserve(n);
    for (int dpos = 0; dpos < n; ++dpos) {
        BitDebug dbg; dbg.d = dpos;
        // Precompute per-leaf moment systems for both candidate prefixes.
        // Constraints depend on prefix length d+1 (positions 0..d fixed).
        for (int bit = 0; bit <= 1; ++bit) {
            std::vector<int> uu = u;
            uu.push_back(bit);
            int dd = dpos + 1;
            FeasResult fr;
            if (opt.D == 1 && n > 64) {
                D1Feas sys(n, dd);
                sys.build_free(n, p, q, tmeans, uu);
                fr = sys.solve(fopt);
            } else {
                GenericFeas sys(n, dd, opt.D);
                sys.build(cls, tmeans, uu);
                fr = sys.solve(fopt);
            }
            if (bit == 0) {
                dbg.feas0 = fr.accept; dbg.viol0 = fr.max_violation;
                dbg.eig0 = fr.min_eig; dbg.rounds0 = fr.rounds;
            } else {
                dbg.feas1 = fr.accept; dbg.viol1 = fr.max_violation;
                dbg.eig1 = fr.min_eig; dbg.rounds1 = fr.rounds;
            }
        }
        // paper rule: if precisely one accepts append it, else append 0
        if (dbg.feas0 && !dbg.feas1) dbg.chosen = 0;
        else if (dbg.feas1 && !dbg.feas0) dbg.chosen = 1;
        else dbg.chosen = 0;
        u.push_back(dbg.chosen);
        result.bits.push_back(dbg);
    }
    std::string s;
    for (int b : u) s.push_back(char('0' + b));
    result.out = s;
    return result;
}

}  // namespace tr122
