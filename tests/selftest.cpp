// tests/selftest.cpp — C++ smoke tests (oracle + numerics).
#include <cassert>
#include <cmath>
#include <cstdio>
#include <random>
#include <string>
#include <vector>

#include "channel.hpp"
#include "decoder.hpp"
#include "rank_stats.hpp"

using namespace tr122;

static int failures = 0;
#define CHECK(cond, msg)                                     \
    do {                                                     \
        if (!(cond)) {                                       \
            printf("FAIL: %s (line %d)\n", msg, __LINE__);   \
            ++failures;                                      \
        }                                                    \
    } while (0)

// Exact T_I expectation by brute force over all retention subsets (tiny n).
static double exact_TI(const std::string& x, const std::vector<int>& I, double p) {
    int n = (int)x.size();
    double q = 1 - p, sum = 0.0;
    for (int mask = 0; mask < (1 << n); ++mask) {
        double pw = 1.0;
        std::string trace;
        for (int i = 0; i < n; ++i) {
            if (mask & (1 << i)) { pw *= p; trace += x[i]; }
            else pw *= q;
        }
        double v = 1.0;
        bool ok = true;
        for (int l : I) {
            if (l > (int)trace.size()) { ok = false; break; }
            v *= trace[l - 1] - '0';
        }
        if (ok) sum += pw * v;
    }
    return sum;
}

int main() {
    // 1. Channel coefficients match brute-force expectation of rank observables.
    {
        int n = 8; double p = 0.7, q = 0.3;
        std::mt19937_64 rng(7);
        std::uniform_int_distribution<int> bit(0, 1);
        for (int trial = 0; trial < 5; ++trial) {
            std::string x;
            for (int i = 0; i < n; ++i) x += char('0' + bit(rng));
            for (int D = 1; D <= 2; ++D)
                for (auto rm : enumerate_rank_sets(n, D)) {
                    auto ranks = mask_to_ranks(rm);
                    auto cl = channel_coefficients(ranks, n, p, q, 1e-14);
                    // polynomial evaluated at x = T_I(x)
                    double px = 0.0;
                    for (size_t t = 0; t < cl.tuples.size(); ++t) {
                        double v = cl.coef[t];
                        for (int i = 0; i < n; ++i)
                            if (cl.tuples[t] >> i & 1) v *= x[i] - '0';
                        px += v;
                    }
                    double ex = exact_TI(x, ranks, p);
                    CHECK(std::fabs(px - ex) < 1e-12, "coefficient polynomial vs exact T_I");
                }
        }
        printf("[1] channel coefficients match brute force\n");
    }
    // 2. Empirical means concentrate near exact T_I.
    {
        std::string x = "1011010010101101";
        double p = 0.8;
        auto traces = simulate_traces(x, p, 200000, 42);
        auto rsets = enumerate_rank_sets((int)x.size(), 1);
        auto means = estimate_rank_means(traces, rsets);
        double maxdiff = 0;
        for (size_t s = 0; s < rsets.size(); ++s) {
            double ex = exact_TI(x, mask_to_ranks(rsets[s]), p);
            maxdiff = std::max(maxdiff, std::fabs(means[s] - ex));
        }
        CHECK(maxdiff < 0.01, "empirical means near exact");
        printf("[2] empirical rank means: max |diff| = %.4f\n", maxdiff);
    }
    // 3. Oracle: decode known strings, require exact reconstruction.
    {
        std::mt19937_64 rng(99);
        std::uniform_int_distribution<int> bit(0, 1);
        int ok = 0, trials = 10;
        for (int t = 0; t < trials; ++t) {
            std::string x;
            for (int i = 0; i < 24; ++i) x += char('0' + bit(rng));
            auto traces = simulate_traces(x, 0.8, 50000, 1000 + t);
            DecodeOptions opt; opt.D = 1; opt.zeta = 0.02;
            auto res = decode(traces, (int)x.size(), 0.8, opt);
            if (res.out == x) ++ok;
            else printf("    miss trial %d: x=%s got=%s\n", t, x.c_str(), res.out.c_str());
        }
        printf("[3] D=1 oracle n=24 p=0.8 T=50000: %d/%d exact\n", ok, trials);
        CHECK(ok == trials, "D=1 oracle decode");
    }
    // 4. D=2 oracle on smaller n.
    {
        std::mt19937_64 rng(5);
        std::uniform_int_distribution<int> bit(0, 1);
        int ok = 0, trials = 5;
        for (int t = 0; t < trials; ++t) {
            std::string x;
            for (int i = 0; i < 16; ++i) x += char('0' + bit(rng));
            auto traces = simulate_traces(x, 0.7, 20000, 2000 + t);
            DecodeOptions opt; opt.D = 2; opt.zeta = 0.03;
            auto res = decode(traces, (int)x.size(), 0.7, opt);
            if (res.out == x) ++ok;
            else printf("    miss trial %d\n", t);
        }
        printf("[4] D=2 oracle n=16 p=0.7 T=20000: %d/%d exact\n", ok, trials);
        CHECK(ok == trials, "D=2 oracle decode");
    }
    if (failures) { printf("%d FAILURES\n", failures); return 1; }
    printf("ALL SELFTESTS PASSED\n");
    return 0;
}
