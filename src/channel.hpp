// tr122/channel.hpp — deletion-channel trace simulator.
// Each input position is retained independently with probability p (retention),
// deleted with probability q = 1-p. Trace = concatenation of retained bits.
#pragma once
#include <cstdint>
#include <random>
#include <string>
#include <vector>

namespace tr122 {

inline std::vector<std::string> simulate_traces(const std::string& x, double p,
                                                int64_t T, uint64_t seed = 0) {
    std::mt19937_64 rng(seed ? seed : 0x9E3779B97F4A7C15ull);
    // binomial-per-bit via uniform threshold
    std::uniform_real_distribution<double> unif(0.0, 1.0);
    std::vector<std::string> out;
    out.reserve((size_t)T);
    for (int64_t t = 0; t < T; ++t) {
        std::string s;
        s.reserve(x.size());
        for (char c : x)
            if (unif(rng) < p) s.push_back(c);
        out.push_back(std::move(s));
    }
    return out;
}

}  // namespace tr122
