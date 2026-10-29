// bindings.cpp — pybind11 module `tr122`
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <string>
#include <vector>

#include "../src/channel.hpp"
#include "../src/decoder.hpp"

namespace py = pybind11;

namespace {

py::dict feas_to_dict(const tr122::FeasResult& fr) {
    py::dict d;
    d["accept"] = fr.accept;
    d["max_violation"] = fr.max_violation;
    d["min_eig"] = fr.min_eig;
    d["rounds"] = fr.rounds;
    return d;
}

// Low-level single-prefix feasibility (generic path, n<=64).
py::dict prefix_feasible(const std::vector<std::string>& traces, int n, double p,
                         const std::vector<int>& prefix, int D, double zeta,
                         double psd_tol, int pocs_rounds, const std::string& method) {
    double q = 1.0 - p;
    std::vector<tr122::Mask> rsets = tr122::enumerate_rank_sets(n, D);
    std::vector<double> tmeans = tr122::estimate_rank_means(traces, rsets);
    std::vector<tr122::CoeffList> cls(rsets.size());
    for (size_t s = 0; s < rsets.size(); ++s)
        cls[s] = tr122::channel_coefficients(tr122::mask_to_ranks(rsets[s]), n, p, q, 1e-10);
    tr122::GenericFeas sys(n, (int)prefix.size(), D);
    sys.build(cls, tmeans, prefix);
    tr122::FeasOptions fopt;
    fopt.method = method; fopt.zeta = zeta; fopt.psd_tol = psd_tol;
    fopt.pocs_rounds = pocs_rounds;
    return feas_to_dict(sys.solve(fopt));
}

py::list rank_means(const std::vector<std::string>& traces, int n, int D) {
    auto rsets = tr122::enumerate_rank_sets(n, D);
    auto means = tr122::estimate_rank_means(traces, rsets);
    py::list out;
    for (size_t i = 0; i < rsets.size(); ++i) {
        py::tuple item = py::make_tuple(tr122::mask_to_ranks(rsets[i]), means[i]);
        out.append(item);
    }
    return out;
}

py::dict decode_py(const std::vector<std::string>& traces, int n, double p, int D,
                   double zeta, double psd_tol, int pocs_rounds, int kaczmarz_sweeps,
                   int agmon_iters, const std::string& method) {
    tr122::DecodeOptions opt;
    opt.D = D; opt.zeta = zeta; opt.psd_tol = psd_tol;
    opt.pocs_rounds = pocs_rounds; opt.kaczmarz_sweeps = kaczmarz_sweeps;
    opt.agmon_iters = agmon_iters; opt.method = method;
    auto res = tr122::decode(traces, n, p, opt);
    py::dict out;
    out["out"] = res.out;
    py::list bits;
    for (auto& b : res.bits) {
        py::dict d;
        d["d"] = b.d; d["chosen"] = b.chosen;
        d["feas0"] = b.feas0; d["feas1"] = b.feas1;
        d["viol0"] = b.viol0; d["viol1"] = b.viol1;
        d["eig0"] = b.eig0; d["eig1"] = b.eig1;
        d["rounds0"] = b.rounds0; d["rounds1"] = b.rounds1;
        bits.append(d);
    }
    out["bits"] = bits;
    return out;
}

}  // namespace

PYBIND11_MODULE(tr122, m) {
    m.doc() = "Uniform quasipolynomial-time trace reconstruction (paper #122)";
    m.def("simulate_traces", &tr122::simulate_traces, py::arg("x"), py::arg("p"),
          py::arg("T"), py::arg("seed") = 0);
    m.def("rank_means", &rank_means, py::arg("traces"), py::arg("n"), py::arg("D"));
    m.def("prefix_feasible", &prefix_feasible, py::arg("traces"), py::arg("n"),
          py::arg("p"), py::arg("prefix"), py::arg("D") = 1, py::arg("zeta") = 0.1,
          py::arg("psd_tol") = 1e-9, py::arg("pocs_rounds") = 60,
          py::arg("method") = "pocs");
    m.def("decode", &decode_py, py::arg("traces"), py::arg("n"), py::arg("p"),
          py::arg("D") = 1, py::arg("zeta") = 0.1, py::arg("psd_tol") = 1e-9,
          py::arg("pocs_rounds") = 60, py::arg("kaczmarz_sweeps") = 3,
          py::arg("agmon_iters") = 4000, py::arg("method") = "pocs");
}
