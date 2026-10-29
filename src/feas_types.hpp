#pragma once
#include <string>
namespace tr122 {
struct FeasOptions {
    std::string method = "pocs";
    double zeta = 0.1;
    double psd_tol = 1e-9;
    int pocs_rounds = 60;
    int kaczmarz_sweeps = 3;
    int agmon_iters = 4000;
    double agmon_lambda_scale = 1.0;
};
struct FeasResult {
    bool accept = false;
    double max_violation = 1e18;
    double min_eig = 1e18;
    int rounds = 0;
};
}
