"""Benchmarks: map the practical frontier of the #122 trace-reconstruction
decoder over (n, p, T, D). Writes CSV rows to bench/results.csv.

Each cell: `trials` random strings, decode, record exact-reconstruction
success and wall-clock time per decode.
"""
import csv
import os
import random
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "build"))
import tr122  # noqa: E402

OUT = os.path.join(os.path.dirname(__file__), "results.csv")


def run(n, p, T, D, zeta, trials, seed0=0):
    rows = []
    for t in range(trials):
        rng = random.Random(seed0 + 1000 * n + 10 * t + int(100 * p))
        x = "".join(rng.choice("01") for _ in range(n))
        traces = tr122.simulate_traces(x, p, T, seed0 + t + 1)
        t0 = time.time()
        res = tr122.decode(traces, n, p, D, zeta=zeta)
        dt = time.time() - t0
        exact = res["out"] == x
        errs = sum(1 for a, b in zip(res["out"], x) if a != b)
        rows.append(dict(n=n, p=p, T=T, D=D, zeta=zeta, trial=t, exact=int(exact),
                         bit_errors=errs, seconds=round(dt, 3)))
        print(f"n={n} p={p} T={T} D={D} trial={t}: "
              f"{'EXACT' if exact else f'{errs} errs'} {dt:.2f}s", flush=True)
    return rows


def done_set():
    """Parse bench.log for already-completed (n,p,T,D,trial) cells."""
    done = set()
    try:
        for line in open(os.path.join(os.path.dirname(__file__), "bench.log")):
            if "trial=" not in line:
                continue
            try:
                parts = dict(kv.split("=") for kv in
                             line.split(":", 1)[0].split())
                done.add((int(parts["n"]), float(parts["p"]),
                          int(parts["T"]), int(parts["D"]), int(parts["trial"])))
            except Exception:
                continue
    except FileNotFoundError:
        pass
    return done


def main():
    done = done_set()

    def go(n, p, T, D, zeta, trials):
        todo = [t for t in range(trials)
                if (n, p, T, D, t) not in done]
        for t in todo:
            run(n, p, T, D, zeta, 1, seed0=0)  # run() does its own trial loop

    # NOTE: run() loops trials internally; emulate skip by calling with 1 trial
    # filtered above (kept simple: skip only whole cells if fully done)
    for n in (16, 24, 32, 48, 64, 96, 128, 192, 256):
        ps = (0.6, 0.7, 0.8, 0.9, 0.95) if n <= 64 else (0.9, 0.95)
        for p in ps:
            Ts = (2000, 10000, 50000) if n <= 64 else (10000, 50000)
            for T in Ts:
                if all((n, p, T, 1, t) in done for t in range(3)):
                    continue
                run(n, p, T, 1, 0.02, 3)
    for n in (16, 24, 32):
        for p in (0.5, 0.6, 0.7, 0.8):
            for T in (5000, 20000, 50000):
                if all((n, p, T, 2, t) in done for t in range(3)):
                    continue
                run(n, p, T, 2, 0.03, 3)
    for n, p, T in ((48, 0.7, 50000), (64, 0.7, 50000), (48, 0.8, 50000)):
        run(n, p, T, 2, 0.03, 1)
    # finalize CSV from the full log
    rows = []
    for line in open(os.path.join(os.path.dirname(__file__), "bench.log")):
        if "trial=" not in line:
            continue
        try:
            head, rest = line.split(":", 1)
            parts = dict(kv.split("=") for kv in head.split())
            out = rest.strip().split()[0]
            secs = float(rest.strip().split()[-1].rstrip("s"))
            exact = 1 if out == "EXACT" else 0
            errs = 0 if exact else int(out.split()[0])
            rows.append(dict(n=int(parts["n"]), p=float(parts["p"]),
                             T=int(parts["T"]), D=int(parts["D"]),
                             zeta=0.02 if parts["D"] == "1" else 0.03,
                             trial=int(parts["trial"].rstrip(":")),
                             exact=exact, bit_errors=errs, seconds=secs))
        except Exception:
            continue
    seen, uniq = set(), []
    for r in rows:
        k = (r["n"], r["p"], r["T"], r["D"], r["trial"])
        if k in seen:
            continue
        seen.add(k)
        uniq.append(r)
    with open(OUT, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(uniq[0].keys()))
        w.writeheader()
        w.writerows(uniq)
    print(f"wrote {len(uniq)} rows to {OUT}")


if __name__ == "__main__":
    main()
