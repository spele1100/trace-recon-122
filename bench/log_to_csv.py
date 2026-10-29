"""Convert bench.log lines into bench/results.csv (idempotent; keeps first
occurrence of each (n,p,T,D,trial) cell)."""
import csv
import os
import re

HERE = os.path.dirname(__file__)
rows, seen = [], set()
pat = re.compile(
    r"n=(\d+) p=([\d.]+) T=(\d+) D=(\d+) trial=(\d+): (EXACT|\d+ errs) ([\d.]+)s")
for line in open(os.path.join(HERE, "bench.log")):
    m = pat.match(line.strip())
    if not m:
        continue
    n, p, T, D, trial, out, secs = m.groups()
    k = (int(n), float(p), int(T), int(D), int(trial))
    if k in seen:
        continue
    seen.add(k)
    rows.append(dict(n=int(n), p=float(p), T=int(T), D=int(D),
                     zeta=0.02 if int(D) == 1 else 0.03, trial=int(trial),
                     exact=1 if out == "EXACT" else 0,
                     bit_errors=0 if out == "EXACT" else int(out.split()[0]),
                     seconds=float(secs)))
rows.sort(key=lambda r: (r["D"], r["n"], r["p"], r["T"], r["trial"]))
out_path = os.path.join(HERE, "results.csv")
with open(out_path, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    w.writeheader()
    w.writerows(rows)
print(f"wrote {len(rows)} rows to {out_path}")
