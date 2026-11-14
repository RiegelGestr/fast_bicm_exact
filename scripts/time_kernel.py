#!/usr/bin/env python
"""Time grouped exact p-values on the synthetic k histogram (no parquet I/O)."""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("NEW_BICM_LOG", "1")

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from new_bicm.core import exact_pvals_grouped, moments_grouped

DEFAULT_FIT = ROOT / "data" / "synthetic_fit.npz"
DEFAULT_HIST = ROOT / "data" / "synthetic_k_hist.npz"


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--fit", type=Path, default=DEFAULT_FIT)
    p.add_argument("--hist", type=Path, default=DEFAULT_HIST)
    p.add_argument("--n-jobs", type=int, default=None)
    args = p.parse_args()

    with np.load(args.fit, allow_pickle=False) as z:
        x_u = np.asarray(z["x_u"], dtype=np.float64)
        n_c = np.asarray(z["n_c"], dtype=np.float64)
        y = np.asarray(z["y"], dtype=np.float64)
        pi = np.asarray(z["pairs_i"], dtype=np.int64)
        pj = np.asarray(z["pairs_j"], dtype=np.int64)
    with np.load(args.hist, allow_pickle=False) as h:
        w = h["weight"].astype(np.int64)
    if w.size == 0:
        raise ValueError("histogram weight array is empty")
    if len(pi) != len(w):
        pi = np.resize(pi, len(w)) % len(y)
        pj = np.resize(pj, len(w)) % len(y)
        same = pi == pj
        pj[same] = (pj[same] + 1) % len(y)

    print(
        f"pairs={len(w):,} k_max={int(w.max()):,} G={len(x_u)} C={len(y)} "
        f"n>512={int((w > 512).sum())} n>1e5={int((w > 100000).sum())}",
        flush=True,
    )
    t0 = time.perf_counter()
    moments_grouped(x_u, n_c, y, pi, pj)
    print(f"moments {time.perf_counter() - t0:.2f}s", flush=True)
    t1 = time.perf_counter()
    exact_pvals_grouped(x_u, n_c, y, pi, pj, w, n_jobs=args.n_jobs)
    print(f"pvals {time.perf_counter() - t1:.2f}s", flush=True)


if __name__ == "__main__":
    main()
