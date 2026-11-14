#!/usr/bin/env python
"""Stratified wall-clock probe for exact Poisson-binomial p-values on synthetic data."""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from new_bicm.core import exact_pvals_grouped, moments_grouped

DEFAULT_HIST = ROOT / "data" / "synthetic_k_hist.npz"
DEFAULT_FIT = ROOT / "data" / "synthetic_fit.npz"

BINS = [
    ("1-512", 1, 512, 512, 1),
    ("513-2000", 513, 2000, 512, 1),
    ("2001-10000", 2001, 10000, 256, 1),
    ("10001-50000", 10001, 50000, 64, 1),
    (">50000", 50001, 10**12, None, 1),
]


def _load_fit(fit_path: Path):
    with np.load(fit_path, allow_pickle=False) as z:
        x_u = np.asarray(z["x_u"], dtype=np.float64)
        n_c = np.asarray(z["n_c"], dtype=np.float64)
        y = np.asarray(z["y"], dtype=np.float64)
        pairs_i = np.asarray(z["pairs_i"], dtype=np.int64)
        pairs_j = np.asarray(z["pairs_j"], dtype=np.int64)
    return x_u, n_c, y, pairs_i, pairs_j


def _pair_slice(pool_i, pool_j, idx, ks):
    if len(pool_i) > int(idx.max()):
        return pool_i[idx], pool_j[idx], np.asarray(ks, dtype=np.int64)
    n = len(ks)
    return np.resize(pool_i, n), np.resize(pool_j, n), np.asarray(ks, dtype=np.int64)


def _time_call(x_u, n_c, y, pi, pj, ks, n_jobs):
    t0 = time.perf_counter()
    exact_pvals_grouped(x_u, n_c, y, pi, pj, ks, n_jobs=n_jobs)
    return time.perf_counter() - t0


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--hist", type=Path, default=DEFAULT_HIST)
    p.add_argument("--fit", type=Path, default=DEFAULT_FIT)
    p.add_argument(
        "--out-json",
        type=Path,
        default=ROOT / "data" / "probe_exact_timing.json",
    )
    p.add_argument("--n-jobs", type=int, default=None)
    args = p.parse_args()

    with np.load(args.hist, allow_pickle=False) as h:
        w = h["weight"].astype(np.int64)
    if w.size == 0:
        raise ValueError("histogram weight array is empty")
    n_pairs = int(len(w))
    x_u, n_c, y, pool_i, pool_j = _load_fit(args.fit)
    rng = np.random.default_rng(0)

    print(
        f"pairs={n_pairs:,}  k_mean={float(w.mean()):.2f}  "
        f"k_max={int(w.max()):,}  G={len(x_u)}  n_cols={len(y)}",
        flush=True,
    )

    warmup_k = np.concatenate(
        [
            np.array([1, 8, 64, 256, 512], dtype=np.int64),
            np.full(min(32, len(pool_i)), 800, dtype=np.int64),
        ]
    )
    wi = np.resize(pool_i, len(warmup_k))
    wj = np.resize(pool_j, len(warmup_k))
    _time_call(x_u, n_c, y, wi, wj, warmup_k, args.n_jobs)
    print("warmup done", flush=True)

    rows = []
    for name, lo, hi, n_samp, n_rep in BINS:
        idx = np.flatnonzero((w >= lo) & (w <= hi))
        pop = int(idx.size)
        if pop == 0:
            print(f"{name:16s} empty", flush=True)
            continue
        take = pop if n_samp is None else min(int(n_samp), pop)
        dts = []
        for rep in range(n_rep):
            if n_samp is None or take == pop:
                pick = idx
            else:
                pick = rng.choice(idx, size=take, replace=False)
            ks_rep = w[pick]
            pi, pj, ks_rep = _pair_slice(pool_i, pool_j, pick, ks_rep)
            dt = _time_call(x_u, n_c, y, pi, pj, ks_rep, args.n_jobs)
            dts.append(dt)
            print(
                f"{name:16s} rep={rep} pop={pop:8d} samp={take:4d} "
                f"k=[{int(ks_rep.min())},{int(ks_rep.max())}] "
                f"{dt:.3f}s  {1e3 * dt / take:.2f} ms/pair",
                flush=True,
            )
        per = np.array(dts, dtype=np.float64) / take
        ext = per * pop
        rows.append(
            {
                "bin": name,
                "pop": pop,
                "n_samp": take,
                "n_rep": n_rep,
                "ms_per_pair_mean": float(np.mean(per) * 1e3),
                "ms_per_pair_min": float(np.min(per) * 1e3),
                "ms_per_pair_max": float(np.max(per) * 1e3),
                "extrap_s_mean": float(np.mean(ext)),
                "extrap_s_min": float(np.min(ext)),
                "extrap_s_max": float(np.max(ext)),
            }
        )

    kernel_mean = float(sum(r["extrap_s_mean"] for r in rows))
    kernel_min = float(sum(r["extrap_s_min"] for r in rows))
    kernel_max = float(sum(r["extrap_s_max"] for r in rows))
    fft_mean = float(
        sum(r["extrap_s_mean"] for r in rows if r["bin"] != "1-512")
    )

    t0 = time.perf_counter()
    if len(pool_i) == n_pairs:
        moments_grouped(x_u, n_c, y, pool_i, pool_j)
        moments_s = time.perf_counter() - t0
    else:
        n_mom = min(30_000, n_pairs)
        moments_grouped(
            x_u, n_c, y, np.resize(pool_i, n_mom), np.resize(pool_j, n_mom)
        )
        moments_s = (time.perf_counter() - t0) / n_mom * n_pairs

    summary = {
        "synthetic": True,
        "n_pairs": n_pairs,
        "G": int(len(x_u)),
        "bins": rows,
        "kernel_pval_s": {
            "mean": kernel_mean,
            "min": kernel_min,
            "max": kernel_max,
        },
        "fft_tail_s_mean": fft_mean,
        "fft_frac_of_kernel": fft_mean / kernel_mean if kernel_mean else None,
        "moments_s": float(moments_s),
    }

    print("\n=== kernel exact p-values (Poisson-binomial grouped) ===", flush=True)
    for r in rows:
        print(
            f"  {r['bin']:16s}  {r['extrap_s_mean']:6.3f} s  "
            f"[{r['extrap_s_min']:.3f}, {r['extrap_s_max']:.3f}]",
            flush=True,
        )
    pct = (100 * fft_mean / kernel_mean) if kernel_mean else 0.0
    print(
        f"  TOTAL kernel     {kernel_mean:6.3f} s  "
        f"[{kernel_min:.3f}, {kernel_max:.3f}]  "
        f"FFT tail {pct:.1f}%",
        flush=True,
    )
    print(f"  moments grouped  {moments_s:6.3f} s", flush=True)

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(summary, indent=2) + "\n")
    try:
        out_label = str(args.out_json.resolve().relative_to(ROOT))
    except ValueError:
        out_label = str(args.out_json)
    print(f"\nwrote {out_label}", flush=True)


if __name__ == "__main__":
    main()
