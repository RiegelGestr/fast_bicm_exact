#!/usr/bin/env python
"""Streamed exact BiCM benchmark on chunked parquet data (no pd.concat).

By default, generates a deterministic synthetic multi-chunk bipartite dataset
in a temporary directory and streams the parquet chunks through
aggregate_chunks -> Newton -> grouped exact p-values -> Bonferroni.
Pass --cache to benchmark an existing directory of parquet chunks with
row_id / col_id columns.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import time
from pathlib import Path

os.environ.setdefault("NEW_BICM_LOG", "1")

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from new_bicm.core import (
    aggregate_chunks,
    exact_pvals_grouped,
    fit_bicm_reduced,
    moments_grouped,
    rss_bytes,
)
from new_bicm.validate import validate_edges

GATE_S = 10.0 * 60.0


def generate_synthetic_parquets(
    out_dir: Path,
    *,
    n_chunks: int = 8,
    rows_per_chunk: int = 3_000,
    n_cols: int = 240,
    seed: int = 123,
) -> Path:
    """Create synthetic chunked parquet files with skewed degrees and planted blocks."""
    if n_chunks < 1:
        raise ValueError("n_chunks must be >= 1")
    if rows_per_chunk < 1:
        raise ValueError("rows_per_chunk must be >= 1")
    if n_cols < 3:
        raise ValueError("n_cols must be >= 3 to generate non-trivial co-occurrences")
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    ranks = np.arange(1, n_cols + 1, dtype=np.float64)
    probs = 1.0 / np.power(ranks + 1.5, 0.85)
    probs /= probs.sum()

    # Plant correlated column blocks when n_cols is large enough so Bonferroni
    # selects true excess co-occurrences.
    blocks = []
    if n_cols >= 24:
        blocks = [
            np.arange(0, 12, dtype=np.int64),
            np.arange(12, 24, dtype=np.int64),
        ]

    max_deg = min(24, max(2, n_cols))
    min_deg = min(4, max_deg - 1)

    for c_idx in range(n_chunks):
        r_offset = c_idx * rows_per_chunk
        row_parts = []
        col_parts = []
        degs = rng.integers(min_deg, max_deg, size=rows_per_chunk)
        for local_r in range(rows_per_chunk):
            d = int(degs[local_r])
            rid = r_offset + local_r
            u = rng.random()
            if blocks and u < 0.14:
                blk = blocks[0]
                k_blk = min(d, len(blk), int(rng.integers(min(4, d), min(9, d + 1))))
                chosen_blk = rng.choice(blk, size=k_blk, replace=False)
                rest_n = d - k_blk
                if rest_n > 0:
                    pool = np.setdiff1d(np.arange(n_cols), chosen_blk, assume_unique=True)
                    p_rest = probs[pool]
                    p_rest = p_rest / p_rest.sum()
                    chosen_rest = rng.choice(pool, size=rest_n, replace=False, p=p_rest)
                    cols = np.concatenate([chosen_blk, chosen_rest])
                else:
                    cols = chosen_blk
            elif len(blocks) > 1 and u < 0.24:
                blk = blocks[1]
                k_blk = min(d, len(blk), int(rng.integers(min(4, d), min(8, d + 1))))
                chosen_blk = rng.choice(blk, size=k_blk, replace=False)
                rest_n = d - k_blk
                if rest_n > 0:
                    pool = np.setdiff1d(np.arange(n_cols), chosen_blk, assume_unique=True)
                    p_rest = probs[pool]
                    p_rest = p_rest / p_rest.sum()
                    chosen_rest = rng.choice(pool, size=rest_n, replace=False, p=p_rest)
                    cols = np.concatenate([chosen_blk, chosen_rest])
                else:
                    cols = chosen_blk
            else:
                # Weighted sampling without replacement via exponential race.
                keys = rng.exponential(1.0, size=n_cols) / probs
                cols = np.argpartition(keys, d)[:d]
            row_parts.append(np.full(d, rid, dtype=np.int64))
            col_parts.append(cols.astype(np.int64, copy=False))

        df = pd.DataFrame(
            {
                "row_id": np.concatenate(row_parts),
                "col_id": np.concatenate(col_parts),
            }
        )
        df.to_parquet(out_dir / f"chunk_{c_idx:02d}.parquet", index=False)
    return out_dir


def iter_parquet_chunks(
    cache: Path,
    subset_path: Path | None = None,
    *,
    chunk_col: str = "chunk_id",
    row_col: str = "row_id",
    col_col: str = "col_id",
):
    files = sorted(Path(cache).glob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"no parquet files in {cache}")

    gb = None
    if subset_path is not None:
        sub = pd.read_parquet(subset_path)
        gb = sub.assign(**{chunk_col: sub[chunk_col].astype(str)}).groupby(
            chunk_col, sort=False
        )[row_col]

    n_hit = 0
    for path in files:
        id_set = None
        if gb is not None:
            try:
                ids = gb.get_group(path.stem)
            except KeyError:
                continue
            id_set = set(ids.to_numpy())
        df = pd.read_parquet(path, columns=[row_col, col_col])
        if id_set is not None:
            df = df[df[row_col].isin(id_set)]
        n_hit += 1
        rows = df[row_col].to_numpy()
        cols = df[col_col].to_numpy()
        del df, id_set
        if rows.size:
            yield rows, cols
    if gb is not None and n_hit == 0:
        raise RuntimeError(f"no chunk files matched {chunk_col!r} stems")


class _TimedIter:
    def __init__(self, inner):
        self.inner = inner
        self.io_s = 0.0
        self.n_chunks = 0
        self.n_records = 0

    def __iter__(self):
        it = iter(self.inner)
        while True:
            t0 = time.perf_counter()
            try:
                rows, cols = next(it)
            except StopIteration:
                return
            self.io_s += time.perf_counter() - t0
            self.n_chunks += 1
            self.n_records += int(len(rows))
            if self.n_chunks % 4 == 0 or self.n_chunks == 1:
                print(
                    f"  chunk {self.n_chunks} records={self.n_records:,} io={self.io_s:.2f}s",
                    flush=True,
                )
            yield rows, cols


def _run_pipeline(
    cache: Path,
    args: argparse.Namespace,
    *,
    is_synthetic: bool,
) -> dict:
    timed = _TimedIter(
        iter_parquet_chunks(
            cache,
            args.subset,
            chunk_col=args.chunk_col,
            row_col=args.row_col,
            col_col=args.col_col,
        )
    )
    t0 = time.perf_counter()
    agg = aggregate_chunks(timed)
    t_agg = time.perf_counter() - t0
    if agg is None:
        raise SystemExit("aggregate_chunks returned None")

    print(
        f"agg  {t_agg:.2f}s  io={timed.io_s:.2f}s  cpu={t_agg - timed.io_s:.2f}s  "
        f"chunks={timed.n_chunks} records={timed.n_records:,}  "
        f"N={agg.n_rows:,} C={agg.n_cols:,} pairs={len(agg.weight):,}  "
        f"G={len(agg.n_c)} k_max={int(agg.weight.max()):,}  "
        f"rss={rss_bytes()/1e6:.1f} MB",
        flush=True,
    )

    t1 = time.perf_counter()
    x_u, y = fit_bicm_reduced(agg.deg_g, agg.n_c, agg.col_deg)
    t_fit = time.perf_counter() - t1
    print(f"fit  {t_fit:.2f}s  rss={rss_bytes()/1e6:.1f} MB", flush=True)

    t_m = time.perf_counter()
    mu, var, skew = moments_grouped(x_u, agg.n_c, y, agg.pairs_i, agg.pairs_j)
    t_mom = time.perf_counter() - t_m
    print(f"moments {t_mom:.2f}s  rss={rss_bytes()/1e6:.1f} MB", flush=True)

    t2 = time.perf_counter()
    exact_pvals = exact_pvals_grouped(
        x_u, agg.n_c, y, agg.pairs_i, agg.pairs_j, agg.weight,
        n_jobs=args.n_jobs,
    )
    t_pvals = time.perf_counter() - t2
    print(f"pvals {t_pvals:.2f}s  rss={rss_bytes()/1e6:.1f} MB", flush=True)

    gate = t_agg + t_fit + t_mom + t_pvals
    print(
        f"GATE aggregate+fit+pvals {gate:.2f}s ({gate/60:.2f} min)  "
        f"{'PASS' if gate < GATE_S else 'OVER'} < {GATE_S/60:.0f} min",
        flush=True,
    )

    t3 = time.perf_counter()
    val = validate_edges(
        observed_weights=agg.weight,
        mu=mu,
        var=var,
        skew=skew,
        exact_pvals=exact_pvals,
        method="exact",
        correction_type="bonferroni",
        alpha=args.alpha,
    )
    t_bonf = time.perf_counter() - t3
    n_keep = int(np.sum(val["reject_mask"]))
    print(f"bonf {t_bonf:.2f}s  edges={n_keep:,}", flush=True)

    total = gate + t_bonf
    summary = {
        "synthetic": is_synthetic,
        "n_chunks": timed.n_chunks,
        "n_records": timed.n_records,
        "n_rows": int(agg.n_rows),
        "n_cols": int(agg.n_cols),
        "n_pairs": int(len(agg.weight)),
        "G": int(len(agg.n_c)),
        "k_max": int(agg.weight.max()),
        "n_validated": n_keep,
        "seconds": {
            "io": timed.io_s,
            "aggregate": t_agg,
            "aggregate_cpu": t_agg - timed.io_s,
            "fit": t_fit,
            "moments": t_mom,
            "pvals": t_pvals,
            "bonferroni": t_bonf,
            "gate_aggregate_fit_pvals": gate,
            "compute_no_io": (t_agg - timed.io_s) + t_fit + t_mom + t_pvals,
            "total": total,
        },
        "gate_pass": gate < GATE_S,
        "rss_bytes": rss_bytes(),
    }
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(summary, indent=2) + "\n")
    print(f"wrote {_rel(args.out_json)}", flush=True)

    if args.save_synthetic:
        hist_path = ROOT / "data" / "synthetic_k_hist.npz"
        fit_path = ROOT / "data" / "synthetic_fit.npz"
        hist_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            hist_path,
            weight=agg.weight.astype(np.int32),
            n_cols=np.int64(agg.n_cols),
            n_pairs=np.int64(len(agg.weight)),
        )
        np.savez_compressed(
            fit_path,
            x_u=x_u,
            n_c=agg.n_c,
            y=y,
            pairs_i=agg.pairs_i,
            pairs_j=agg.pairs_j,
        )
        print(f"wrote {_rel(hist_path)} and {_rel(fit_path)}", flush=True)

    if args.out_parquet is not None:
        args.out_parquet.parent.mkdir(parents=True, exist_ok=True)
        keep = val["reject_mask"]
        labels = agg.col_labels
        out = pd.DataFrame(
            {
                "col_i": labels[agg.pairs_i[keep]],
                "col_j": labels[agg.pairs_j[keep]],
                "weight": agg.weight[keep],
                "z_score": val["z_scores"][keep],
                "pval_raw": val["raw_target_pvals"][keep],
                "pval_corrected": val["corrected_pvals"][keep],
            }
        )
        out.to_parquet(args.out_parquet, index=False)
        print(f"wrote {_rel(args.out_parquet)}", flush=True)

    return summary


def _rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT))
    except ValueError:
        return str(path)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cache", type=Path, default=None)
    p.add_argument("--subset", type=Path, default=None)
    p.add_argument("--chunk-col", default="chunk_id")
    p.add_argument("--row-col", default="row_id")
    p.add_argument("--col-col", default="col_id")
    p.add_argument("--n-chunks", type=int, default=8)
    p.add_argument("--rows-per-chunk", type=int, default=3_000)
    p.add_argument("--n-cols", type=int, default=240)
    p.add_argument("--seed", type=int, default=123)
    p.add_argument("--n-jobs", type=int, default=None)
    p.add_argument("--alpha", type=float, default=0.001)
    p.add_argument(
        "--out-json",
        type=Path,
        default=ROOT / "data" / "streamed_exact_gate.json",
    )
    p.add_argument("--out-parquet", type=Path, default=None)
    p.add_argument("--save-synthetic", action="store_true")
    args = p.parse_args()

    if args.cache is not None:
        _run_pipeline(args.cache, args, is_synthetic=False)
    else:
        with tempfile.TemporaryDirectory() as tmp:
            cache = generate_synthetic_parquets(
                Path(tmp),
                n_chunks=args.n_chunks,
                rows_per_chunk=args.rows_per_chunk,
                n_cols=args.n_cols,
                seed=args.seed,
            )
            _run_pipeline(cache, args, is_synthetic=True)


if __name__ == "__main__":
    main()
