#!/usr/bin/env python
"""Equivalence tests for new_bicm."""
from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import scipy.sparse as sp
import scipy.stats as stats

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from bicm import BipartiteGraph
from new_bicm import (
    aggregate_bipartite,
    aggregate_chunks,
    project,
    project_chunks,
    validate_edges,
)
from new_bicm.core import (
    HUGE_K,
    K_DIRECT,
    _binom_pmf_trunc,
    exact_pvals_direct,
    exact_pvals_grouped,
    exact_pvals_naive_one,
    expand_x_from_classes,
    fit_bicm_reduced,
    moments_grouped,
    rss_bytes,
)


def _random_bipartite(n_rows=80, n_cols=12, seed=0):
    rng = np.random.default_rng(seed)
    rows, cols = [], []
    for i in range(n_rows):
        d = int(rng.integers(2, 6))
        for c in rng.choice(n_cols, size=d, replace=False):
            rows.append(f"r{i}")
            cols.append(f"c{c}")
    return np.array(rows), np.array(cols)


def _legacy_clean_matrix(rows, cols):
    row_codes, row_u = pd.factorize(rows)
    col_codes, col_u = pd.factorize(cols)
    n_r, n_c = len(row_u), len(col_u)
    M = sp.csr_matrix(
        (np.ones(len(row_codes), dtype=np.uint8), (row_codes, col_codes)),
        shape=(n_r, n_c),
    )
    M.data[:] = 1
    row_deg = np.asarray(M.sum(axis=1)).ravel()
    col_deg = np.asarray(M.sum(axis=0)).ravel()
    valid_r = (row_deg > 0) & (row_deg < n_c)
    valid_c = (col_deg > 0) & (col_deg < n_r)
    return M[valid_r][:, valid_c]


def test_cooccurrence_counts_above_255():
    """Bipartite incidence may be uint8; W = M.T @ M must not wrap."""
    n_r = 300
    rows = np.concatenate([np.repeat(np.arange(n_r), 2), [n_r]])
    cols = np.concatenate([np.tile(np.array(["a", "b"]), n_r), ["c"]])
    agg = aggregate_bipartite(rows, cols, min_overlap=1)
    assert agg is not None
    assert agg.weight.size == 1
    assert int(agg.weight[0]) == n_r


def test_naive_vs_grouped_pvalues():
    rows, cols = _random_bipartite()
    agg = aggregate_bipartite(rows, cols)
    x_u, y = fit_bicm_reduced(agg.deg_g, agg.n_c, agg.col_deg)
    x_arr = expand_x_from_classes(agg.deg_g, agg.n_c, x_u)
    grouped = exact_pvals_grouped(
        x_u, agg.n_c, y, agg.pairs_i, agg.pairs_j, agg.weight, k_direct=K_DIRECT
    )
    naive = np.array(
        [
            exact_pvals_naive_one(x_arr, y[i], y[j], int(k))
            for i, j, k in zip(agg.pairs_i, agg.pairs_j, agg.weight)
        ]
    )
    abs_err = np.abs(grouped - naive)
    assert abs_err.max() < 1e-8, abs_err.max()


def test_fft_matches_schoolbook():
    rows, cols = _random_bipartite(n_rows=40, n_cols=8, seed=1)
    agg = aggregate_bipartite(rows, cols)
    x_u, y = fit_bicm_reduced(agg.deg_g, agg.n_c, agg.col_deg)
    keep = agg.weight >= 2
    if not np.any(keep):
        return
    school = exact_pvals_grouped(
        x_u, agg.n_c, y, agg.pairs_i[keep], agg.pairs_j[keep], agg.weight[keep],
        k_direct=10_000,
    )
    fft = exact_pvals_grouped(
        x_u, agg.n_c, y, agg.pairs_i[keep], agg.pairs_j[keep], agg.weight[keep],
        k_direct=1,
    )
    assert np.abs(school - fft).max() < 1e-6


def test_fft_parallel_matches_serial():
    rows, cols = _random_bipartite(n_rows=50, n_cols=10, seed=7)
    agg = aggregate_bipartite(rows, cols)
    x_u, y = fit_bicm_reduced(agg.deg_g, agg.n_c, agg.col_deg)
    keep = agg.weight >= 2
    if not np.any(keep):
        return
    kwargs = dict(
        x_u=x_u,
        n_c=agg.n_c,
        y=y,
        pairs_i=agg.pairs_i[keep],
        pairs_j=agg.pairs_j[keep],
        weight=agg.weight[keep],
        k_direct=1,
    )
    serial = exact_pvals_grouped(**kwargs, n_jobs=1)
    parallel = exact_pvals_grouped(**kwargs, n_jobs=4)
    assert np.abs(serial - parallel).max() < 1e-10


def test_moments_match_naive_expansion():
    rows, cols = _random_bipartite(seed=2)
    agg = aggregate_bipartite(rows, cols)
    x_u, y = fit_bicm_reduced(agg.deg_g, agg.n_c, agg.col_deg)
    mu, var, _ = moments_grouped(x_u, agg.n_c, y, agg.pairs_i, agg.pairs_j)
    x_arr = expand_x_from_classes(agg.deg_g, agg.n_c, x_u)
    mu2 = np.zeros_like(mu)
    var2 = np.zeros_like(var)
    for t in range(len(mu)):
        yi, yj = y[agg.pairs_i[t]], y[agg.pairs_j[t]]
        m = v = 0.0
        for xc in x_arr:
            p = (xc * yi) / (1.0 + xc * yi) * (xc * yj) / (1.0 + xc * yj)
            m += p
            v += p * (1.0 - p)
        mu2[t], var2[t] = m, v
    assert np.max(np.abs(mu - mu2) / np.maximum(mu2, 1e-12)) < 1e-10
    assert np.max(np.abs(var - var2) / np.maximum(var2, 1e-12)) < 1e-10


def test_fitness_vs_bicm_package():
    rows, cols = _random_bipartite(n_rows=60, n_cols=10, seed=3)
    agg = aggregate_bipartite(rows, cols)
    x_u, y = fit_bicm_reduced(agg.deg_g, agg.n_c, agg.col_deg)
    M = _legacy_clean_matrix(rows, cols)
    graph = BipartiteGraph(biadjacency=M)
    graph.solve_tool(
        method="newton", tol=1e-9, verbose=False, light_mode=True, print_error=False
    )
    xd, yd = graph.get_bicm_fitnesses()
    x_pkg = np.array([xd[i] for i in range(len(xd))], dtype=np.float64)
    y_pkg = np.array([yd[i] for i in range(len(yd))], dtype=np.float64)
    row_deg = np.asarray(M.sum(axis=1)).ravel().astype(np.int64)
    x_by_deg = {}
    for d, xv in zip(row_deg, x_pkg):
        x_by_deg.setdefault(int(d), []).append(xv)
    for deg, xu in zip(agg.deg_g, x_u):
        ref = np.mean(x_by_deg[int(deg)])
        assert abs(xu - ref) / max(abs(ref), 1e-12) < 1e-4
    assert (np.abs(y - y_pkg) / np.maximum(np.abs(y_pkg), 1e-12)).max() < 1e-4


def test_project_never_builds_bicm_graph():
    rows, cols = _random_bipartite(n_rows=50, n_cols=10, seed=4)
    with patch("bicm.BipartiteGraph") as mock_g:
        out = project(rows, cols, method="exact")
        mock_g.assert_not_called()
    assert out is not None
    assert {"col_i", "col_j", "weight", "z_score", "pval_raw", "pval_corrected"} <= set(out.columns)


def test_rss_scale_not_linear_in_n():
    rows, cols = _random_bipartite(n_rows=4000, n_cols=30, seed=8)
    agg = aggregate_bipartite(rows, cols)
    x_u, y = fit_bicm_reduced(agg.deg_g, agg.n_c, agg.col_deg)
    assert x_u.size == agg.n_c.size
    assert y.size == agg.n_cols
    assert x_u.size * 50 < agg.n_rows
    payload = (
        agg.n_c.nbytes + agg.deg_g.nbytes + agg.col_deg.nbytes
        + agg.pairs_i.nbytes + agg.pairs_j.nbytes + agg.weight.nbytes
        + x_u.nbytes + y.nbytes
    )
    assert payload < 2_000_000, payload


def _weight_map(agg):
    lab = np.asarray(agg.col_labels).astype(str)
    return {
        tuple(sorted((lab[i], lab[j]))): int(w)
        for i, j, w in zip(agg.pairs_i, agg.pairs_j, agg.weight)
    }


def _col_deg_map(agg):
    lab = np.asarray(agg.col_labels).astype(str)
    return {lab[i]: int(agg.col_deg[i]) for i in range(len(lab))}


def test_chunks_match_aggregate_bipartite():
    rows, cols = _random_bipartite(n_rows=120, n_cols=15, seed=11)
    rows = np.concatenate([rows, np.repeat(np.arange(300) + 10_000, 2), [10_300]])
    cols = np.concatenate([cols, np.tile(np.array(["a", "b"]), 300), ["c"]])
    full = aggregate_bipartite(rows, cols, min_overlap=1)
    df = pd.DataFrame({"r": rows, "c": cols})
    chunks = [
        (g["r"].to_numpy(), g["c"].to_numpy())
        for _, g in df.groupby(df["r"].astype(str).str[-1], sort=False)
    ]
    streamed = aggregate_chunks(chunks, min_overlap=1)
    assert full is not None and streamed is not None
    assert _weight_map(full) == _weight_map(streamed)
    assert _col_deg_map(full) == _col_deg_map(streamed)
    assert dict(zip(full.deg_g.tolist(), full.n_c.tolist())) == dict(
        zip(streamed.deg_g.tolist(), streamed.n_c.tolist())
    )
    assert full.n_rows == streamed.n_rows
    assert full.n_edges == streamed.n_edges


def test_project_chunks_matches_project():
    rows, cols = _random_bipartite(n_rows=70, n_cols=11, seed=6)
    ram = project(rows, cols, method="exact", n_jobs=2)
    df = pd.DataFrame({"r": rows, "c": cols})
    chunks = [
        (g["r"].to_numpy(), g["c"].to_numpy())
        for _, g in df.groupby("r", sort=False)
    ]
    streamed = project_chunks(chunks, method="exact", n_jobs=2)

    def keys(frame):
        if frame is None or frame.empty:
            return set()
        return {
            tuple(sorted((str(a), str(b))))
            for a, b in zip(frame.col_i, frame.col_j)
        }

    assert keys(ram) == keys(streamed)


def test_jaccard_vs_naive_backbone():
    rows, cols = _random_bipartite(n_rows=70, n_cols=11, seed=6)
    out = project(rows, cols, method="exact", correction="bonferroni", alpha=0.001)
    agg = aggregate_bipartite(rows, cols)
    x_u, y = fit_bicm_reduced(agg.deg_g, agg.n_c, agg.col_deg)
    x_arr = expand_x_from_classes(agg.deg_g, agg.n_c, x_u)
    from statsmodels.stats.multitest import multipletests

    p_naive = np.array(
        [
            exact_pvals_naive_one(x_arr, y[i], y[j], int(k))
            for i, j, k in zip(agg.pairs_i, agg.pairs_j, agg.weight)
        ]
    )
    rej, *_ = multipletests(p_naive, alpha=0.001, method="bonferroni")
    labels = agg.col_labels
    naive_keys = {
        tuple(sorted((str(labels[i]), str(labels[j]))))
        for i, j, keep in zip(agg.pairs_i, agg.pairs_j, rej)
        if keep
    }
    if out is None or out.empty:
        assert not naive_keys
        return
    got = {
        tuple(sorted((str(a), str(b))))
        for a, b in zip(out.col_i, out.col_j)
    }
    union = naive_keys | got
    jacc = 1.0 if not union else len(naive_keys & got) / len(union)
    assert jacc == 1.0, (jacc, len(got), len(naive_keys))


def _one_class_pval(n, p, k, k_direct=10**9, n_jobs=1):
    """Poisson-binomial that is exactly Bin(n, p_both), p_both ≈ p."""
    s = np.sqrt(p)
    yv = s / (1.0 - s)
    link = (yv) / (1.0 + yv)
    p_both = float(link * link)
    x_u = np.array([1.0])
    n_c = np.array([float(n)])
    y = np.array([yv, yv])
    got = exact_pvals_grouped(
        x_u,
        n_c,
        y,
        np.array([0], dtype=np.int64),
        np.array([1], dtype=np.int64),
        np.array([k], dtype=np.int64),
        k_direct=k_direct,
        n_jobs=n_jobs,
    )
    return float(got[0]), p_both, x_u, n_c, y


def _trunc_sf(n, p, k):
    """Right tail from the truncated PMF itself: 1 - sum_{j<k} P(X=j)."""
    out = np.zeros(k, dtype=np.float64)
    _binom_pmf_trunc(float(n), float(p), int(k), out)
    return float(1.0 - out.sum())


def test_binom_pmf_trunc_loader_vs_scipy():
    cases = [
        (20_000_000, 0.001, 20_950, 1e-15),
        (10_000, 0.1, 1_100, 1e-15),
        (1000, 0.9999, 1005, 0.0),
        (1000, 0.999, 1000, 1e-15),
        (1000, 0.9995, 1000, 1e-15),
        (100, 0.991, 99, 1e-15),
        (10, 0.955, 10, 1e-15),
        (1000, 0.8, 5, 1e-15),
    ]
    for n, p, k, tol in cases:
        ref = float(stats.binom.sf(k - 1, n, p))
        pmf_sf = _trunc_sf(n, p, k)
        err = abs(pmf_sf - ref)
        assert err <= tol, (n, p, k, pmf_sf, ref, err)
        # One-class grouped path uses the same PMF. Schoolbook convolution of a
        # length-k vector adds a few ulps; a large-scale Bonferroni threshold is ~1e-10.
        got, p_both, *_ = _one_class_pval(n, p, k)
        ref_p = float(stats.binom.sf(k - 1, n, p_both))
        assert abs(got - ref_p) <= 1e-14, (n, p, k, got, ref_p)


def test_underflow_no_longer_returns_one():
    n, p, k = 10_000, 0.1, 1100
    got, _p_both, x_u, n_c, y = _one_class_pval(n, p, k)
    assert got != 1.0
    assert got < 0.5
    x_arr = expand_x_from_classes(np.array([0.0]), n_c, x_u)
    naive = float(exact_pvals_naive_one(x_arr, y[0], y[1], k))
    err = abs(got - naive)
    assert err < 1e-12 or err / max(abs(naive), 1e-300) < 1e-8, (got, naive, err)


def test_exact_pvals_direct_k_zero():
    out = exact_pvals_direct(
        np.array([0.2, 0.5]),
        np.array([3.0, 4.0]),
        np.array([0.3, 0.4]),
        np.array([0], dtype=np.int64),
        np.array([1], dtype=np.int64),
        np.array([0], dtype=np.int64),
        1,
    )
    assert out[0] == 1.0


def test_loader_fft_matches_scipy():
    n, p, k = 10_000, 0.1, 1_100
    for n_jobs in (1, 2):
        got, p_both, *_ = _one_class_pval(n, p, k, k_direct=K_DIRECT, n_jobs=n_jobs)
        ref = float(stats.binom.sf(k - 1, n, p_both))
        assert abs(got - ref) <= 1e-14, (n_jobs, got, ref, p_both)


def test_empty_and_degenerate_inputs():
    empty = np.array([], dtype=object)
    assert aggregate_bipartite(empty, empty) is None
    assert project(empty, empty) is None
    assert aggregate_chunks([]) is None
    assert project_chunks([(empty, empty)]) is None
    assert aggregate_bipartite(["r0"], ["c0"]) is None
    rows, cols = _random_bipartite(n_rows=30, n_cols=6, seed=9)
    assert aggregate_bipartite(rows, cols, min_overlap=10_000) is None
    assert project(rows, cols, min_overlap=10_000) is None


def test_duplicate_edges_and_full_drop_match():
    rows = np.array([
        "r0", "r0", "r0", "r0",
        "r1", "r1", "r1",
        "r2", "r2", "r2",
        "r3", "r3",
        "r4", "r4", "r4", "r4",
    ])
    cols = np.array([
        "c0", "c0", "c1", "c2",
        "c0", "c1", "c2",
        "c0", "c2", "c3",
        "c0", "c3",
        "c0", "c1", "c2", "c3",
    ])
    full = aggregate_bipartite(rows, cols)
    chunks = [
        (rows[:7], cols[:7]),
        (np.array([], dtype=str), np.array([], dtype=str)),
        (rows[7:], cols[7:]),
    ]
    streamed = aggregate_chunks(chunks)
    assert full is not None and streamed is not None
    assert _weight_map(full) == _weight_map(streamed)
    assert _col_deg_map(full) == _col_deg_map(streamed)
    assert dict(zip(full.deg_g.tolist(), full.n_c.tolist())) == dict(
        zip(streamed.deg_g.tolist(), streamed.n_c.tolist())
    )
    assert full.n_rows == streamed.n_rows
    assert full.n_cols == streamed.n_cols


def test_project_methods_and_group_col():
    rows, cols = _random_bipartite(n_rows=60, n_cols=10, seed=5)
    for method in ("standard", "rna", "exact"):
        out = project(rows, cols, method=method, alpha=1.0, group_id="g0", group_col="grp")
        assert out is not None
        assert list(out.columns) == [
            "grp", "col_i", "col_j", "weight", "z_score", "pval_raw", "pval_corrected"
        ]
        assert (out["grp"] == "g0").all()


def test_invalid_method_and_all_full_drop():
    import pytest

    rows, cols = _random_bipartite(n_rows=30, n_cols=6, seed=10)
    with pytest.raises(ValueError, match="unknown method"):
        project(rows, cols, method="invalid")

    # Complete 2x2 bipartite graph: every row and column is full -> pruned to None.
    full_rows = np.array(["r0", "r0", "r1", "r1"])
    full_cols = np.array(["c0", "c1", "c0", "c1"])
    assert aggregate_bipartite(full_rows, full_cols) is None
    assert aggregate_chunks([(full_rows, full_cols)]) is None


def test_huge_k_multiprocessing_matches_scipy():
    """Exercise the HUGE_K ProcessPoolExecutor('spawn') path across nfft buckets."""
    n = 250_000
    p = 0.5
    s = np.sqrt(p)
    yv = s / (1.0 - s)
    p_both = float((yv / (1.0 + yv)) ** 2)
    x_u = np.array([1.0], dtype=np.float64)
    n_c = np.array([float(n)], dtype=np.float64)
    y = np.array([yv, yv, yv], dtype=np.float64)
    pi = np.array([0, 0, 1], dtype=np.int64)
    pj = np.array([1, 2, 2], dtype=np.int64)
    # 125_000 -> nfft=262_144; 132_000 -> nfft=524_288 (two huge FFT chunks).
    w = np.array([HUGE_K + 25_000, HUGE_K + 25_200, HUGE_K + 32_000], dtype=np.int64)
    got = exact_pvals_grouped(x_u, n_c, y, pi, pj, w, n_jobs=2)
    ref = np.array([float(stats.binom.sf(int(k) - 1, n, p_both)) for k in w])
    assert np.max(np.abs(got - ref)) < 1e-13


def test_validate_edges_edge_cases():
    import warnings
    import pytest

    empty = np.array([], dtype=np.float64)
    out_empty = validate_edges(empty, empty, empty, empty, exact_pvals=empty)
    assert out_empty["reject_mask"].shape == (0,)
    assert out_empty["z_scores"].shape == (0,)

    with pytest.raises(ValueError, match="exact_pvals is required"):
        validate_edges(np.array([1.0]), np.array([0.5]), np.array([0.2]), np.array([0.0]))

    w = np.array([2.0, 0.0, 1.0, 3.0])
    mu = np.array([0.0, 1.0, 1.0, 1.5])
    var = np.array([0.0, 0.0, 0.0, 0.5])
    skew = np.array([0.0, 0.0, 0.0, 0.1])
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", RuntimeWarning)
        for method in ("standard", "rna"):
            res = validate_edges(w, mu, var, skew, method=method)
            assert not np.any(np.isnan(res["z_scores"]))
            assert not np.any(np.isnan(res["raw_target_pvals"]))
            assert not np.any(np.isnan(res["corrected_pvals"]))
            assert bool(res["reject_mask"][0]) is True
            assert res["raw_target_pvals"][0] == 0.0
            assert res["raw_target_pvals"][1] == 1.0
    assert not [c for c in caught if issubclass(c.category, RuntimeWarning)]


def test_aggregation_input_validation_and_min_overlap():
    import pytest

    rows = np.array(["r0", "r0", "r1", "r1", "r2", "r2", "r3", "r3"])
    cols = np.array(["c0", "c1", "c0", "c2", "c1", "c2", "c3", "c4"])
    a_ram = aggregate_bipartite(rows, cols, min_overlap=0)
    a_chk = aggregate_chunks([(rows, cols)], min_overlap=0)
    assert a_ram is not None and a_chk is not None
    assert _weight_map(a_ram) == _weight_map(a_chk)

    with pytest.raises(ValueError, match="same length"):
        aggregate_bipartite(["r0", "r1"], ["c0"])
    with pytest.raises(ValueError, match="same length"):
        aggregate_chunks([(["r0", "r1"], ["c0"])])
    with pytest.raises(ValueError, match="missing values"):
        aggregate_bipartite(["r0", None, "r1"], ["c0", "c1", "c0"])
    with pytest.raises(ValueError, match="missing values"):
        aggregate_chunks([(["r0", "r0"], np.array(["c0", np.nan], dtype=object))])


def test_rss_bytes_positive():
    assert rss_bytes() > 0


def test_streamed_benchmark_helpers():
    import importlib.util
    import tempfile
    import pytest

    if importlib.util.find_spec("pyarrow") is None and importlib.util.find_spec("fastparquet") is None:
        return
    sys.path.insert(0, str(ROOT / "scripts"))
    try:
        from run_streamed_exact import generate_synthetic_parquets, iter_parquet_chunks
    finally:
        sys.path.pop(0)

    with tempfile.TemporaryDirectory() as td:
        base = Path(td)
        with pytest.raises(ValueError, match="n_cols must be >= 3"):
            generate_synthetic_parquets(base / "unused", n_cols=2)
        with pytest.raises(ValueError, match="n_chunks must be >= 1"):
            generate_synthetic_parquets(base / "unused", n_chunks=0)
        with pytest.raises(ValueError, match="rows_per_chunk must be >= 1"):
            generate_synthetic_parquets(base / "unused", rows_per_chunk=0)

        cache = generate_synthetic_parquets(
            base / "chunks", n_chunks=2, rows_per_chunk=20, n_cols=10, seed=7
        )
        chunks = list(iter_parquet_chunks(cache))
        assert len(chunks) == 2

        # Subset matching works even when chunk_id is integer matching numeric stem.
        p0 = pd.read_parquet(cache / "chunk_00.parquet")
        int_cache = base / "int_chunks"
        int_cache.mkdir()
        p0.to_parquet(int_cache / "0.parquet", index=False)
        sub_path = base / "subset.parquet"
        pd.DataFrame({"chunk_id": [0], "row_id": [int(p0["row_id"].iloc[0])]}).to_parquet(
            sub_path, index=False
        )
        sub_chunks = list(iter_parquet_chunks(int_cache, subset_path=sub_path))
        assert len(sub_chunks) == 1
        assert np.all(sub_chunks[0][0] == int(p0["row_id"].iloc[0]))


def main():
    tests = [
        test_cooccurrence_counts_above_255,
        test_moments_match_naive_expansion,
        test_naive_vs_grouped_pvalues,
        test_fft_matches_schoolbook,
        test_fft_parallel_matches_serial,
        test_fitness_vs_bicm_package,
        test_project_never_builds_bicm_graph,
        test_rss_scale_not_linear_in_n,
        test_chunks_match_aggregate_bipartite,
        test_project_chunks_matches_project,
        test_jaccard_vs_naive_backbone,
        test_binom_pmf_trunc_loader_vs_scipy,
        test_underflow_no_longer_returns_one,
        test_exact_pvals_direct_k_zero,
        test_loader_fft_matches_scipy,
        test_empty_and_degenerate_inputs,
        test_duplicate_edges_and_full_drop_match,
        test_project_methods_and_group_col,
        test_invalid_method_and_all_full_drop,
        test_huge_k_multiprocessing_matches_scipy,
        test_validate_edges_edge_cases,
        test_aggregation_input_validation_and_min_overlap,
        test_rss_bytes_positive,
        test_streamed_benchmark_helpers,
    ]
    for fn in tests:
        print(f"RUN {fn.__name__}", flush=True)
        fn()
        print(f"OK  {fn.__name__}", flush=True)
    print("ALL_PASSED", flush=True)


if __name__ == "__main__":
    main()
