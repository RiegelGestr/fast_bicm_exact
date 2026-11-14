"""Degree-class aggregation, reduced Newton, grouped exact Poisson-binomial."""
from __future__ import annotations

import math
import multiprocessing as mp
import os
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from dataclasses import dataclass

import numpy as np
import pandas as pd
import scipy.sparse as sp
from numba import get_num_threads, get_thread_id, njit, prange

import bicm.models_functions as bmof
import bicm.solver_functions as bsof

K_DIRECT = 512
SOLVER_TOL = 1e-9
SOLVER_EPS = 1e-8
HUGE_K = 100_000
FFT_CHUNK_CAP = 256
FFT_CHUNK_BYTES = 24 * 1024 * 1024


def rss_bytes() -> int:
    import resource
    import sys

    scale = 1 if sys.platform == "darwin" else 1024
    return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * scale


@dataclass
class BipartiteAgg:
    n_c: np.ndarray
    deg_g: np.ndarray
    col_deg: np.ndarray
    pairs_i: np.ndarray
    pairs_j: np.ndarray
    weight: np.ndarray
    col_labels: np.ndarray
    n_rows: int
    n_cols: int
    n_edges: int


def aggregate_bipartite(
    row_ids,
    col_ids,
    min_overlap: int = 1,
) -> BipartiteAgg | None:
    row_arr = np.asarray(row_ids)
    col_arr = np.asarray(col_ids)
    if row_arr.shape != col_arr.shape or row_arr.ndim != 1:
        raise ValueError("row_ids and col_ids must be 1D arrays of the same length")
    row_codes, row_uniques = pd.factorize(row_arr, sort=False)
    del row_uniques
    col_codes, col_uniques = pd.factorize(col_arr, sort=False)
    if (row_codes < 0).any() or (col_codes < 0).any():
        raise ValueError("row_ids and col_ids must not contain missing values")
    row_codes = row_codes.astype(np.int32, copy=False)
    col_codes = col_codes.astype(np.int32, copy=False)

    n_r = int(row_codes.max()) + 1 if row_codes.size else 0
    n_m = int(len(col_uniques))
    if n_r == 0 or n_m == 0:
        return None

    M = sp.csr_matrix(
        (np.ones(len(row_codes), dtype=np.uint8), (row_codes, col_codes)),
        shape=(n_r, n_m),
    )
    del row_codes, col_codes
    if M.nnz == 0:
        return None
    M.data[:] = 1

    row_deg = np.asarray(M.sum(axis=1)).ravel()
    col_deg = np.asarray(M.sum(axis=0)).ravel()
    valid_r = (row_deg > 0) & (row_deg < n_m)
    valid_c = (col_deg > 0) & (col_deg < n_r)
    if not np.any(valid_r) or not np.any(valid_c):
        return None

    M = M[valid_r][:, valid_c]
    col_labels = np.asarray(col_uniques)[np.where(valid_c)[0]]
    del col_uniques, row_deg, col_deg

    if M.nnz == 0:
        return None

    row_deg = np.asarray(M.sum(axis=1)).ravel().astype(np.int64)
    col_deg = np.asarray(M.sum(axis=0)).ravel().astype(np.int64)
    n_rows = int(M.shape[0])
    n_cols = int(M.shape[1])
    n_edges = int(row_deg.sum())

    deg_g, n_c = np.unique(row_deg, return_counts=True)
    deg_g = deg_g.astype(np.int64)
    n_c = n_c.astype(np.float64)
    del row_deg

    # Incidence can stay uint8; co-occurrence must not. uint8 matmul wraps at 255
    # when co-occurrence counts exceed 255, so promote to int64 before M.T @ M.
    W = (M.astype(np.int64).T @ M.astype(np.int64)).tocoo()
    del M
    diag = W.row != W.col
    mask = (W.data >= min_overlap) & (W.row < W.col) & diag
    pairs_i = W.row[mask].astype(np.int64)
    pairs_j = W.col[mask].astype(np.int64)
    weight = W.data[mask].astype(np.int64)
    del W
    if pairs_i.size == 0:
        return None

    return BipartiteAgg(
        n_c=n_c,
        deg_g=deg_g,
        col_deg=col_deg,
        pairs_i=pairs_i,
        pairs_j=pairs_j,
        weight=weight,
        col_labels=col_labels,
        n_rows=n_rows,
        n_cols=n_cols,
        n_edges=n_edges,
    )


def _pairs_from_W(W: np.ndarray, min_overlap: int):
    n_m = int(W.shape[0])
    iu, ju = np.triu_indices(n_m, k=1)
    w = W[iu, ju]
    keep = w >= max(1, int(min_overlap))
    if not np.any(keep):
        return None
    return iu[keep].astype(np.int64), ju[keep].astype(np.int64), w[keep].astype(np.int64)


def _agg_from_hist_and_W(
    row_deg_vals: np.ndarray,
    row_deg_counts: np.ndarray,
    col_deg: np.ndarray,
    labels: np.ndarray,
    W: np.ndarray,
    min_overlap: int,
) -> BipartiteAgg | None:
    """Empty/full drop from a row-degree histogram (no length-N degree vector)."""
    row_deg_vals = np.asarray(row_deg_vals, dtype=np.int64)
    row_deg_counts = np.asarray(row_deg_counts, dtype=np.int64)
    col_deg = np.asarray(col_deg, dtype=np.int64)
    n_r = int(row_deg_counts.sum()) if row_deg_counts.size else 0
    n_m = int(col_deg.size)
    if n_r == 0 or n_m == 0:
        return None
    valid_r = (row_deg_vals > 0) & (row_deg_vals < n_m)
    valid_c = (col_deg > 0) & (col_deg < n_r)
    if not np.any(valid_r) or not np.any(valid_c):
        return None
    n_full_r = int(row_deg_counts[row_deg_vals == n_m].sum())
    n_full_c = int(np.sum(col_deg == n_r))
    labels = np.asarray(labels)[valid_c]
    col_deg = col_deg[valid_c] - n_full_r
    new_deg = row_deg_vals[valid_r] - n_full_c
    new_cnt = row_deg_counts[valid_r]
    W = np.asarray(W, dtype=np.int64)[np.ix_(valid_c, valid_c)]
    if n_full_r:
        W = W - n_full_r
        np.fill_diagonal(W, 0)
        np.maximum(W, 0, out=W)
    n_rows = int(new_cnt.sum())
    n_cols = int(col_deg.size)
    if n_rows == 0 or n_cols == 0:
        return None
    n_edges = int(np.dot(new_deg, new_cnt))
    if n_edges <= 0:
        return None
    order = np.argsort(new_deg, kind="mergesort")
    deg_g = new_deg[order]
    n_c = new_cnt[order].astype(np.float64)
    pairs = _pairs_from_W(W, min_overlap)
    if pairs is None:
        return None
    pairs_i, pairs_j, weight = pairs
    return BipartiteAgg(
        n_c=n_c,
        deg_g=deg_g,
        col_deg=col_deg.astype(np.int64),
        pairs_i=pairs_i,
        pairs_j=pairs_j,
        weight=weight,
        col_labels=labels,
        n_rows=n_rows,
        n_cols=n_cols,
        n_edges=n_edges,
    )


def _chunk_local_stats(row_ids, col_ids):
    row_arr = np.asarray(row_ids)
    col_arr = np.asarray(col_ids)
    if row_arr.shape != col_arr.shape or row_arr.ndim != 1:
        raise ValueError("row_ids and col_ids must be 1D arrays of the same length")
    row_codes, _ = pd.factorize(row_arr, sort=False)
    col_codes, col_uniques = pd.factorize(col_arr, sort=False)
    if (row_codes < 0).any() or (col_codes < 0).any():
        raise ValueError("row_ids and col_ids must not contain missing values")
    row_codes = row_codes.astype(np.int32, copy=False)
    col_codes = col_codes.astype(np.int32, copy=False)
    n_r = int(row_codes.max()) + 1 if row_codes.size else 0
    n_m = int(len(col_uniques))
    if n_r == 0 or n_m == 0:
        return None
    M = sp.csr_matrix(
        (np.ones(len(row_codes), dtype=np.uint8), (row_codes, col_codes)),
        shape=(n_r, n_m),
    )
    if M.nnz == 0:
        return None
    M.data[:] = 1
    row_deg = np.asarray(M.sum(axis=1)).ravel().astype(np.int32)
    col_deg = np.asarray(M.sum(axis=0)).ravel().astype(np.int64)
    W = (M.astype(np.int64).T @ M.astype(np.int64)).tocoo()
    return (
        np.asarray(col_uniques),
        W.row.astype(np.int32),
        W.col.astype(np.int32),
        W.data.astype(np.int64),
        row_deg,
        col_deg,
    )


def aggregate_chunks(
    chunks,
    min_overlap: int = 1,
) -> BipartiteAgg | None:
    """Stream (row_ids, col_ids) chunks. Row IDs must not span chunks.

    Builds an int64 co-occurrence in the global column vocabulary. Peak RAM
    after streaming is O(n_cols^2) plus a row-degree histogram, not N ids.
    """
    col_to_i: dict = {}
    labels: list = []
    cap = 64
    W = np.zeros((cap, cap), dtype=np.int64)
    col_deg = np.zeros(cap, dtype=np.int64)

    def _grow(n_need: int) -> None:
        nonlocal cap, W, col_deg
        while cap < n_need:
            cap *= 2
        if W.shape[0] < cap:
            W = np.pad(W, ((0, cap - W.shape[0]), (0, cap - W.shape[1])))
            col_deg = np.pad(col_deg, (0, cap - col_deg.size))

    def _index(c) -> int:
        if isinstance(c, np.generic):
            c = c.item()
        i = col_to_i.get(c)
        if i is None:
            i = len(labels)
            col_to_i[c] = i
            labels.append(c)
            if i >= cap:
                _grow(i + 1)
        return i

    any_chunk = False
    row_hist: Counter[int] = Counter()
    for row, col in chunks:
        stats = _chunk_local_stats(row, col)
        if stats is None:
            continue
        any_chunk = True
        cu, wr, wc, wd, rdeg, cdeg = stats
        gidx = np.fromiter((_index(c) for c in cu), dtype=np.int32, count=len(cu))
        _grow(len(labels))
        np.add.at(W, (gidx[wr], gidx[wc]), wd)
        col_deg[gidx] += cdeg
        u, c = np.unique(rdeg, return_counts=True)
        row_hist.update({int(d): int(n) for d, n in zip(u, c)})

    if not any_chunk or not labels or not row_hist:
        return None
    n_m = len(labels)
    W = W[:n_m, :n_m]
    col_deg = col_deg[:n_m]
    deg_vals = np.fromiter(row_hist.keys(), dtype=np.int64, count=len(row_hist))
    deg_counts = np.fromiter(
        (row_hist[int(d)] for d in deg_vals), dtype=np.int64, count=len(deg_vals)
    )
    return _agg_from_hist_and_W(
        deg_vals, deg_counts, col_deg, np.asarray(labels), W, min_overlap
    )


def fit_bicm_reduced(
    deg_g: np.ndarray,
    n_c: np.ndarray,
    col_deg: np.ndarray,
    tol: float = SOLVER_TOL,
) -> tuple[np.ndarray, np.ndarray]:
    deg_g = np.asarray(deg_g, dtype=np.float64)
    n_c = np.asarray(n_c, dtype=np.float64)
    col_deg = np.asarray(col_deg, dtype=np.float64)
    r_cols, invert_cols, cols_mult = np.unique(
        col_deg, return_inverse=True, return_counts=True
    )
    r_cols = r_cols.astype(np.float64)
    cols_mult = cols_mult.astype(np.float64)

    n_edges = float(np.dot(deg_g, n_c))
    if n_edges <= 0.0:
        raise ValueError("BiCM reduced fit: zero edges")
    scale = np.sqrt(n_edges)
    r_x = np.clip(deg_g / scale, 1e-12, 1e12)
    r_y = np.clip(r_cols / scale, 1e-12, 1e12)
    x0 = np.concatenate((-np.log(r_x), -np.log(r_y)))

    args = (deg_g, r_cols, n_c, cols_mult)
    lins_args = (bmof.loglikelihood_bicm, args)

    def fun(z):
        return -bmof.loglikelihood_prime_bicm(z, args)

    def fun_jac(z):
        return -bmof.loglikelihood_hessian_bicm(z, args)

    def step_fun(z):
        return -bmof.loglikelihood_bicm(z, args)

    def fun_linsearch(xx):
        return bmof.linsearch_fun_BiCM(xx, lins_args)

    sol = bsof.solver(
        x0,
        fun=fun,
        fun_jac=fun_jac,
        step_fun=step_fun,
        linsearch_fun=fun_linsearch,
        hessian_regulariser=bsof.matrix_regulariser_function,
        tol=tol,
        eps=SOLVER_EPS,
        max_steps=100,
        method="newton",
        verbose=False,
        regularise=True,
        linsearch=True,
    )
    n_g = len(deg_g)
    x_u = np.exp(-np.asarray(sol[:n_g], dtype=np.float64))
    y_u = np.exp(-np.asarray(sol[n_g:], dtype=np.float64))
    return x_u, y_u[invert_cols]


# Switch to the Loader anchor before (1-p)^n loses digits, not only at underflow.
_LOG_LOADER = -30.0
_LOG_UNDER = -745.0
_LN_SQRT_2PI = 0.9189385332046727  # 0.5 * log(2*pi)


@njit(nogil=True)
def _bd0(x, np_):
    """Loader deviance x*log(x/np_) + np_ - x, stable when x ≈ np_."""
    if x == 0.0:
        return np_
    if np_ == 0.0 or x == np_:
        return 0.0
    ax = x + np_
    if abs(x - np_) < 0.1 * ax:
        d = x - np_
        v = d / ax
        s = d * v
        if s == 0.0:
            return 0.0
        ej = 2.0 * x * v
        vv = v * v
        j = 1
        while j < 1000:
            ej *= vv
            s1 = s + ej / ((j << 1) + 1)
            if s1 == s:
                return s1
            s = s1
            j += 1
        return s
    ratio = x / np_
    if np.isfinite(ratio):
        lg = np.log(ratio)
    else:
        lg = np.log(x) - np.log(np_)
    if x > np_:
        return x * (lg - 1.0) + np_
    return x * lg + np_ - x


@njit(nogil=True)
def _stirlerr(x):
    """log(x!) - log(sqrt(2*pi*x)*(x/e)^x). x > 0; callers skip x == 0."""
    if x <= 15.0:
        return math.lgamma(x + 1.0) - (x + 0.5) * np.log(x) + x - _LN_SQRT_2PI
    nn = x * x
    s0 = 0.083333333333333333333  # 1/12
    s1 = 0.00277777777777777777778  # 1/360
    s2 = 0.00079365079365079365079365  # 1/1260
    s3 = 0.000595238095238095238095238  # 1/1680
    s4 = 0.0008417508417508417508417508  # 1/1188
    if x > 500.0:
        return (s0 - s1 / nn) / x
    if x > 80.0:
        return (s0 - (s1 - s2 / nn) / nn) / x
    if x > 35.0:
        return (s0 - (s1 - (s2 - s3 / nn) / nn) / nn) / x
    return (s0 - (s1 - (s2 - (s3 - s4 / nn) / nn) / nn) / nn) / x


@njit(nogil=True)
def _recur_up(n, ni, p, q, k, t_start, out):
    ratio = p / q
    t = t_start
    while t < k - 1:
        if t >= ni:
            break
        nxt = out[t] * (n - t) / (t + 1.0) * ratio
        out[t + 1] = nxt
        if nxt == 0.0:
            break
        t += 1


@njit(nogil=True)
def _recur_down(n, p, q, t_start, out):
    inv = q / p
    t = t_start
    while t > 0:
        nxt = out[t] * t / (n - t + 1.0) * inv
        out[t - 1] = nxt
        if nxt == 0.0:
            break
        t -= 1


@njit(nogil=True)
def _loader_logpmf(n, t, p, q):
    nt = n - t
    return (
        _stirlerr(n)
        - _stirlerr(t)
        - _stirlerr(nt)
        - _bd0(t, n * p)
        - _bd0(nt, n * q)
        + 0.5 * np.log(n / (2.0 * np.pi * t * nt))
    )


@njit(nogil=True)
def _binom_pmf_trunc(n, p, k, out):
    for i in range(k):
        out[i] = 0.0
    ni = int(n)
    if k <= 0:
        return
    if ni <= 0:
        out[0] = 1.0
        return
    if p <= 0.0:
        out[0] = 1.0
        return
    if p >= 1.0:
        if ni < k:
            out[ni] = 1.0
        return
    q = 1.0 - p
    nf = float(ni)
    logb = nf * np.log1p(-p)
    if logb >= _LOG_LOADER:
        out[0] = np.exp(logb)
        _recur_up(nf, ni, p, q, k, 0, out)
        return
    mode = int(np.floor((nf + 1.0) * p))
    t0 = mode if mode < k - 1 else k - 1
    if t0 >= ni:
        # mode == ni and ni < k: anchor at n, fill downward only.
        log_pmf = nf * np.log(p)
        if log_pmf < _LOG_UNDER:
            return
        out[ni] = np.exp(log_pmf)
        _recur_down(nf, p, q, ni, out)
        return
    if t0 <= 0:
        if logb < _LOG_UNDER:
            return
        out[0] = np.exp(logb)
        _recur_up(nf, ni, p, q, k, 0, out)
        return
    log_pmf = _loader_logpmf(nf, float(t0), p, q)
    if log_pmf < _LOG_UNDER:
        return
    out[t0] = np.exp(log_pmf)
    _recur_down(nf, p, q, t0, out)
    _recur_up(nf, ni, p, q, k, t0, out)


@njit(nogil=True)
def _convolve_direct(dp, b, k, tmp):
    for j in range(k):
        acc = 0.0
        for t in range(j + 1):
            acc += dp[j - t] * b[t]
        tmp[j] = acc
    for j in range(k):
        dp[j] = tmp[j]


def _nfft_for_k(k: int) -> int:
    return 1 << int(np.ceil(np.log2(max(8, 2 * int(k)))))


def _convolve_fft_linear(dp: np.ndarray, b: np.ndarray, k: int) -> None:
    from scipy.fft import irfft, rfft

    nfft = _nfft_for_k(k)
    conv = irfft(rfft(dp[:k], nfft, workers=1) * rfft(b[:k], nfft, workers=1), nfft, workers=1)[:k]
    np.maximum(conv, 0.0, out=conv)
    dp[:k] = conv


@njit(nogil=True)
def _fill_binoms_class(x_g, n_g, yi, yj, weight, b):
    n_pairs = weight.shape[0]
    for i in range(n_pairs):
        k = int(weight[i])
        p_ci = (x_g * yi[i]) / (1.0 + x_g * yi[i])
        p_cj = (x_g * yj[i]) / (1.0 + x_g * yj[i])
        _binom_pmf_trunc(n_g, p_ci * p_cj, k, b[i])


@njit(nogil=True)
def _zero_from_k(dp, weight):
    n_pairs, nfft = dp.shape
    for i in range(n_pairs):
        k = int(weight[i])
        for j in range(k, nfft):
            dp[i, j] = 0.0


@njit(nogil=True)
def _left_mass(dp, weight, out):
    n_pairs = weight.shape[0]
    for i in range(n_pairs):
        k = int(weight[i])
        s = 0.0
        for j in range(k):
            s += dp[i, j]
        sf = 1.0 - s
        if sf < 0.0:
            sf = 0.0
        if sf > 1.0:
            sf = 1.0
        out[i] = sf


def _fft_chunk_bytes(nfft: int) -> int:
    return max(1, FFT_CHUNK_BYTES // max(16 * int(nfft), 1))


def _fft_index_chunks(weight: np.ndarray, idx: np.ndarray) -> list[np.ndarray]:
    if idx.size == 0:
        return []
    ks = weight[idx]
    order = np.argsort(ks, kind="mergesort")
    idx = idx[order]
    ks = ks[order]
    nffts = np.empty(len(ks), dtype=np.int64)
    for t in range(len(ks)):
        nffts[t] = _nfft_for_k(int(ks[t]))
    chunks: list[np.ndarray] = []
    start = 0
    n = len(idx)
    while start < n:
        nf = int(nffts[start])
        end_run = start + 1
        while end_run < n and int(nffts[end_run]) == nf:
            end_run += 1
        b_max = min(FFT_CHUNK_CAP, max(1, _fft_chunk_bytes(nf)))
        pos = start
        while pos < end_run:
            nxt = min(pos + b_max, end_run)
            chunks.append(idx[pos:nxt])
            pos = nxt
        start = end_run
    return chunks


def _exact_fft_chunk(x_u, n_c, y, pairs_i, pairs_j, weight) -> np.ndarray:
    """Linear truncated conv on a chunk of pairs that share nfft (k may differ)."""
    from scipy.fft import irfft, rfft

    n_pairs = len(weight)
    if n_pairs == 0:
        return np.empty(0, dtype=np.float64)
    k_max = int(weight.max())
    nfft = _nfft_for_k(k_max)
    dp = np.zeros((n_pairs, nfft), dtype=np.float64)
    b = np.zeros((n_pairs, nfft), dtype=np.float64)
    dp[:, 0] = 1.0
    yi = y[pairs_i]
    yj = y[pairs_j]
    for g in range(len(x_u)):
        b.fill(0.0)
        _fill_binoms_class(x_u[g], n_c[g], yi, yj, weight, b)
        spec = rfft(dp, nfft, axis=1, workers=1) * rfft(b, nfft, axis=1, workers=1)
        dp = irfft(spec, nfft, axis=1, workers=1)
        np.maximum(dp, 0.0, out=dp)
        _zero_from_k(dp, weight)
    out = np.empty(n_pairs, dtype=np.float64)
    _left_mass(dp, weight, out)
    return out


def _fft_worker_init():
    for key in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        os.environ[key] = "1"
    try:
        from threadpoolctl import threadpool_limits

        threadpool_limits(1)
    except ImportError:
        pass
    x = np.array([0.1], dtype=np.float64)
    n = np.array([2.0], dtype=np.float64)
    y = np.array([0.1, 0.2], dtype=np.float64)
    _exact_fft_chunk(x, n, y, np.array([0]), np.array([1]), np.array([4], dtype=np.int64))


def _exact_fft_chunk_pack(pack):
    x_u, n_c, y, pairs_i, pairs_j, weight = pack
    return _exact_fft_chunk(x_u, n_c, y, pairs_i, pairs_j, weight)


@njit(nogil=True)
def _p_k1(x_u, n_c, yi, yj):
    log_q = 0.0
    for g in range(len(x_u)):
        p_ci = (x_u[g] * yi) / (1.0 + x_u[g] * yi)
        p_cj = (x_u[g] * yj) / (1.0 + x_u[g] * yj)
        p = p_ci * p_cj
        if p >= 1.0:
            return 1.0
        if p > 0.0:
            log_q += n_c[g] * np.log1p(-p)
    sf = 1.0 - np.exp(log_q)
    if sf < 0.0:
        sf = 0.0
    if sf > 1.0:
        sf = 1.0
    return sf


@njit(nogil=True)
def _moments_one(x_u, n_c, yi, yj):
    m = 0.0
    v = 0.0
    m3 = 0.0
    for g in range(len(x_u)):
        p_ci = (x_u[g] * yi) / (1.0 + x_u[g] * yi)
        p_cj = (x_u[g] * yj) / (1.0 + x_u[g] * yj)
        p = p_ci * p_cj
        v1 = p * (1.0 - p)
        ng = n_c[g]
        m += ng * p
        v += ng * v1
        m3 += ng * v1 * (1.0 - 2.0 * p)
    skew = m3 / (v ** 1.5) if v > 0.0 else 0.0
    return m, v, skew


@njit(parallel=True, nogil=True)
def moments_grouped(x_u, n_c, y, pairs_i, pairs_j):
    n_pairs = len(pairs_i)
    mu = np.zeros(n_pairs, dtype=np.float64)
    var = np.zeros(n_pairs, dtype=np.float64)
    skew = np.zeros(n_pairs, dtype=np.float64)
    for idx in prange(n_pairs):
        m, v, s = _moments_one(x_u, n_c, y[pairs_i[idx]], y[pairs_j[idx]])
        mu[idx] = m
        var[idx] = v
        skew[idx] = s
    return mu, var, skew


@njit(nogil=True)
def _exact_one_direct(x_u, n_c, yi, yj, k, dp, b, tmp):
    if k <= 0:
        return 1.0
    for i in range(k):
        dp[i] = 0.0
    dp[0] = 1.0
    for g in range(len(x_u)):
        p_ci = (x_u[g] * yi) / (1.0 + x_u[g] * yi)
        p_cj = (x_u[g] * yj) / (1.0 + x_u[g] * yj)
        _binom_pmf_trunc(n_c[g], p_ci * p_cj, k, b)
        _convolve_direct(dp, b, k, tmp)
    left = 0.0
    for i in range(k):
        left += dp[i]
    sf = 1.0 - left
    if sf < 0.0:
        sf = 0.0
    if sf > 1.0:
        sf = 1.0
    return sf


@njit(parallel=True, nogil=True)
def exact_pvals_direct(x_u, n_c, y, pairs_i, pairs_j, weight, k_max):
    n_pairs = len(pairs_i)
    out = np.empty(n_pairs, dtype=np.float64)
    nt = get_num_threads()
    buf_dp = np.zeros((nt, k_max), dtype=np.float64)
    buf_b = np.zeros((nt, k_max), dtype=np.float64)
    buf_tmp = np.zeros((nt, k_max), dtype=np.float64)
    for idx in prange(n_pairs):
        k = int(weight[idx])
        if k <= 0:
            out[idx] = 1.0
        elif k == 1:
            out[idx] = _p_k1(x_u, n_c, y[pairs_i[idx]], y[pairs_j[idx]])
        else:
            t = get_thread_id()
            out[idx] = _exact_one_direct(
                x_u, n_c, y[pairs_i[idx]], y[pairs_j[idx]], k,
                buf_dp[t], buf_b[t], buf_tmp[t],
            )
    return out


def _exact_one_fft(x_u, n_c, yi, yj, k):
    dp = np.zeros(k, dtype=np.float64)
    b = np.zeros(k, dtype=np.float64)
    dp[0] = 1.0
    for g in range(len(x_u)):
        p_ci = (x_u[g] * yi) / (1.0 + x_u[g] * yi)
        p_cj = (x_u[g] * yj) / (1.0 + x_u[g] * yj)
        _binom_pmf_trunc(float(n_c[g]), float(p_ci * p_cj), k, b)
        _convolve_fft_linear(dp, b, k)
    return float(min(1.0, max(0.0, 1.0 - float(dp.sum()))))


def exact_pvals_grouped(
    x_u, n_c, y, pairs_i, pairs_j, weight, k_direct: int = K_DIRECT,
    n_jobs: int | None = None,
) -> np.ndarray:
    x_u = np.asarray(x_u, dtype=np.float64)
    n_c = np.asarray(n_c, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    pairs_i = np.asarray(pairs_i, dtype=np.int64)
    pairs_j = np.asarray(pairs_j, dtype=np.int64)
    weight = np.asarray(weight, dtype=np.int64)
    out = np.empty(len(weight), dtype=np.float64)

    def _log(msg: str) -> None:
        if os.environ.get("NEW_BICM_LOG"):
            print(msg, flush=True)

    small = weight <= k_direct
    if np.any(small):
        k_max = int(max(int(weight[small].max()), 1))
        out[small] = exact_pvals_direct(
            x_u, n_c, y, pairs_i[small], pairs_j[small], weight[small], k_max
        )
        _log(f"exact schoolbook pairs={int(np.sum(small))} k_max={k_max}")
    large = np.flatnonzero(~small)
    if large.size == 0:
        return out
    chunks = _fft_index_chunks(weight, large)
    cpu = os.cpu_count() or 1
    workers = cpu if n_jobs is None else max(1, int(n_jobs))
    huge, rest = [], []
    for ix in chunks:
        (huge if int(weight[ix].max()) > HUGE_K else rest).append(ix)
    _log(f"exact fft chunks huge={len(huge)} rest={len(rest)} workers={workers}")

    def _run(ix_list, kind: str, n_w: int):
        if not ix_list:
            return
        n_w = max(1, min(int(n_w), len(ix_list)))
        if n_w == 1 or len(ix_list) == 1:
            for ix in ix_list:
                out[ix] = _exact_fft_chunk(
                    x_u, n_c, y, pairs_i[ix], pairs_j[ix], weight[ix]
                )
            return
        packs = [
            (x_u, n_c, y, pairs_i[ix], pairs_j[ix], weight[ix]) for ix in ix_list
        ]
        if kind == "process":
            pool_cm = ProcessPoolExecutor(
                max_workers=n_w,
                mp_context=mp.get_context("spawn"),
                initializer=_fft_worker_init,
            )
        else:
            pool_cm = ThreadPoolExecutor(max_workers=n_w)
        with pool_cm as pool:
            parts = list(pool.map(_exact_fft_chunk_pack, packs))
        for ix, vals in zip(ix_list, parts):
            out[ix] = vals

    huge_pairs = int(sum(ix.size for ix in huge))
    if huge_pairs:
        _run(huge, "process", min(8, workers, max(1, len(huge))))
        _log("exact fft huge done")
    _run(rest, "thread", workers)
    _log("exact fft rest done")
    return out


def compute_candidate_stats_grouped(
    x_u, n_c, y, pairs_i, pairs_j, observed_weights, compute_exact: bool,
    k_direct: int = K_DIRECT,
    n_jobs: int | None = None,
):
    mu, var, skew = moments_grouped(x_u, n_c, y, pairs_i, pairs_j)
    if compute_exact:
        exact_pvals = exact_pvals_grouped(
            x_u, n_c, y, pairs_i, pairs_j, observed_weights,
            k_direct=k_direct, n_jobs=n_jobs,
        )
    else:
        exact_pvals = np.full(len(pairs_i), np.nan, dtype=np.float64)
    return mu, var, skew, exact_pvals


@njit(nogil=True)
def exact_pvals_naive_one(x, yi, yj, k):
    if k <= 0:
        return 1.0
    dp = np.zeros(k, dtype=np.float64)
    dp[0] = 1.0
    for c in range(len(x)):
        p_ci = (x[c] * yi) / (1.0 + x[c] * yi)
        p_cj = (x[c] * yj) / (1.0 + x[c] * yj)
        p_both = p_ci * p_cj
        q = 1.0 - p_both
        for v in range(k - 1, 0, -1):
            dp[v] = dp[v] * q + dp[v - 1] * p_both
        dp[0] = dp[0] * q
    left = 0.0
    for v in range(k):
        left += dp[v]
    sf = 1.0 - left
    if sf < 0.0:
        sf = 0.0
    if sf > 1.0:
        sf = 1.0
    return sf


def expand_x_from_classes(deg_g, n_c, x_u):
    chunks = [
        np.full(int(n), x, dtype=np.float64)
        for n, x in zip(n_c, x_u)
    ]
    return np.concatenate(chunks) if chunks else np.zeros(0, dtype=np.float64)
