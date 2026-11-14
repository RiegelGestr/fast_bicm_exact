# fast_bicm_exact (`new_bicm`)

Fast, low-memory **exact monopartite projection** of bipartite networks under the **Bipartite Configuration Model (BiCM)**.

This repository builds directly on [`mat701/BiCM`](https://github.com/mat701/BiCM/) (`bicm` on PyPI) and accelerates the projection pipeline—most importantly the **exact Poisson-binomial p-value computation** and **chunked aggregation**—for large bipartite graphs where the row layer is much larger than the number of distinct row degrees ($N \gg G$).

---

## Goal and Relationship to `mat701/BiCM`

[`mat701/BiCM`](https://github.com/mat701/BiCM/) is the reference Python package for the Bipartite Configuration Model (Saracco et al., 2015, 2017): it solves the maximum-entropy fitness equations for a bipartite network and validates co-occurrence projections (shared neighbors between column nodes) against the BiCM null hypothesis.

`new_bicm` **does not change the statistical model or approximate the exact test**. Instead, it depends on `bicm` and keeps exact mathematical equivalence while removing memory and runtime bottlenecks at scale:

| Pipeline stage | Upstream `bicm` (`BipartiteGraph`) | `new_bicm` (`fast_bicm_exact`) |
|---|---|---|
| **Graph ingestion & co-occurrence** | Builds an in-memory `BipartiteGraph` object | Bypasses `BipartiteGraph` completely; aggregates 1D `(row_ids, col_ids)` arrays (`aggregate_bipartite`) or streams chunks (`aggregate_chunks`) into a dense `int64` co-occurrence matrix $W = M^\top M$ ($O(n_{\text{cols}}^2)$) and a row-degree histogram ($G$ classes) |
| **BiCM fitness solver** | Solves reduced or full system via `BipartiteGraph.solve_tool` | **Reuses `bicm` directly**: `fit_bicm_reduced` calls `bicm.models_functions` and `bicm.solver_functions` (Newton + Hessian regularizer) on unique row/column degree classes |
| **Exact p-values $P(K_{ij} \ge k)$** | Per-row dynamic programming over all $N$ rows ($O(N \cdot k)$ per pair) | **Grouped degree-class convolution** (`exact_pvals_grouped`): reduces $N$ Bernoullis to $G$ Binomials $\mathrm{Bin}(n_g, p_g)$, using Numba parallel schoolbook DP for $k \le 512$ and batched linear real FFT for $k > 512$ |
| **Moments & validation** | Pairwise expectations/p-values + validation | Numba-parallel `moments_grouped` ($\mu, \sigma^2$, skewness) + `validate_edges` via `statsmodels` `multipletests` (default: Bonferroni) |

### Key Design Details

1. **Grouped Poisson-Binomial Convolution ($N \to G$)**  
   Under the BiCM, all $n_g$ rows sharing the same degree $d_g$ have the same Lagrange multiplier $x_g$, and therefore the same probability $p_g = p_{gi} p_{gj}$ of connecting to both columns $i$ and $j$. Instead of convolving $N$ individual Bernoulli trials per column pair, `exact_pvals_grouped` convolves only $G$ truncated Binomial distributions $\mathrm{Bin}(n_g, p_g)$:
   - **$k = 1$**: closed-form log-space product $1 - \exp\!\left(\sum_g n_g \log(1 - p_g)\right)$.
   - **$2 \le k \le 512$** (`K_DIRECT = 512`): Numba `@njit(parallel=True)` schoolbook truncated convolution (`exact_pvals_direct`).
   - **$k > 512$**: exact **linear** real FFT convolution (`scipy.fft.rfft` / `irfft`) with `nfft = next_pow2(max(8, 2*k)) >= 2k` (zero-padded to avoid circular wrap-around, without renormalization), batched by shared `nfft` in memory-bounded chunks (`ThreadPoolExecutor`, switching to `ProcessPoolExecutor` with `spawn` for huge overlaps $k > 100{,}000$).
   - **Exact, not an approximation**: when `method="exact"`, p-values match per-row DP (no Poisson, saddlepoint, normal, or RNA substitution). Optional `"rna"` (skewness-corrected Edgeworth) and `"standard"` (Gaussian z-score) modes are also supported.

2. **Numerical Stability at Large $n_g$**  
   Evaluating $(1 - p_g)^{n_g}$ directly underflows or loses precision when $n_g p_g$ is large. `_binom_pmf_trunc` switches to Loader's saddlepoint/Stirling/deviance anchor (`_loader_logpmf`) whenever $n_g \log(1 - p_g) < -30$ and recurses upward/downward from the mode.

3. **Safe `int64` Co-occurrence Counts**  
   The sparse incidence matrix $M$ uses `uint8`, then promotes to `int64` before computing $W = M^\top M$. Doing `M.T @ M` in `uint8` silently wraps modulo 256 whenever co-occurrence counts exceed 255.

4. **Low-RAM Chunked Streaming**  
   `project_chunks` / `aggregate_chunks` stream `(row_ids, col_ids)` iterators (e.g., partitioned Parquet files) without calling `pd.concat`. Peak memory after streaming is $O(n_{\text{cols}}^2)$ for $W$ plus a compact row-degree histogram of size $G$, rather than $O(N)$ row identifier structures.

---

## Installation

Requires Python $\ge 3.10$.

```bash
pip install -e .
```

To include `pytest` for running the test suite:

```bash
pip install -e ".[test]"
```

---

## Usage

### In-Memory Arrays (`project`)

```python
from new_bicm import project

# row_ids and col_ids are aligned 1D arrays (one entry per bipartite edge)
edges = project(
    row_ids,
    col_ids,
    method="exact",           # "exact", "rna", or "standard"
    correction="bonferroni",  # any method accepted by statsmodels.stats.multitest.multipletests
    alpha=0.001,
    min_overlap=1,
    n_jobs=8,                 # FFT thread pool (or process pool for k > 100,000)
)
```

`edges` is a `pandas.DataFrame` (or `None` if pruning leaves no valid rows/columns/pairs) with columns:
`col_i`, `col_j`, `weight`, `z_score`, `pval_raw`, `pval_corrected` (plus an optional `group_col` if `group_id` is provided).

### Streaming Chunks (`project_chunks`)

> **Important:** Row IDs **must not span chunks** (all edges belonging to a given `row_id` must appear in the same chunk). Column IDs may appear across any chunks.

```python
from pathlib import Path
import pandas as pd
from new_bicm import project_chunks

def parquet_chunks(cache: Path):
    for path in sorted(cache.glob("*.parquet")):
        df = pd.read_parquet(path, columns=["row_id", "col_id"])
        yield df["row_id"].to_numpy(), df["col_id"].to_numpy()

edges = project_chunks(
    parquet_chunks(Path("path/to/parquet_chunks")),
    method="exact",
    correction="bonferroni",
    alpha=0.001,
    n_jobs=8,
)
```

### Lower-Level Building Blocks

All pipeline stages are also exposed directly from `new_bicm`:

- `aggregate_bipartite(row_ids, col_ids, min_overlap=1) -> BipartiteAgg | None`
- `aggregate_chunks(chunks, min_overlap=1) -> BipartiteAgg | None`
- `fit_bicm_reduced(deg_g, n_c, col_deg, tol=1e-9) -> (x_u, y)`
- `exact_pvals_grouped(x_u, n_c, y, pairs_i, pairs_j, weight, k_direct=512, n_jobs=None) -> np.ndarray`
- `validate_edges(observed_weights, mu, var, skew, exact_pvals=None, method="exact", correction_type="bonferroni", alpha=0.001)`

---

## Repository Layout and Benchmarks

- `new_bicm/core.py` — bipartite aggregation (in-RAM and chunked), `fit_bicm_reduced` wrapper around `bicm`, Loader-anchored truncated Binomial PMFs, Numba schoolbook & linear-FFT exact Poisson-binomial kernels, and `moments_grouped`.
- `new_bicm/pipeline.py` — high-level `project` and `project_chunks` entry points.
- `new_bicm/validate.py` — `validate_edges` supporting `"exact"`, `"rna"`, and `"standard"` p-values with `multipletests`.
- `scripts/run_streamed_exact.py` — end-to-end chunked Parquet benchmark (generates synthetic chunks by default, or pass `--cache <dir>`).
- `scripts/time_kernel.py` — kernel-only timing of `moments_grouped` and `exact_pvals_grouped` using `data/synthetic_k_hist.npz` and `data/synthetic_fit.npz`.
- `scripts/probe_exact_timing.py` — stratified wall-clock timing probe across overlap bins (`1-512`, `513-2000`, `2001-10000`, `10001-50000`, `>50000`).
- `tests/test_new_bicm.py` — equivalence and regression tests (against `bicm.BipartiteGraph`, naive per-row DP, `scipy.stats.binom.sf`, chunked vs. in-RAM parity, and `uint8` overflow protection).

Run the synthetic benchmarks:

```bash
python scripts/run_streamed_exact.py
python scripts/time_kernel.py
python scripts/probe_exact_timing.py
```

---

## Tests

```bash
pytest -q
```
