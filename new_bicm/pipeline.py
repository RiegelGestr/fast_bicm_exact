"""End-to-end exact BiCM projection."""
from __future__ import annotations

import pandas as pd

from .core import (
    BipartiteAgg,
    aggregate_bipartite,
    aggregate_chunks,
    compute_candidate_stats_grouped,
    fit_bicm_reduced,
)
from .validate import validate_edges


def _edges_from_agg(
    agg: BipartiteAgg,
    *,
    method: str,
    correction: str,
    alpha: float,
    compute_exact: bool,
    group_id,
    group_col: str,
    n_jobs: int | None,
) -> pd.DataFrame | None:
    x_u, y = fit_bicm_reduced(agg.deg_g, agg.n_c, agg.col_deg)
    mu, var, skew, exact_pvals = compute_candidate_stats_grouped(
        x_u, agg.n_c, y, agg.pairs_i, agg.pairs_j, agg.weight, compute_exact,
        n_jobs=n_jobs,
    )
    val = validate_edges(
        observed_weights=agg.weight,
        mu=mu,
        var=var,
        skew=skew,
        exact_pvals=exact_pvals,
        method=method,
        correction_type=correction,
        alpha=alpha,
    )
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
    if group_id is not None:
        out.insert(0, group_col, group_id)
    return out


def project(
    row_ids,
    col_ids,
    *,
    method: str = "exact",
    correction: str = "bonferroni",
    alpha: float = 0.001,
    min_overlap: int = 1,
    compute_exact: bool | None = None,
    group_id=None,
    group_col: str = "group_id",
    n_jobs: int | None = None,
) -> pd.DataFrame | None:
    """Fit BiCM and return Bonferroni-validated co-occurrence edges.

    row_ids / col_ids are aligned arrays (one entry per bipartite edge).
    For data that does not fit in RAM, use ``project_chunks``.
    """
    if compute_exact is None:
        compute_exact = method == "exact"
    agg = aggregate_bipartite(row_ids, col_ids, min_overlap=min_overlap)
    if agg is None:
        return None
    return _edges_from_agg(
        agg,
        method=method,
        correction=correction,
        alpha=alpha,
        compute_exact=compute_exact,
        group_id=group_id,
        group_col=group_col,
        n_jobs=n_jobs,
    )


def project_chunks(
    chunks,
    *,
    method: str = "exact",
    correction: str = "bonferroni",
    alpha: float = 0.001,
    min_overlap: int = 1,
    compute_exact: bool | None = None,
    group_id=None,
    group_col: str = "group_id",
    n_jobs: int | None = None,
) -> pd.DataFrame | None:
    """Same as ``project``, streaming ``(row_ids, col_ids)`` chunks.

    Row IDs must not span chunks. Co-occurrence is accumulated as a dense
    int64 matrix on the global column vocabulary; row degrees are kept as
    a histogram, not N identifiers.
    """
    if compute_exact is None:
        compute_exact = method == "exact"
    agg = aggregate_chunks(chunks, min_overlap=min_overlap)
    if agg is None:
        return None
    return _edges_from_agg(
        agg,
        method=method,
        correction=correction,
        alpha=alpha,
        compute_exact=compute_exact,
        group_id=group_id,
        group_col=group_col,
        n_jobs=n_jobs,
    )
