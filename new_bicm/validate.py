"""Bonferroni / FDR validation of co-occurrence p-values."""
from __future__ import annotations

import numpy as np
import scipy.stats as stats
from statsmodels.stats.multitest import multipletests


def validate_edges(
    observed_weights,
    mu,
    var,
    skew,
    exact_pvals=None,
    method="exact",
    correction_type="bonferroni",
    alpha=0.001,
):
    observed_weights = np.asarray(observed_weights, dtype=np.float64)
    mu = np.asarray(mu, dtype=np.float64)
    var = np.asarray(var, dtype=np.float64)
    skew = np.asarray(skew, dtype=np.float64)

    if method == "exact" and exact_pvals is None:
        raise ValueError("exact_pvals is required when method='exact'")
    if method not in ("standard", "rna", "exact"):
        raise ValueError(f"unknown method {method!r}")

    if observed_weights.size == 0:
        empty_f = np.empty(0, dtype=np.float64)
        return {
            "reject_mask": np.empty(0, dtype=bool),
            "corrected_pvals": empty_f,
            "raw_target_pvals": empty_f,
            "z_scores": empty_f,
        }

    pos = var > 0.0
    z_scores = np.empty_like(observed_weights, dtype=np.float64)
    np.divide(observed_weights - mu, np.sqrt(var), out=z_scores, where=pos)
    if not np.all(pos):
        diff = observed_weights[~pos] - mu[~pos]
        z_scores[~pos] = np.where(
            diff > 0.0, np.inf, np.where(diff < 0.0, -np.inf, 0.0)
        )

    std_pvals = stats.norm.sf(z_scores)
    norm_pdf = np.zeros_like(z_scores, dtype=np.float64)
    finite = np.isfinite(z_scores)
    if np.any(finite):
        norm_pdf[finite] = stats.norm.pdf(z_scores[finite])
    rna_pvals = std_pvals.copy()
    if np.any(finite):
        rna_pvals[finite] = np.clip(
            std_pvals[finite]
            + (skew[finite] / 6.0) * (z_scores[finite] ** 2 - 1.0) * norm_pdf[finite],
            0.0,
            1.0,
        )

    if method == "standard":
        target_pvals = std_pvals
    elif method == "rna":
        target_pvals = rna_pvals
    else:
        target_pvals = np.asarray(exact_pvals, dtype=np.float64)

    reject, pvals_corrected, _, _ = multipletests(
        pvals=target_pvals, alpha=alpha, method=correction_type
    )
    return {
        "reject_mask": reject,
        "corrected_pvals": pvals_corrected,
        "raw_target_pvals": target_pvals,
        "z_scores": z_scores,
    }
