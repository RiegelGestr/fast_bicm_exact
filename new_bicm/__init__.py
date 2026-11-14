"""Fast, low-RAM exact BiCM projection."""

from .core import (
    BipartiteAgg,
    aggregate_bipartite,
    aggregate_chunks,
    exact_pvals_grouped,
    fit_bicm_reduced,
)
from .pipeline import project, project_chunks
from .validate import validate_edges

__all__ = [
    "BipartiteAgg",
    "aggregate_bipartite",
    "aggregate_chunks",
    "exact_pvals_grouped",
    "fit_bicm_reduced",
    "project",
    "project_chunks",
    "validate_edges",
]
