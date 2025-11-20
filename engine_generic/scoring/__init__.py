"""
Módulo de scoring para asignación de clusters y cálculo de propensiones.
"""

from scoring.cluster_assignment import assign_clusters, load_segmentation_pipeline
from scoring.propensity_assignment import assign_propensities, load_unified_model

__all__ = [
    "assign_clusters",
    "load_segmentation_pipeline",
    "assign_propensities",
    "load_unified_model",
]

