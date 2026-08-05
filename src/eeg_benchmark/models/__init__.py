"""Model definitions: CSP+LogReg baseline and a compact EEGNet-like CNN."""

from .compact_eegnet import CompactEEGNet, count_parameters
from .csp_logreg import build_csp_pipeline

__all__ = ["CompactEEGNet", "count_parameters", "build_csp_pipeline"]
