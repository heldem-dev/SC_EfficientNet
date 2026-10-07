"""Backward-compatible imports for the canonical SC-EfficientNet implementation.

The canonical model definitions live in :mod:`models.sc_model_unified` and
support EfficientNet-B0 and EfficientNet-B4. This module is retained only so
older commands importing ``models.sc_model`` continue to work.
"""

from .sc_model_unified import EfficientNetBaseline, SCEfficientNet

__all__ = ["EfficientNetBaseline", "SCEfficientNet"]
