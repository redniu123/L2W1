# -*- coding: utf-8 -*-
"""Data engineering pipeline for L2W1.

This package contains:
- alignment/: DTW-based OCR-GT alignment (L2W1-DE-001)
- preprocess.py: Full data pipeline with cropping (L2W1-DE-002)
"""

from data.alignment.dtw_core import AlignedPair, AlignmentType, DTWAligner
from data.preprocess import DataProcessor, SampleType

__all__ = [
    "DTWAligner",
    "AlignedPair",
    "AlignmentType",
    "DataProcessor",
    "SampleType",
]
