# -*- coding: utf-8 -*-
"""Alignment modules for OCR-GT sequence matching.

This package implements:
- DTW-based forced alignment (dtw_core.py)
- Character-level matching logic (matcher.py)
"""

from data.alignment.dtw_core import AlignedPair, AlignmentType, DTWAligner

__all__ = ["DTWAligner", "AlignedPair", "AlignmentType"]
