# -*- coding: utf-8 -*-
"""Core modules for L2W1 hierarchical multi-agent framework.

This package contains:
- agent_a.py: PaddleOCR wrapper with entropy calculation (The Scout)
- router.py: Entropy + PPL based routing (The Gatekeeper)
- agent_b.py: VLM inference wrapper with V-CoT (The Judge)
"""

from core.agent_a import AgentA
from core.agent_b import AgentB
from core.router import Router

__all__ = ["AgentA", "Router", "AgentB"]

