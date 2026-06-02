"""Hierarchical Localization Module (HLM).

Coarse-to-fine visual target localization on high-resolution maps using a single
VLM. See ``hlm.py`` for the orchestrator and the submodules for geometry
(``coordinate_mapper``), cropping (``cropper``), output parsing (``parsers``) and
prompts (``prompt_templates``).
"""

from navgym.hlm.hlm import (
    HierarchicalLocalizationModule,
    HLMConfig,
    HLMResult,
)

__all__ = [
    "HierarchicalLocalizationModule",
    "HLMConfig",
    "HLMResult",
]
