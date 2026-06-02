from open_r1.models.vib_adapter import VIBAdapter, VIBConfig, update_dual_beta
from open_r1.models.qwen_vib_patch import attach_vib, patch_qwen2_5_vl_with_vib

__all__ = [
    "VIBAdapter",
    "VIBConfig",
    "update_dual_beta",
    "attach_vib",
    "patch_qwen2_5_vl_with_vib",
]
