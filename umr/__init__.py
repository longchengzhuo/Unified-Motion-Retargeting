"""UMR: Unified Motion Retargeting with learned point cloud correspondence.

复现 arXiv:2609.02134v1《Unified Motion Retargeting for Humanoids with Learned
Point Cloud Correspondence》。运动学、优化与仿真全部构建在 MuJoCo + mink 之上。
"""

__version__ = "0.1.0"

from umr.paths import (
    ASSET_DIR,
    CONFIG_DIR,
    DATA_DIR,
    OUTPUT_DIR,
    PROJECT_ROOT,
)

__all__ = [
    "__version__",
    "PROJECT_ROOT",
    "ASSET_DIR",
    "CONFIG_DIR",
    "DATA_DIR",
    "OUTPUT_DIR",
]
