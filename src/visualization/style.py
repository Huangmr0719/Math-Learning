"""Matplotlib 的统一颜色和本地可写缓存设置。"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path


COLORS = {
    "data": "#2563eb",
    "model": "#ea580c",
    "prior": "#7c3aed",
    "success": "#15803d",
    "danger": "#b91c1c",
    "muted": "#64748b",
}


def configure_matplotlib() -> None:
    cache_dir = Path(tempfile.gettempdir()) / "math-learning-matplotlib"
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(cache_dir))

    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "figure.figsize": (8.0, 4.6),
            "figure.dpi": 110,
            "axes.grid": True,
            "grid.alpha": 0.22,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "font.sans-serif": [
                "Arial Unicode MS",
                "PingFang SC",
                "Hiragino Sans GB",
                "DejaVu Sans",
            ],
            "axes.unicode_minus": False,
        }
    )

