"""Matplotlib 的统一颜色和本地可写缓存设置。"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path


COLORS = {
    "data": "#2563eb",
    "model": "#ea580c",
    "prior": "#7c3aed",
    "warning": "#b45309",
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
            # Notebook 外层可以是深色或浅色主题，但教学图统一使用浅色画布。
            # 显式锁定前景和背景，避免同一进程中先前启用 dark_background 后，
            # 坐标、图例或公式文字在白底上失去对比度。
            "figure.facecolor": "#ffffff",
            "figure.edgecolor": "#ffffff",
            "savefig.facecolor": "#ffffff",
            "savefig.edgecolor": "#ffffff",
            "axes.facecolor": "#ffffff",
            "axes.edgecolor": "#475569",
            "axes.labelcolor": "#1f2933",
            "axes.titlecolor": "#1f2933",
            "text.color": "#1f2933",
            "xtick.color": "#475569",
            "ytick.color": "#475569",
            "legend.facecolor": "#ffffff",
            "legend.edgecolor": "#cbd5e1",
            "legend.labelcolor": "#1f2933",
            "axes.grid": True,
            "grid.color": "#94a3b8",
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
