import marimo

__generated_with = "0.23.13"
app = marimo.App(width="medium")


@app.cell
def _():
    import sys as _sys
    from pathlib import Path as _Path
    _root = _Path(__file__).resolve().parents[2]
    if str(_root) not in _sys.path:
        _sys.path.insert(0, str(_root))
    import marimo as mo
    import numpy as np
    from src.teaching import CHAPTERS, chapter_footer, chapter_header, derivation_map, exercise_block, intuition_and_rigor
    from src.visualization import COLORS, configure_matplotlib
    configure_matplotlib()
    import matplotlib.pyplot as plt
    return CHAPTERS, COLORS, chapter_footer, chapter_header, derivation_map, exercise_block, intuition_and_rigor, mo, np, plt


@app.cell
def _(CHAPTERS, chapter_header):
    chapter_header(CHAPTERS[12], duration="85–115 分钟")
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 本章为什么存在

    语言、音素和物体部件常呈现离散结构。VQ-VAE 不让 encoder 输出任意连续 latent，
    而是从 codebook 中选择最近的向量：

    \[
    k^*=\arg\min_k\|z_e(x)-e_k\|_2,\qquad z_q=e_{k^*}.
    \]

    ## 你已经知道什么

    - 第 1 章：欧氏距离衡量向量接近程度。
    - 第 7 章：decoder 根据 latent 重构。
    - **回忆：** 最近邻选择为什么不可普通求导？
    """)
    return


@app.cell
def _(intuition_and_rigor):
    intuition_and_rigor(
        "Codebook 像一盒有限颜色的蜡笔。Encoder 想出任意颜色，量化器把它替换成最近的蜡笔颜色，decoder 只能看到被选中的代码。",
        r"""
        `argmin` 的输出索引在边界处跳变，几乎处处对 encoder output 的导数为 0，
        因此使用 straight-through estimator：forward 采用量化值，backward
        近似把 decoder 梯度原样传给 encoder。它是梯度估计策略，不是 argmin 的真实导数。
        """,
    )
    return


@app.cell
def _(derivation_map, mo):
    mo.vstack(
        [
            mo.md("## 最近邻、codebook 与 straight-through"),
            derivation_map(["encoder 得到 z_e", "计算到所有 e_k 的距离", "argmin 选择索引", "decoder 读取 z_q", "ST 传梯度", "更新 codebook 与 commitment"]),
            mo.md(r"""
            常见目标：
            \[
            L=L_{\rm recon}
            +\|\operatorname{sg}[z_e]-e\|^2
            +\beta\|z_e-\operatorname{sg}[e]\|^2.
            \]
            `sg` 是 stop-gradient。第二项移动 codebook 去追 encoder；第三项要求 encoder
            承诺靠近所选 code，避免输出任意漂移。距离矩阵 shape `[B,K]`，argmin 沿 K 轴。

            ### 三条梯度路径分别去哪里？

            设当前选中的 code 为 \(e\)：

            - reconstruction loss 通过 straight-through 路径把 decoder 梯度近似传给
              encoder；它不会把 `argmin` 变成可微函数。
            - \(\|\operatorname{sg}[z_e]-e\|^2\) 中 \(z_e\) 被视为常数，
              梯度只更新 codebook，使 e 靠近 encoder 输出。
            - \(\beta\|z_e-\operatorname{sg}[e]\|^2\) 中 e 被视为常数，
              梯度只更新 encoder，要求它靠近已选择的 code。

            对 straight-through 写法

            \[
            z_{\rm st}=z_e+\operatorname{sg}(z_q-z_e),
            \]

            forward 数值为 \(z_q\)，因为两项相加后 \(z_e\) 抵消；
            backward 时 detach 项导数为 0，所以
            \(\partial z_{\rm st}/\partial z_e\approx I\)。
            这个梯度与真实离散量化函数的导数不同，因此是有偏 surrogate gradient。
            """),
        ],
        gap=0.8,
    )
    return


@app.cell
def _(mo):
    encoder_x = mo.ui.slider(-2.5, 2.5, value=0.4, step=0.1, show_value=True, label="encoder z_e 第1维")
    encoder_y = mo.ui.slider(-2.5, 2.5, value=-0.2, step=0.1, show_value=True, label="encoder z_e 第2维")
    return encoder_x, encoder_y


@app.cell
def _(COLORS, encoder_x, encoder_y, mo, np, plt):
    _codebook = np.array([[-2,-1],[-1,1.7],[0,0],[1.5,-1.3],[2,1.4]], dtype=float)
    _z = np.array([encoder_x.value, encoder_y.value])
    _distances = np.sum((_codebook - _z) ** 2, axis=1)
    _index = int(np.argmin(_distances))
    _quantized = _codebook[_index]
    _fig, _axes = plt.subplots(1, 2, figsize=(9, 3.8))
    _axes[0].scatter(_codebook[:,0], _codebook[:,1], s=100, color=COLORS["prior"])
    for _i, _point in enumerate(_codebook):
        _axes[0].text(_point[0]+0.05, _point[1]+0.05, f"e{_i}")
    _axes[0].scatter(*_z, s=100, marker="x", color=COLORS["danger"], label="z_e")
    _axes[0].plot([_z[0],_quantized[0]], [_z[1],_quantized[1]], "--", color=COLORS["model"])
    _axes[0].legend(fontsize=8)
    _axes[0].set_xlim(-3,3); _axes[0].set_ylim(-3,3)
    _axes[1].bar([f"e{i}" for i in range(len(_codebook))], _distances, color=[COLORS["success"] if i==_index else COLORS["data"] for i in range(len(_codebook))])
    _axes[1].set_title("squared distance")
    _fig.tight_layout()
    mo.vstack([mo.md(fr"""## 量化边界交互

    选择 code **e{_index}**，\(z_q={_quantized}\)，最小平方距离={_distances[_index]:.3f}。
    缓慢移动 z_e，观察越过 Voronoi 边界时索引突然跳变。"""),
    mo.hstack([encoder_x, encoder_y], widths="equal"), _fig])
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 代码映射

    ```python
    # z_e: [B,D]；codebook: [K,D]。
    # 广播得到 [B,K,D]，再沿 D 求平方距离。
    distances = ((z_e[:, None, :] - codebook[None, :, :]) ** 2).sum(dim=-1)
    indices = distances.argmin(dim=1)       # [B]
    z_q = codebook[indices]                 # [B,D]

    # forward 数值等于 z_q；backward 对 z_e 的局部梯度近似为 identity。
    z_st = z_e + (z_q - z_e).detach()
    ```

    ## 错误与反例

    - 把 straight-through 称为精确微分：它是有偏梯度估计。
    - argmin 沿 feature 轴：会为每个坐标选不同 code，破坏整向量量化。
    - 忽略 code usage：少数 code 独占会发生 codebook collapse。

    原始资料：[van den Oord et al., Neural Discrete Representation Learning](https://arxiv.org/abs/1711.00937)
    """)
    return


@app.cell
def _(exercise_block, mo):
    mo.vstack(
        [
            mo.md("## 分层练习"),
            exercise_block(
                ("为什么 VQ latent 是离散的？", "虽然每个 code 是连续向量，但可选择的索引只有有限 K 个。"),
                ("B=16、K=512、D=64，distance shape？", "`[16,512]`。"),
                ("`z_e + (z_q-z_e).detach()` forward 等于什么？", "forward 数值上等于 `z_q`；反向传播时 `detach()` 切断括号内路径，因此来自后续网络的梯度按恒等映射近似传到 `z_e`。"),
                ("移动 z_e 穿过边界，观察索引。", "索引突然跳变说明量化映射不连续，也解释了普通梯度困难。"),
            ),
        ],
        gap=0.8,
    )
    return


@app.cell
def _(CHAPTERS, chapter_footer):
    chapter_footer(CHAPTERS[12], ["VQ-VAE 用最近邻 codebook 建立离散 latent。", "argmin 不可普通求导，ST 是近似梯度。", "codebook 与 commitment loss 分工不同。", "下一章回到连续 posterior，学习可逆变量变换。"])
    return


if __name__ == "__main__":
    app.run()
