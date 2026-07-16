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
    chapter_header(CHAPTERS[8], duration="75–105 分钟")
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 1. 本章为什么存在

    标准 VAE 把 reconstruction 与 KL 直接相加，但这两个目标会竞争：前者希望
    latent 保存尽可能多的样本细节，后者希望 posterior 靠近统一 prior。
    Beta-VAE 用一个旋钮明确控制这场竞争：

    \[
    L_\beta=L_{\mathrm{recon}}+\beta D_{KL}(q_\phi(z|x)\|p(z)).
    \]

    ## 2. 你已经知道什么

    - 第 7 章：标准 VAE 对应 \(\beta=1\)。
    - KL 小表示 posterior 接近 prior，但不等于表示一定“有意义”。
    - **回忆：** 去掉 KL 后，为什么随机 prior 采样失去依据？
    """)
    return


@app.cell
def _(intuition_and_rigor):
    intuition_and_rigor(
        "把 latent 看成行李箱：重构项要求多装细节，KL 项要求行李箱形状统一、便于随机抽取。beta 是机场对行李规格的严格程度。",
        r"""
        \(\beta>0\) 是正权重。它改变有限容量模型的最优折中，而不是简单改变 loss
        显示尺度。Beta-VAE 也可从带约束优化理解：在限制编码信息容量的条件下改善重构，
        Lagrange multiplier 对应 KL 约束的价格。但“更大 beta 必然 disentangle”并无保证，
        结果依赖数据生成因素、架构和优化。
        """,
    )
    return


@app.cell
def _(derivation_map, mo):
    mo.md("## 3–6. 数学暂停站：加权多目标")
    derivation_map(["标准 negative ELBO", "给 KL 乘 beta", "beta 改变梯度比例", "posterior 容量改变", "观察重构—规整折中"])
    mo.md(r"""
    对参数 \(\theta,\phi\)：

    \[
    \nabla L_\beta=\nabla L_{\rm recon}+\beta\nabla L_{\rm KL}.
    \]

    因此 beta 不只改变数值，还改变更新方向。定义域检查：\(\beta\ge0\)；
    reconstruction 与 KL 均为 batch 标量，reduction 必须一致。若一个按像素求和、
    另一个按所有元素平均，beta 的含义会随图像大小改变。

    ### 从约束问题得到 beta

    假设我们真正想解决的是：

    \[
    \min_{\theta,\phi}R(\theta,\phi)
    \quad\text{subject to}\quad
    K(\phi)\le C,
    \]

    其中 \(R\) 是 reconstruction loss，\(K\) 是平均 KL，\(C\) 是允许的
    信息容量上限。引入非负 Lagrange multiplier \(\lambda\)：

    \[
    \mathcal J=R+\lambda(K-C)
    =R+\lambda K-\lambda C.
    \]

    对固定的 \(\lambda,C\)，最后一项与模型参数无关，所以优化模型参数时可省略，
    得到 \(R+\lambda K\)。这解释了 beta 为什么像“违反容量约束的价格”。

    但真实神经网络优化是非凸的，不同 beta 不一定与某个唯一容量 C 一一对应；
    这个推导提供解释框架，不是 disentanglement 保证。
    """)
    return


@app.cell
def _(mo):
    beta = mo.ui.slider(0.0, 8.0, value=1.0, step=0.1, show_value=True, label="beta")
    capacity = mo.ui.slider(0.0, 5.0, value=2.0, step=0.1, show_value=True, label="toy latent 信息量 c")
    return beta, capacity


@app.cell
def _(COLORS, beta, capacity, mo, np, plt):
    _c = np.linspace(0, 5, 400)
    _recon = (5 - _c) ** 2 / 5
    _kl = _c ** 2 / 5
    _total = _recon + beta.value * _kl
    _current_recon = (5 - capacity.value) ** 2 / 5
    _current_kl = capacity.value ** 2 / 5
    _fig, _axes = plt.subplots(1, 2, figsize=(10, 3.8))
    _axes[0].plot(_c, _recon, label="reconstruction", color=COLORS["data"])
    _axes[0].plot(_c, _kl, label="KL", color=COLORS["prior"])
    _axes[0].plot(_c, _total, label="total", color=COLORS["model"])
    _axes[0].axvline(capacity.value, linestyle="--", color=COLORS["muted"])
    _axes[0].legend(fontsize=8)
    _axes[0].set_xlabel("toy latent 信息量")
    _axes[1].bar(["recon", "beta × KL"], [_current_recon, beta.value * _current_kl], color=[COLORS["data"], COLORS["prior"]])
    _axes[1].set_title(f"total={_current_recon + beta.value * _current_kl:.3f}")
    _fig.tight_layout()
    mo.vstack([mo.md(fr"""## 7、9. 交互验证

    当前 reconstruction={_current_recon:.3f}，KL={_current_kl:.3f}，
    \(L_\beta={_current_recon + beta.value * _current_kl:.3f}\)。
    该 toy 曲线只展示权衡方向，不证明真实神经网络一定沿相同曲线变化。"""),
    mo.hstack([beta, capacity], widths="equal"), _fig])
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 8. 代码与公式

    ```python
    # recon_loss、kl_loss 都是“每个样本平均”的标量。
    # beta 是无量纲正数，直接缩放 KL 对参数梯度的贡献。
    total_loss = recon_loss + beta * kl_loss
    ```

    若把 beta 设为 1，回到标准 VAE；若 beta=0，KL 不再参与训练，
    此时仍可能保留随机采样，但已经失去把 posterior 对齐 prior 的依据；
    beta 很大时模型可能牺牲重构，甚至进入下一章的 posterior collapse。

    ## 10. 错误与反例

    - 把 beta 越大等同于表示越好：没有这种单调保证。
    - 比较不同 beta 却改变 reduction：实验不公平。
    - 只看 total loss：不同 beta 的 total 数值不可直接横向比较，应同时看 recon、KL 和生成。

    原始资料：[Higgins et al., beta-VAE](https://openreview.net/forum?id=Sy2fzU9gl)
    """)
    return


@app.cell
def _(exercise_block, mo):
    mo.md("## 11. 分层练习")
    exercise_block(
        ("beta 控制什么？", "它控制 KL 梯度相对 reconstruction 梯度的权重，从而改变信息保存与 prior 规整的折中。"),
        ("recon=30、KL=2，beta=4 时 total 是多少？", "30+4×2=38。"),
        ("如何记录一次 beta 消融？", "固定数据、seed、架构和 reduction，分别记录 total、recon、KL、重构与 prior 采样。"),
        ("拖动 beta，观察 toy total 最小点如何移动。", "beta 增大时最优容量向更小方向移动；真实模型还受容量和优化影响。"),
    )
    return


@app.cell
def _(CHAPTERS, chapter_footer):
    chapter_footer(CHAPTERS[8], ["beta 改变重构与 KL 的梯度权衡。", "加权目标可从约束优化理解。", "大 beta 不保证 disentanglement。", "KL 压力过强会自然引出 posterior collapse。"])
    return


if __name__ == "__main__":
    app.run()
