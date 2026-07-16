import marimo

__generated_with = "0.23.13"
app = marimo.App(width="medium")


@app.cell
def _():
    import sys as _sys
    from pathlib import Path as _Path
    _root = _Path(__file__).resolve().parents[2]
    if str(_root) not in _sys.path: _sys.path.insert(0, str(_root))
    import marimo as mo
    import numpy as np
    from src.teaching import CHAPTERS, chapter_footer, chapter_header, derivation_map, exercise_block, intuition_and_rigor
    from src.visualization import COLORS, configure_matplotlib
    configure_matplotlib()
    import matplotlib.pyplot as plt
    return CHAPTERS, COLORS, chapter_footer, chapter_header, derivation_map, exercise_block, intuition_and_rigor, mo, np, plt


@app.cell
def _(CHAPTERS, chapter_header):
    chapter_header(CHAPTERS[19], duration="90–120 分钟")
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 1. 本章为什么存在

    DDPM 训练网络预测噪声，但生成必须走完很多随机小步。训练目标是否强制我们只能
    使用这条 Markov reverse chain？DDIM 的答案是否定的：保持相同 noisy marginals，
    可以选择另一族采样过程，并用更少时间步生成。

    ## 2. 你已经知道什么

    - 第 15 章：\(x_t=\sqrt{\bar\alpha_t}x_0+\sqrt{1-\bar\alpha_t}\epsilon\)。
    - 第 17 章：网络预测 \(\epsilon_\theta(x_t,t)\)。
    - **回忆：** 已知 xt 和预测噪声，怎样估计 x0？
    """)
    return


@app.cell
def _(intuition_and_rigor):
    intuition_and_rigor(
        "DDPM 像每一步都重新掷骰子的下山路线；DDIM 可以固定一条更确定的轨迹跨步下山。山势由同一个网络提供，走法却可改变。",
        r"""
        先估计
        \[
        \hat x_0=\frac{x_t-\sqrt{1-\bar\alpha_t}\epsilon_\theta}{\sqrt{\bar\alpha_t}}.
        \]
        从 t 跳到较早 s 的一般更新：
        \[
        x_s=\sqrt{\bar\alpha_s}\hat x_0+
        \sqrt{1-\bar\alpha_s-\sigma_{t\to s}^2}\epsilon_\theta+
        \sigma_{t\to s}z.
        \]
        \(\eta=0\) 时 \(\sigma=0\)，给定初始噪声与网络后轨迹确定；\(\eta>0\) 增加随机性。
        """,
    )
    return


@app.cell
def _(derivation_map, mo):
    mo.vstack(
        [
            mo.md("## 3–6. 从 xt 分解到 DDIM 更新"),
            derivation_map(["网络预测 epsilon", "代数解出 x0_hat", "选择更早时刻 s", "按 alpha_bar_s 重组信号与噪声", "eta 控制新随机噪声"]),
            mo.md(r"""
            常用
            \[
            \sigma_{t\to s}=\eta
            \sqrt{\frac{1-\bar\alpha_s}{1-\bar\alpha_t}}
            \sqrt{1-\frac{\bar\alpha_t}{\bar\alpha_s}}.
            \]
            定义域要求 \(s<t\)、\(\bar\alpha_s>\bar\alpha_t\)，且根号内非负。
            跳步 schedule 必须严格递减；代码中常见 off-by-one 来自训练索引 0-based 与论文 1-based。

            ### 更新式的三块分别负责什么？

            \[
            x_s=
            \underbrace{\sqrt{\bar\alpha_s}\hat x_0}_{\text{预测的干净信号}}
            +
            \underbrace{\sqrt{1-\bar\alpha_s-\sigma^2}\epsilon_\theta}_{\text{沿当前噪声方向}}
            +
            \underbrace{\sigma z}_{\text{新注入的随机性}}.
            \]

            三个系数的平方构成目标时刻的方差预算。根号中的
            \(1-\bar\alpha_s-\sigma^2\) 必须非负。eta=0 只删除第三项，
            并不删除网络预测的 \(\epsilon_\theta\)；因此“确定性采样”仍然依赖噪声预测模型。

            DDIM 的关键不是简单把 DDPM 方差手工设为 0，而是构造具有相同训练目标的
            非 Markov forward family，再选择其 reverse process。这里给出的更新式是
            该构造的采样结果，而非仅由 marginal 相同自动推出。
            """),
        ],
        gap=0.8,
    )
    return


@app.cell
def _(mo):
    eta = mo.ui.slider(0.0, 1.0, value=0.0, step=0.05, show_value=True, label="eta")
    steps = mo.ui.slider(5, 50, value=15, step=5, show_value=True, label="采样步数")
    return eta, steps


@app.cell
def _(COLORS, eta, mo, np, plt, steps):
    _rng = np.random.default_rng(7)
    _t = np.linspace(1,0,steps.value)
    _base_x = 2*np.cos(np.pi*_t)
    _base_y = 2*np.sin(np.pi*_t)
    _noise = eta.value*0.25*_rng.standard_normal((steps.value,2))
    _path = np.column_stack([_base_x,_base_y])+_noise
    _fig, _axes = plt.subplots(1,2,figsize=(9,3.8))
    _axes[0].plot(_path[:,0],_path[:,1],"o-",color=COLORS["model"])
    _axes[0].set_aspect("equal"); _axes[0].set_title("toy reverse path")
    _axes[1].plot(range(steps.value), np.linalg.norm(np.diff(_path,axis=0,prepend=_path[:1]),axis=1), color=COLORS["data"])
    _axes[1].set_title("每步移动长度")
    _fig.tight_layout()
    mo.vstack([mo.md(fr"""## 7、9. 跳步与随机性

    当前 {steps.value} 步、eta={eta.value:.2f}。eta=0 时重复执行得到同一路径；
    eta>0 时新噪声改变轨迹。此 toy 图解释控制量，不代表真实图像质量。"""),
    mo.hstack([steps,eta],widths="equal"),_fig])
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 8. 核心代码

    ```python
    x0_hat = (
        x_t - torch.sqrt(1-alpha_bar_t) * eps_hat
    ) / torch.sqrt(alpha_bar_t)
    sigma = eta * torch.sqrt(
        (1-alpha_bar_s)/(1-alpha_bar_t)
        * (1-alpha_bar_t/alpha_bar_s)
    )
    direction = torch.sqrt(1-alpha_bar_s-sigma**2) * eps_hat
    x_s = torch.sqrt(alpha_bar_s)*x0_hat + direction
    if eta > 0:
        x_s = x_s + sigma * torch.randn_like(x_t)
    ```

    ## 10. 错误与反例

    - eta=0 等同于“没有噪声模型”：网络仍预测训练噪声，只有采样额外随机项为 0。
    - 时间索引升序：会重新加噪。
    - 步数越少必然同质量：跳得太大时网络误差会累积。

    原始资料：[Song et al., DDIM](https://arxiv.org/abs/2010.02502)
    """)
    return


@app.cell
def _(exercise_block, mo):
    mo.vstack(
        [
            mo.md("## 11. 分层练习"),
            exercise_block(
                ("eta=0 的确定性指什么？", "固定初始 xT、网络和时间表后不再注入新随机噪声，输出可复现。"),
                ("alpha_bar_t=.25、xt=1、eps=.5，求 x0_hat。", r"\((1-\sqrt{.75}\times.5)/.5\approx1.134\)。"),
                ("为什么 schedule 必须递减？", "采样要从高噪声时刻走向低噪声时刻。"),
                ("同时减少 steps、增大 eta，观察路径。", "步长变大且随机扰动增强，轨迹更粗糙；真实质量需模型实验判断。"),
            ),
        ],
        gap=0.8,
    )
    return


@app.cell
def _(CHAPTERS, chapter_footer):
    chapter_footer(CHAPTERS[19], ["训练网络不唯一决定采样路径。", "DDIM 先预测 x0 再重组较早状态。", "eta 控制额外随机性。", "确定性轨迹使 inversion 成为可能。"])
    return


if __name__ == "__main__": app.run()
