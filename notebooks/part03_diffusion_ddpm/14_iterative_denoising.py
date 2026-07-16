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
    from src.teaching import (
        CHAPTERS,
        chapter_footer,
        chapter_header,
        derivation_map,
        exercise_block,
        intuition_and_rigor,
    )
    from src.visualization import COLORS, configure_matplotlib

    configure_matplotlib()
    import matplotlib.pyplot as plt

    return (
        CHAPTERS,
        COLORS,
        chapter_footer,
        chapter_header,
        derivation_map,
        exercise_block,
        intuition_and_rigor,
        mo,
        np,
        plt,
    )


@app.cell
def _(CHAPTERS, chapter_header):
    chapter_header(CHAPTERS[14], duration="75–105 分钟")
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 1. 本章为什么存在

    假设让一个完全不会画画的人，一步画出一张清晰的人脸。这一步需要同时决定轮廓、
    五官、纹理和光影，非常困难。

    但如果任务改成：

    > 眼前已经有一张稍微模糊的图，只需让它比刚才清楚一点。

    每一步就简单得多。Diffusion model 的核心策略正是把“从噪声直接生成数据”
    拆成许多次小幅去噪。

    为了获得这些训练用的“稍微被破坏的数据”，我们先定义一个已知的 forward
    noising process；生成时再学习反方向。

    ## 2. 你已经知道什么

    - 第 6 章：标准 Gaussian 噪声 \(\epsilon\sim\mathcal N(0,I)\) 容易采样。
    - 向量的每个坐标都可以同时加噪。
    - 条件概率描述“已知当前状态后，下一个状态如何分布”。

    **回忆问题：** 为什么“很多次小修正”可能比“一次猜出最终答案”容易学习？
    """)
    return


@app.cell
def _(intuition_and_rigor):
    intuition_and_rigor(
        r"""
        把一张清晰照片放进复印机，每复印一次就随机增加一点雪花。
        第 \(t\) 张只由第 \(t-1\) 张继续损坏；机器不必重新查看最初照片。
        这就是本章所需的 Markov 直觉。
        """,
        r"""
        随机变量序列 \(X_0,X_1,\ldots,X_T\) 满足 Markov property，若

        \[
        q(x_t\mid x_{0:t-1})=q(x_t\mid x_{t-1}).
        \]

        DDPM 选择 Gaussian transition kernel：

        \[
        q(x_t\mid x_{t-1})
        =\mathcal N(\sqrt{1-\beta_t}\,x_{t-1},\beta_t I),
        \qquad 0<\beta_t<1.
        \]

        条件只依赖 \(x_{t-1}\)，但这不表示 \(x_t\) 与更早状态完全无关；
        更早信息已经通过 \(x_{t-1}\) 间接传递。
        """,
    )
    return


@app.cell
def _(derivation_map, mo):
    mo.vstack(
        [
            mo.md("## 3–5. 从小幅破坏到 Markov chain"),
            derivation_map(
                [
                    "从数据 x0 出发",
                    "选择小噪声强度 beta_t",
                    "保留 sqrt(1-beta_t) 倍旧状态",
                    "加入 sqrt(beta_t) 倍新噪声",
                    "重复 T 次得到近似纯噪声",
                    "学习反向小步骤",
                ]
            ),
        ],
        gap=0.8,
    )
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### 新数学 1：Markov chain

    日常语言是“预测下一步只需要看现在”。正式语言是条件分布只需要
    \(x_{t-1}\)，而不需要把 \(x_0,\ldots,x_{t-2}\) 全部再次输入。

    ### 新数学 2：transition kernel

    `kernel` 在这里不是一块矩阵，而是一条规则：

    > 给我当前状态 \(x_{t-1}\)，我返回下一个状态 \(x_t\) 的概率分布。

    DDPM 的一步采样写成：

    \[
    x_t=\sqrt{1-\beta_t}\,x_{t-1}+\sqrt{\beta_t}\,\epsilon_t,
    \qquad \epsilon_t\sim\mathcal N(0,I).
    \]

    两个平方根不是装饰。若当前每个坐标方差为 1，且新噪声独立，则新方差为
    \((1-\beta_t)+\beta_t=1\)，不会不断爆炸。

    ### 定义域与 shape

    - \(x_t,\epsilon_t\)：与数据相同 shape，例如 `[batch, 2]` 或 `[batch, C, H, W]`；
    - \(\beta_t\)：标量，必须满足 \(0<\beta_t<1\)；
    - \(I\)：与单个样本展平维度相同的单位协方差；
    - 每一步使用新的独立噪声 \(\epsilon_t\)。
    """)
    return


@app.cell
def _(mo):
    noise_step = mo.ui.slider(
        0, 20, value=0, step=1, show_value=True, label="当前加噪步 t"
    )
    experiment_seed = mo.ui.number(
        value=7, start=0, stop=9999, step=1, label="随机种子（修改即可重新采样）"
    )
    return experiment_seed, noise_step


@app.cell
def _(COLORS, experiment_seed, mo, noise_step, np, plt):
    _rng = np.random.default_rng(int(experiment_seed.value))
    _count = 800
    _labels = _rng.integers(0, 8, size=_count)
    _angles = _labels * (2 * np.pi / 8)
    _x0 = np.column_stack([2 * np.cos(_angles), 2 * np.sin(_angles)])
    _x0 = _x0 + 0.08 * _rng.standard_normal((_count, 2))

    _beta = 0.12
    _states = [_x0]
    for _ in range(20):
        _next_noise = _rng.standard_normal((_count, 2))
        _states.append(
            np.sqrt(1.0 - _beta) * _states[-1]
            + np.sqrt(_beta) * _next_noise
        )
    _xt = _states[noise_step.value]

    _fig, _axes = plt.subplots(1, 3, figsize=(11, 3.6))
    _axes[0].scatter(_x0[:, 0], _x0[:, 1], s=5, alpha=0.45, color=COLORS["data"])
    _axes[0].set_title(r"$x_0$：八个数据团")
    _axes[1].scatter(_xt[:, 0], _xt[:, 1], s=5, alpha=0.45, color=COLORS["model"])
    _axes[1].set_title(rf"$x_t$：第 {noise_step.value} 步")
    _prior = _rng.standard_normal((_count, 2))
    _axes[2].scatter(
        _prior[:, 0], _prior[:, 1], s=5, alpha=0.45, color=COLORS["prior"]
    )
    _axes[2].set_title("参考：标准 Gaussian")
    for _ax in _axes:
        _ax.set_xlim(-3.5, 3.5)
        _ax.set_ylim(-3.5, 3.5)
        _ax.set_aspect("equal")
    _fig.tight_layout()

    mo.vstack(
        [
            mo.md(
                fr"""
                ## 7. 交互实验：逐步抹去数据结构

                当前 \(t={noise_step.value}\)。拖动 stepper，观察八个团的身份信息怎样
                逐步消失。这里使用固定 \(\beta={_beta}\) 只为清楚展示；
                正式 DDPM 通常使用随时间变化的 schedule。
                """
            ),
            mo.hstack([noise_step, experiment_seed], widths="equal"),
            _fig,
        ]
    )
    return


@app.cell
def _(mo, np):
    _current = np.array([0.4, -1.2])
    _same_new_noise = np.array([0.3, 0.7])
    _beta = 0.1

    # 两条不同历史只要抵达同一个 current state，使用相同 transition
    # 与相同新噪声时，下一状态就完全相同。
    _next_from_history_a = (
        np.sqrt(1 - _beta) * _current + np.sqrt(_beta) * _same_new_noise
    )
    _next_from_history_b = (
        np.sqrt(1 - _beta) * _current + np.sqrt(_beta) * _same_new_noise
    )
    _difference = np.max(np.abs(_next_from_history_a - _next_from_history_b))

    mo.md(
        fr"""
        ## 6、9. Markov 性的可执行检查

        给定同一个当前状态、同一个 transition rule 和同一份新噪声，两条不同历史得到：

        - history A 的下一状态：`{_next_from_history_a}`
        - history B 的下一状态：`{_next_from_history_b}`
        - 最大差异：`{_difference:.1e}`

        这段实验检查代码是否只使用当前状态；Markov property 本身来自我们对联合分布
        的建模定义，不是由这一个数值例子证明。

        ## 8. 公式与代码逐行对应

        ```python
        # beta_t 是当前一步加入的方差比例，shape 可为标量，
        # 或广播成 [batch, 1, 1, 1]。
        retained = torch.sqrt(1.0 - beta_t) * x_previous

        # randn_like 保证 epsilon_t 与 x_previous shape 完全相同。
        epsilon_t = torch.randn_like(x_previous)

        # 新噪声乘 sqrt(beta_t)，因此其方差贡献是 beta_t。
        injected = torch.sqrt(beta_t) * epsilon_t

        # 对应 q(x_t | x_{{t-1}}) 的一次采样。
        x_t = retained + injected
        ```
        """
    )
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 10. 错误与反例

    1. **把“Markov”理解成与过去毫无关系。**
       \(x_t\) 仍然包含 \(x_0\) 的残余信息，只是给定 \(x_{t-1}\) 后，不必再次条件于更早状态。

    2. **直接写 \(x_t=x_{t-1}+\sqrt{\beta_t}\epsilon_t\)。**
       这样方差会持续增加；DDPM 同时缩小旧信号，维持受控的尺度。

    3. **用同一份 \(\epsilon\) 代替每一步独立噪声。**
       这会改变定义的 transition chain，后续闭式方差推导也随之失效。

    ## 原始资料

    - [Sohl-Dickstein et al., Nonequilibrium Thermodynamics](https://arxiv.org/abs/1503.03585)
    - [Ho et al., Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2006.11239)
    """)
    return


@app.cell
def _(exercise_block, mo):
    mo.vstack(
        [
            mo.md("## 11. 分层练习"),
            exercise_block(
                (
                    "用自己的话解释 Markov property，不使用“无记忆”三个字。",
                    "预测下一状态的条件分布时，当前状态已经汇总了所需历史信息，因此规则只读取当前状态。",
                ),
                (
                    r"若 \(\beta_t=0.04\)，旧状态与新噪声分别乘什么系数？",
                    r"旧状态乘 \(\sqrt{0.96}\approx0.980\)，新噪声乘 \(\sqrt{0.04}=0.2\)。",
                ),
                (
                    "若删除 `sqrt(1-beta_t) * x_previous` 会发生什么？",
                    "每一步只剩新随机噪声，链条会立即丢失此前状态，而不是逐步破坏数据。",
                ),
                (
                    "修改随机种子并比较同一个 t 的点云。哪些性质改变，哪些总体趋势不变？",
                    "单个点位置会改变，但随 t 增大、数据结构被抹平并趋向 Gaussian 的总体趋势应保持。",
                ),
            ),
        ],
        gap=0.8,
    )
    return


@app.cell
def _(CHAPTERS, chapter_footer):
    chapter_footer(
        CHAPTERS[14],
        [
            "Diffusion 把一次困难生成拆成许多次小去噪。",
            "Forward process 是只依赖当前状态的 Markov chain。",
            "一步 Gaussian transition 同时缩小旧信号并注入新噪声。",
            "逐步采样容易理解，但训练时还需要从 x0 直接得到任意 xt。",
        ],
    )
    return


if __name__ == "__main__":
    app.run()
