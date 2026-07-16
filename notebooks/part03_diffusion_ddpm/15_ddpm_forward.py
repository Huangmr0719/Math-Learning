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
    from src.math_checks import diffusion_coefficients, q_sample_from_x0
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
        diffusion_coefficients,
        exercise_block,
        intuition_and_rigor,
        mo,
        np,
        plt,
        q_sample_from_x0,
    )


@app.cell
def _(CHAPTERS, chapter_header):
    chapter_header(CHAPTERS[15], duration="90–120 分钟")
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 1. 本章为什么存在

    第 14 章的 forward chain 必须从 \(x_0\) 依次计算
    \(x_1,x_2,\ldots,x_t\)。如果训练网络时随机抽到 \(t=800\)，难道每个 batch
    都要先执行 800 次加噪吗？

    DDPM 最关键的便利之一是：由于每一步都是线性 Gaussian，
    可以把许多小步骤合并成一步：

    \[
    x_t=\sqrt{\bar\alpha_t}x_0+
    \sqrt{1-\bar\alpha_t}\epsilon.
    \]

    ## 2. 你已经知道什么

    - 第 14 章：\(\alpha_t=1-\beta_t\) 表示一步保留比例。
    - 独立随机变量相加时，均值相加；方差按系数平方后相加。
    - 第 6 章：Gaussian 经过平移和缩放后仍是 Gaussian。

    **回忆问题：** 为什么旧信号乘 \(\sqrt{\alpha_t}\)，它的方差贡献却是 \(\alpha_t\)？
    """)
    return


@app.cell
def _(intuition_and_rigor):
    intuition_and_rigor(
        r"""
        每一步都把旧图像“保留一点、再加一点噪声”。连续保留的总比例要相乘，
        就像连续打八折：不是 \(1-0.2-0.2\)，而是 \(0.8\times0.8\)。
        \(\bar\alpha_t\) 就是从第 1 步到第 t 步的累计保留比例。
        """,
        r"""
        定义

        \[
        \alpha_t=1-\beta_t,\qquad
        \bar\alpha_t=\prod_{s=1}^t\alpha_s.
        \]

        则线性 Gaussian 递推给出

        \[
        q(x_t\mid x_0)
        =\mathcal N\!\left(
        \sqrt{\bar\alpha_t}x_0,\,
        (1-\bar\alpha_t)I
        \right).
        \]

        该结论要求各步噪声相互独立，并使用指定的 Gaussian transition。
        """,
    )
    return


@app.cell
def _(derivation_map, mo):
    mo.vstack(
        [
            mo.md("## 3–5. 两步展开，再推广到任意 t"),
            derivation_map(
                [
                    "写出 x1",
                    "把 x1 代入 x2",
                    "合并 x0 的系数",
                    "合并独立 Gaussian 噪声方差",
                    "识别累计乘积 alpha_bar_t",
                    "得到一步闭式采样",
                ]
            ),
        ],
        gap=0.8,
    )
    return


@app.cell
def _(mo):
    mo.md(r"""
    从两步开始：

    \[
    x_1=\sqrt{\alpha_1}x_0+\sqrt{1-\alpha_1}\epsilon_1,
    \]

    \[
    \begin{aligned}
    x_2
    &=\sqrt{\alpha_2}x_1+\sqrt{1-\alpha_2}\epsilon_2\\
    &=\sqrt{\alpha_2\alpha_1}x_0
      +\sqrt{\alpha_2(1-\alpha_1)}\epsilon_1
      +\sqrt{1-\alpha_2}\epsilon_2.
    \end{aligned}
    \]

    最后两项是独立 Gaussian 的线性组合。它们的总方差是

    \[
    \alpha_2(1-\alpha_1)+(1-\alpha_2)
    =1-\alpha_1\alpha_2.
    \]

    因此可以用一份新的标准 Gaussian \(\epsilon\) 表示整个噪声和：

    \[
    x_2=\sqrt{\alpha_1\alpha_2}x_0+
    \sqrt{1-\alpha_1\alpha_2}\epsilon.
    \]

    递推到任意 \(t\) 就得到 \(\bar\alpha_t=\prod_{s=1}^t\alpha_s\)。

    ### 维度与定义域

    - `betas`, `alphas`, `alpha_bars`：shape `[T]`；
    - \(t\) 是离散整数索引，不是浮点连续时间；
    - \(\bar\alpha_t\in(0,1)\)，并随 t 单调减小；
    - batch 中不同样本可使用不同 t，系数需 reshape 后与数据广播。
    """)
    return


@app.cell
def _(mo):
    forward_t = mo.ui.slider(
        1, 100, value=35, step=1, show_value=True, label="扩散时间 t"
    )
    schedule_strength = mo.ui.slider(
        0.005,
        0.05,
        value=0.02,
        step=0.005,
        show_value=True,
        label="beta_end",
    )
    return forward_t, schedule_strength


@app.cell
def _(
    COLORS,
    diffusion_coefficients,
    forward_t,
    mo,
    np,
    plt,
    q_sample_from_x0,
    schedule_strength,
):
    _betas = np.linspace(1e-4, schedule_strength.value, 100)
    _alphas, _alpha_bars = diffusion_coefficients(_betas)
    _index = forward_t.value - 1
    _alpha_bar = float(_alpha_bars[_index])

    _rng = np.random.default_rng(7)
    _labels = _rng.integers(0, 8, size=900)
    _angles = _labels * 2 * np.pi / 8
    _x0 = np.column_stack([2 * np.cos(_angles), 2 * np.sin(_angles)])
    _x0 += 0.08 * _rng.standard_normal(_x0.shape)
    _noise = _rng.standard_normal(_x0.shape)
    _xt = q_sample_from_x0(_x0, _alpha_bar, _noise)

    _fig, _axes = plt.subplots(1, 3, figsize=(11, 3.5))
    _axes[0].plot(np.arange(1, 101), _alphas, label="alpha_t", color=COLORS["data"])
    _axes[0].plot(
        np.arange(1, 101), _alpha_bars, label="alpha_bar_t", color=COLORS["model"]
    )
    _axes[0].axvline(forward_t.value, color=COLORS["muted"], linestyle="--")
    _axes[0].set_ylim(0, 1.02)
    _axes[0].legend(fontsize=8)
    _axes[0].set_title("单步保留 vs 累计保留")

    _axes[1].scatter(_x0[:, 0], _x0[:, 1], s=5, alpha=0.4, color=COLORS["data"])
    _axes[1].set_title(r"$x_0$")
    _axes[2].scatter(_xt[:, 0], _xt[:, 1], s=5, alpha=0.4, color=COLORS["model"])
    _axes[2].set_title(rf"$x_t$，alpha_bar={_alpha_bar:.3f}")
    for _ax in _axes[1:]:
        _ax.set_xlim(-3.5, 3.5)
        _ax.set_ylim(-3.5, 3.5)
        _ax.set_aspect("equal")
    _fig.tight_layout()

    mo.vstack(
        [
            mo.md(
                fr"""
                ## 7. 交互实验：不要混淆 alpha 与 alpha_bar

                当前：

                \[
                \alpha_t={_alphas[_index]:.5f},\qquad
                \bar\alpha_t={_alpha_bar:.5f}.
                \]

                \(\alpha_t\) 只描述第 t 步；\(\bar\alpha_t\) 描述从 0 到 t 的全部累计效果。
                """
            ),
            mo.hstack([forward_t, schedule_strength], widths="equal"),
            _fig,
        ]
    )
    return


@app.cell
def _(diffusion_coefficients, mo, np, q_sample_from_x0):
    _betas = np.linspace(1e-4, 0.02, 100)
    _, _alpha_bars = diffusion_coefficients(_betas)
    _alpha_bar = float(_alpha_bars[59])
    _x0_scalar = 1.5
    _rng = np.random.default_rng(11)
    _noise = _rng.standard_normal(100_000)
    _samples = q_sample_from_x0(
        np.full_like(_noise, _x0_scalar), _alpha_bar, _noise
    )
    _theory_mean = np.sqrt(_alpha_bar) * _x0_scalar
    _theory_variance = 1.0 - _alpha_bar

    mo.md(
        fr"""
        ## 6、9. 采样矩与理论矩验证

        固定 \(x_0={_x0_scalar}\)、\(t=60\)，采样 100,000 次：

        | 量 | 理论值 | 采样值 |
        |---|---:|---:|
        | mean | {_theory_mean:.5f} | {_samples.mean():.5f} |
        | variance | {_theory_variance:.5f} | {_samples.var():.5f} |

        采样统计接近理论值，支持闭式采样实现正确；公式本身由线性 Gaussian
        的均值、方差递推证明。

        ## 8. 公式到 batch 代码

        ```python
        # alpha_bars shape: [T]；t shape: [batch]。
        alpha_bar_t = alpha_bars.gather(0, t)

        # reshape 成 [batch, 1, 1, 1]，与图像 [batch, C, H, W] 广播。
        alpha_bar_t = alpha_bar_t.reshape(batch_size, 1, 1, 1)

        # epsilon 与 x0 shape 相同。
        epsilon = torch.randn_like(x0)

        # 一步得到任意时刻，不需要循环执行前 t 步。
        x_t = (
            torch.sqrt(alpha_bar_t) * x0
            + torch.sqrt(1.0 - alpha_bar_t) * epsilon
        )
        ```
        """
    )
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 10. 错误与反例

    1. **用 \(\alpha_t\) 代替 \(\bar\alpha_t\)。**
       后期的单步 \(\alpha_t\) 仍接近 1，但累计信号可能已经很小；错误代码会让 \(x_t\) 过于清晰。

    2. **把 \(\bar\alpha_t\) 写成求和。**
       连续缩放需要相乘；相加可能超过 1，使平方根失去定义。

    3. **忘记 reshape 时间系数。**
       `[batch]` 不能按预期广播到 `[batch,C,H,W]`，或会错误对齐到最后一维。

    ## 原始资料

    - [Ho et al., DDPM，公式 (2)–(4)](https://arxiv.org/abs/2006.11239)
    """)
    return


@app.cell
def _(exercise_block, mo):
    mo.vstack(
        [
            mo.md("## 11. 分层练习"),
            exercise_block(
                (
                    "alpha_t 与 alpha_bar_t 分别回答什么问题？",
                    "alpha_t 是单独第 t 步保留多少；alpha_bar_t 是从第 1 步到第 t 步累计保留多少。",
                ),
                (
                    r"若 \(\alpha_1=0.9,\alpha_2=0.8\)，求 \(\bar\alpha_2\) 与噪声方差。",
                    r"\(\bar\alpha_2=0.72\)，所以 \(q(x_2|x_0)\) 的噪声方差是 \(1-0.72=0.28\)。",
                ),
                (
                    "为什么 `gather` 后还需要 reshape？",
                    "gather 只得到每个样本的标量系数 `[B]`；reshape 明确让它沿 channel、height、width 广播。",
                ),
                (
                    "增大 beta_end 后，固定 t 的点云如何变化？先预测再拖动。",
                    "累计 alpha_bar 会下降得更快，因此同一个 t 保留更少数据结构、含有更多噪声。",
                ),
            ),
        ],
        gap=0.8,
    )
    return


@app.cell
def _(CHAPTERS, chapter_footer):
    chapter_footer(
        CHAPTERS[15],
        [
            "alpha_t 是单步保留比例，alpha_bar_t 是累计乘积。",
            "线性独立 Gaussian 允许把 t 次加噪合成一次闭式采样。",
            "q(x_t|x0) 的均值为 sqrt(alpha_bar_t)x0，方差为 1-alpha_bar_t。",
            "知道任意 xt 如何生成后，下一步要推导给定 xt 与 x0 的前一状态。",
        ],
    )
    return


if __name__ == "__main__":
    app.run()
