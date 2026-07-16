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
    from src.math_checks import (
        ddpm_posterior_mean_variance,
        diffusion_coefficients,
    )
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
        ddpm_posterior_mean_variance,
        derivation_map,
        diffusion_coefficients,
        exercise_block,
        intuition_and_rigor,
        mo,
        np,
        plt,
    )


@app.cell
def _(CHAPTERS, chapter_header):
    chapter_header(CHAPTERS[16], duration="105–140 分钟")
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 1. 本章为什么存在

    Forward process 告诉我们怎样从清晰数据走向噪声，但生成需要反过来：

    \[
    x_T\rightarrow x_{T-1}\rightarrow\cdots\rightarrow x_0.
    \]

    假设训练时同时知道原始 \(x_0\) 和加噪后的 \(x_t\)。前一状态
    \(x_{t-1}\) 最可能在哪里？它的不确定性有多大？

    这正是条件后验：

    \[
    q(x_{t-1}\mid x_t,x_0).
    \]

    ## 2. 你已经知道什么

    - 第 3 章：Bayes rule 把 prior 与 likelihood 相乘得到 posterior。
    - 第 15 章：\(q(x_{t-1}\mid x_0)\) 有闭式 Gaussian。
    - 第 14 章：\(q(x_t\mid x_{t-1})\) 是一步 Gaussian transition。

    **回忆问题：** 为什么只知道 \(x_t\) 时，前一状态不是唯一确定的？
    """)
    return


@app.cell
def _(intuition_and_rigor):
    intuition_and_rigor(
        r"""
        想象地面上有一个模糊脚印 \(x_t\)。原始目的地 \(x_0\) 提供“应该往哪里走”的线索，
        最后一步的噪声规律提供“这一步通常能偏多远”的线索。两条线索合并后，
        得到前一位置 \(x_{t-1}\) 的概率范围。
        """,
        r"""
        由 Markov property 与 Bayes rule，

        \[
        q(x_{t-1}\mid x_t,x_0)
        \propto q(x_t\mid x_{t-1})q(x_{t-1}\mid x_0).
        \]

        右侧是关于 \(x_{t-1}\) 的两个 Gaussian 因子。Gaussian 乘 Gaussian
        后仍与 Gaussian 成比例，因此 posterior 可解析写成
        \(\mathcal N(\tilde\mu_t,\tilde\beta_tI)\)。
        """,
    )
    return


@app.cell
def _(derivation_map, mo):
    mo.vstack(
        [
            mo.md("## 3–5. Gaussian conditioning 推导地图"),
            derivation_map(
                [
                    "写 Bayes 比例式",
                    "代入两个 Gaussian 指数",
                    "展开关于 x_{t-1} 的二次项",
                    "完成平方",
                    "读出 posterior mean",
                    "读出 posterior variance",
                ]
            ),
        ],
        gap=0.8,
    )
    return


@app.cell
def _(mo):
    mo.md(r"""
    记 \(\alpha_t=1-\beta_t\)，则

    \[
    q(x_t\mid x_{t-1})
    =\mathcal N(\sqrt{\alpha_t}x_{t-1},\beta_tI),
    \]

    \[
    q(x_{t-1}\mid x_0)
    =\mathcal N(
    \sqrt{\bar\alpha_{t-1}}x_0,
    (1-\bar\alpha_{t-1})I).
    \]

    将两者相乘并完成平方，可得

    \[
    q(x_{t-1}\mid x_t,x_0)
    =\mathcal N(\tilde\mu_t,\tilde\beta_tI),
    \]

    \[
    \tilde\mu_t
    =
    \frac{\sqrt{\bar\alpha_{t-1}}\beta_t}
         {1-\bar\alpha_t}x_0
    +
    \frac{\sqrt{\alpha_t}(1-\bar\alpha_{t-1})}
         {1-\bar\alpha_t}x_t,
    \]

    \[
    \tilde\beta_t
    =
    \frac{1-\bar\alpha_{t-1}}
         {1-\bar\alpha_t}\beta_t.
    \]

    ### “完成平方”到底做了什么？

    高中二次函数中：

    \[
    ax^2+bx=a\left(x+\frac{b}{2a}\right)^2-\frac{b^2}{4a}.
    \]

    Gaussian 指数也是关于未知量的二次函数。把它改写成
    \(-\frac{1}{2\sigma^2}(x-\mu)^2+\text{常数}\)，就能直接读出
    mean \(\mu\) 和 variance \(\sigma^2\)。

    ### 真正把二次项收集一遍

    先看一维，并把未知量 \(x_{t-1}\) 暂记为 \(y\)。Bayes rule 中与 y
    有关的两项为

    \[
    q(x_t|y)q(y|x_0)
    \propto
    \exp\left[
    -\frac{(x_t-\sqrt{\alpha_t}y)^2}{2\beta_t}
    -\frac{(y-\sqrt{\bar\alpha_{t-1}}x_0)^2}
           {2(1-\bar\alpha_{t-1})}
    \right].
    \]

    展开平方，只保留含 y 的项，可写成

    \[
    -\frac12\left(Ay^2-2By\right)+\text{const},
    \]

    其中

    \[
    A=
    \frac{\alpha_t}{\beta_t}
    \frac{1}{1-\bar\alpha_{t-1}}
    =
    \frac{1-\bar\alpha_t}
         {\beta_t(1-\bar\alpha_{t-1})},
    \]

    \[
    B=
    \frac{\sqrt{\alpha_t}}{\beta_t}x_t
    \frac{\sqrt{\bar\alpha_{t-1}}}
         {1-\bar\alpha_{t-1}}x_0.
    \]

    使用
    \(Ay^2-2By=A(y-B/A)^2-B^2/A\)，所以 posterior variance 是
    \(A^{-1}\)，posterior mean 是 \(B/A\)。分别化简就得到上面的
    \(\tilde\beta_t\) 和 \(\tilde\mu_t\)。

    多维情形因为 covariance 都是“标量乘单位矩阵”，每个坐标进行同样运算，
    最终把标量 y 换回与 \(x_t\) 同 shape 的向量即可。若 covariance 不是
    对角或各向同性，就需要矩阵 precision 相加，而不能逐元素照抄。

    ### 检查

    - \(x_0,x_t,\tilde\mu_t\) shape 完全相同；
    - \(\tilde\beta_t\) 是正标量或每个时间步一个标量；
    - \(t=1\) 时 \(\bar\alpha_0=1\)，所以 \(\tilde\beta_1=0\)；
    - posterior 条件中包含 \(x_0\)，但真实生成时 \(x_0\) 尚未知。
    """)
    return


@app.cell
def _(mo):
    posterior_t = mo.ui.slider(
        2, 100, value=45, step=1, show_value=True, label="时间 t（从 2 开始）"
    )
    posterior_x0 = mo.ui.slider(
        -2.0, 2.0, value=1.0, step=0.1, show_value=True, label="x0"
    )
    posterior_xt = mo.ui.slider(
        -3.0, 3.0, value=0.2, step=0.1, show_value=True, label="观测 xt"
    )
    return posterior_t, posterior_x0, posterior_xt


@app.cell
def _(
    COLORS,
    ddpm_posterior_mean_variance,
    diffusion_coefficients,
    mo,
    np,
    plt,
    posterior_t,
    posterior_x0,
    posterior_xt,
):
    _betas = np.linspace(1e-4, 0.02, 100)
    _alphas, _alpha_bars = diffusion_coefficients(_betas)
    _index = posterior_t.value - 1
    _x0 = np.array([posterior_x0.value])
    _xt = np.array([posterior_xt.value])
    _mean, _variance = ddpm_posterior_mean_variance(
        _x0,
        _xt,
        alpha_t=float(_alphas[_index]),
        alpha_bar_t=float(_alpha_bars[_index]),
        alpha_bar_previous=float(_alpha_bars[_index - 1]),
        beta_t=float(_betas[_index]),
    )

    _grid = np.linspace(-3.5, 3.5, 1200)
    _prior_mean = np.sqrt(_alpha_bars[_index - 1]) * _x0[0]
    _prior_var = 1.0 - _alpha_bars[_index - 1]
    _prior_density = np.exp(-0.5 * (_grid - _prior_mean) ** 2 / _prior_var) / np.sqrt(
        2 * np.pi * _prior_var
    )
    _likelihood = np.exp(
        -0.5
        * (_xt[0] - np.sqrt(_alphas[_index]) * _grid) ** 2
        / _betas[_index]
    )
    _product = _prior_density * _likelihood
    _product /= np.trapz(_product, _grid)
    _analytic = np.exp(-0.5 * (_grid - _mean[0]) ** 2 / _variance) / np.sqrt(
        2 * np.pi * _variance
    )
    _max_error = float(np.max(np.abs(_product - _analytic)))

    _fig, _ax = plt.subplots(figsize=(9, 4.2))
    _ax.plot(
        _grid,
        _prior_density,
        label=r"$q(x_{t-1}\mid x_0)$",
        color=COLORS["data"],
    )
    _ax.plot(
        _grid,
        _likelihood / np.trapz(_likelihood, _grid),
        label="likelihood（归一化后）",
        color=COLORS["model"],
    )
    _ax.plot(
        _grid,
        _product,
        label="两因子乘积后归一化",
        color=COLORS["success"],
        linewidth=3,
    )
    _ax.plot(
        _grid,
        _analytic,
        "--",
        label="解析 posterior",
        color=COLORS["prior"],
    )
    _ax.axvline(_mean[0], color=COLORS["muted"], linestyle=":")
    _ax.legend(fontsize=8)
    _ax.set_title("Bayes：prior × likelihood → posterior")

    mo.vstack(
        [
            mo.md(
                fr"""
                ## 7、9. 交互与数值验证

                - posterior mean：\(\tilde\mu_t={_mean[0]:.5f}\)
                - posterior variance：\(\tilde\beta_t={_variance:.6f}\)
                - 网格乘积与解析密度最大误差：`{_max_error:.2e}`

                曲线重合支持“完成平方后的公式”和直接密度相乘一致；它仍不是一般情形的形式证明。
                """
            ),
            mo.hstack(
                [posterior_t, posterior_x0, posterior_xt],
                widths="equal",
            ),
            _fig,
        ]
    )
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 8. 公式到代码

    ```python
    # denominator 是 1 - alpha_bar_t；shape 可广播到 x0。
    denominator = 1.0 - alpha_bar_t

    # 第一项说明 posterior mean 从已知干净样本 x0 获取信息。
    coefficient_x0 = (
        torch.sqrt(alpha_bar_previous) * beta_t / denominator
    )

    # 第二项说明 posterior mean 同时必须尊重当前 noisy observation xt。
    coefficient_xt = (
        torch.sqrt(alpha_t)
        * (1.0 - alpha_bar_previous)
        / denominator
    )

    posterior_mean = coefficient_x0 * x0 + coefficient_xt * xt

    # posterior variance 只由 schedule 和时间决定，不依赖具体 x0、xt。
    posterior_variance = (
        beta_t
        * (1.0 - alpha_bar_previous)
        / denominator
    )
    ```

    ### 从真实 posterior 到可学习 reverse model

    训练时可以构造 \(x_t\)，因此知道 \(x_0\)；但生成时只有噪声 \(x_T\)。
    所以定义

    \[
    p_\theta(x_{t-1}\mid x_t)
    =\mathcal N(\mu_\theta(x_t,t),\sigma_t^2I),
    \]

    用网络从 \(x_t,t\) 预测 posterior mean 所需的信息。
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 10. 错误与反例

    1. **生成时把时间从小到大循环。**
       Forward 是 \(0\to T\)，reverse sampling 必须是 \(T\to0\)。

    2. **把 posterior variance 直接写成 \(\beta_t\)。**
       \(\tilde\beta_t\) 还包含累计条件信息，通常小于 \(\beta_t\)。

    3. **在 \(t=1\) 仍加入随机噪声。**
       理论 posterior variance 为 0；最后一步通常使用 mean，不再注入噪声。

    4. **忘记真实生成时没有 \(x_0\)。**
       解析 posterior 是训练目标和推导工具，不能原样作为生成算法输入。

    ## 原始资料

    - [Ho et al., DDPM，公式 (6)–(7)](https://arxiv.org/abs/2006.11239)
    """)
    return


@app.cell
def _(exercise_block, mo):
    mo.vstack(
        [
            mo.md("## 11. 分层练习"),
            exercise_block(
                (
                    "posterior mean 为什么同时包含 x0 和 xt？",
                    "x0 提供整条链的起点信息，xt 提供当前观测信息；Bayes posterior 合并两者。",
                ),
                (
                    r"当 \(t=1\) 时，为什么 \(\tilde\beta_1=0\)？",
                    r"因为 \(\bar\alpha_0=1\)，分子包含 \(1-\bar\alpha_0=0\)。给定 x0 时，x0 本身没有不确定性。",
                ),
                (
                    "若 reverse loop 写成 `for t in range(T)`，逻辑错误是什么？",
                    "它从接近数据的一端继续走向噪声，而不是从 xT 逐步恢复 x0。",
                ),
                (
                    "移动 xt，观察 posterior mean。它是否简单等于 x0 与 xt 的算术平均？",
                    "不是。权重由 beta、alpha 和累计 alpha_bar 决定，并非固定各占一半。",
                ),
            ),
        ],
        gap=0.8,
    )
    return


@app.cell
def _(CHAPTERS, chapter_footer):
    chapter_footer(
        CHAPTERS[16],
        [
            "Bayes rule 将一步 likelihood 与前一时刻 marginal 合成 Gaussian posterior。",
            "完成平方可以从二次指数中读出 posterior mean 和 variance。",
            "解析 posterior 依赖 x0，而真实生成只有 xt。",
            "因此网络必须预测 x0、噪声或其他等价量来构造 reverse mean。",
        ],
    )
    return


if __name__ == "__main__":
    app.run()
