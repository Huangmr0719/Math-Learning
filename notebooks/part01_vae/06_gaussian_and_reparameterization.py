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
    from src.math_checks import finite_difference_gradient, gaussian_kl_standard_normal
    from src.teaching import CHAPTERS, chapter_footer, chapter_header, course_styles, derivation_map, exercise_block, intuition_and_rigor
    from src.visualization import COLORS, configure_matplotlib

    configure_matplotlib()
    import matplotlib.pyplot as plt

    return (
        CHAPTERS,
        COLORS,
        chapter_footer,
        chapter_header,
        course_styles,
        derivation_map,
        exercise_block,
        finite_difference_gradient,
        gaussian_kl_standard_normal,
        intuition_and_rigor,
        mo,
        np,
        plt,
    )


@app.cell
def _(CHAPTERS, chapter_header):
    chapter_header(CHAPTERS[6], duration="90–120 分钟")
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 1. 本章为什么存在

    ELBO 要计算 \(q_\phi(z\mid x)\) 下的期望，实际训练会从该分布采样。
    但普通“从某个分布随机抽一个数”的操作看起来无法对分布参数求导。

    重参数化的核心不是消除随机性，而是**移动随机性的位置**：

    \[
    z=\mu+\sigma\odot\epsilon,\qquad
    \epsilon\sim\mathcal N(0,I).
    \]

    随机源 \(\epsilon\) 不依赖 encoder 参数；\(z\) 对 \(\mu,\sigma\) 则是普通可微运算。

    ## 2. 你已经知道什么

    - Gaussian 的均值控制中心，标准差控制宽度；
    - ELBO 需要在 \(q_\phi(z\mid x)\) 下取期望；
    - 链式法则描述复合函数的梯度传播。
    """)
    return


@app.cell
def _(intuition_and_rigor):
    intuition_and_rigor(
        r"""
        想象先从一台固定的标准随机数机器取出 \(\epsilon\)，再由 encoder 决定把它平移多少、
        拉伸多少。反向传播不必“穿过随机数机器”，只需穿过后面的平移和缩放。
        """,
        r"""
        若 \(\epsilon\sim\mathcal N(0,I)\)，定义
        \(z=\mu+\operatorname{diag}(\sigma)\epsilon\)，则
        \(z\sim\mathcal N(\mu,\operatorname{diag}(\sigma^2))\)。

        对固定 \(\epsilon\)，有
        \(\partial z/\partial\mu=I\)，
        \(\partial z/\partial\sigma=\operatorname{diag}(\epsilon)\)，
        因而 decoder loss 的梯度可通过普通链式法则传播。
        """,
    )
    return


@app.cell
def _(derivation_map, mo):
    mo.vstack(
        [
            mo.md("## 3–5. Gaussian KL 与重参数化地图"),
            derivation_map(["encoder 输出 mu 与 logvar", "由 logvar 得到 std", "采样固定标准噪声 epsilon", "构造 z", "解析计算 KL(q||N(0,I))"]),
        ],
        gap=0.8,
    )
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### 为什么输出 logvar？

    方差必须大于 0，但神经网络线性层自然输出任意实数。
    令 `logvar = log(sigma^2)` 后，网络可以无约束输出实数，再恢复：

    \[
    \sigma=\exp\left(\frac12\log\sigma^2\right).
    \]

    ### 对角 Gaussian 到标准正态的 KL

    若
    \(q=\mathcal N(\mu,\operatorname{diag}(\sigma_1^2,\ldots,\sigma_d^2))\)，
    \(p=\mathcal N(0,I)\)，则

    \[
    D_{KL}(q\|p)
    =-\frac12\sum_{i=1}^d
    \left(1+\log\sigma_i^2-\mu_i^2-\sigma_i^2\right).
    \]

    注意：这里不是 \(\sigma^2I\)，除非每个维度方差完全相同。

    ### shape 检查

    - `mu`, `logvar`, `std`, `epsilon`, `z`：
      `[batch_size, latent_dim]`；
    - `sum(dim=-1)`：每个样本沿 latent dimension 求和；
    - 再对 batch 做 `mean()`，得到训练使用的标量 KL。
    """)
    return


@app.cell
def _(mo):
    mu_slider = mo.ui.slider(-3.0, 3.0, step=0.1, value=0.5, show_value=True, label="mu")
    logvar_slider = mo.ui.slider(-3.0, 2.0, step=0.1, value=-0.5, show_value=True, label="logvar")
    return logvar_slider, mu_slider


@app.cell
def _(
    COLORS,
    gaussian_kl_standard_normal,
    logvar_slider,
    mo,
    mu_slider,
    np,
    plt,
):
    _mu = float(mu_slider.value)
    _logvar = float(logvar_slider.value)
    _variance = float(np.exp(_logvar))
    _std = float(np.exp(0.5 * _logvar))
    _epsilon = np.random.default_rng(7).standard_normal(2000)
    _samples = _mu + _std * _epsilon

    _analytic_kl = gaussian_kl_standard_normal(np.array([_mu]), np.array([_logvar]))
    _log_q = -0.5 * (np.log(2 * np.pi) + _logvar + ((_samples - _mu) ** 2) / _variance)
    _log_p = -0.5 * (np.log(2 * np.pi) + _samples**2)
    _mc_kl = float(np.mean(_log_q - _log_p))

    _x = np.linspace(-6, 6, 500)
    _q_density = np.exp(-0.5 * ((_x - _mu) / _std) ** 2) / (_std * np.sqrt(2 * np.pi))
    _p_density = np.exp(-0.5 * _x**2) / np.sqrt(2 * np.pi)
    _fig, _ax = plt.subplots()
    _ax.plot(_x, _p_density, label="prior N(0,1)", color=COLORS["prior"])
    _ax.plot(_x, _q_density, label="q(z|x)", color=COLORS["data"])
    _ax.hist(_samples, bins=35, density=True, alpha=0.2, color=COLORS["data"])
    _ax.legend()
    _ax.set_title("同一批 epsilon 经平移和缩放得到 q 样本")

    mo.vstack(
        [
            mo.md(
                fr"""
                ## 7、9. 交互与 Monte Carlo 验证

                - variance \(=\exp(\text{{logvar}})={_variance:.4f}\)
                - std \(=\exp(0.5\,\text{{logvar}})={_std:.4f}\)
                - analytic KL \(={_analytic_kl:.5f}\)
                - Monte Carlo KL \(={_mc_kl:.5f}\)
                - absolute error \(={abs(_analytic_kl-_mc_kl):.5f}\)

                数值接近支持实现正确，但严格公式仍来自 Gaussian KL 推导。
                """
            ),
            mo.hstack([mu_slider, logvar_slider], widths="equal"),
            _fig,
        ]
    )
    return


@app.cell
def _(finite_difference_gradient, mo, np):
    _epsilon_fixed = 0.7

    def _loss_from_parameters(_parameters):
        _mu, _logvar = _parameters
        _std = np.exp(0.5 * _logvar)
        _z = _mu + _std * _epsilon_fixed
        return float(_z**2)

    _point = np.array([0.4, -0.2])
    _numeric_gradient = finite_difference_gradient(_loss_from_parameters, _point)
    _mu, _logvar = _point
    _std = np.exp(0.5 * _logvar)
    _z = _mu + _std * _epsilon_fixed
    _analytic_gradient = np.array([2 * _z, _z * _std * _epsilon_fixed])
    mo.md(
        fr"""
        ## 8. 梯度验证与逐行代码

        ```python
        # logvar = log(sigma^2)，乘 0.5 后再 exp 得到 sigma。
        std = torch.exp(0.5 * logvar)

        # epsilon 与 std shape 相同，不依赖 encoder 参数。
        epsilon = torch.randn_like(std)

        # 对应 z = mu + sigma ⊙ epsilon。
        z = mu + std * epsilon
        ```

        对 toy loss \(L=z^2\)，固定 \(\epsilon=0.7\)：

        - 链式法则解析梯度：`{_analytic_gradient}`
        - 有限差分数值梯度：`{_numeric_gradient}`
        - 最大绝对误差：`{np.max(np.abs(_analytic_gradient-_numeric_gradient)):.2e}`

        有限差分检查的是实现与手推梯度是否一致，不是对所有输入的形式证明。
        """
    )
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 10. 错误与反例

    1. **把 logvar 当作标准差。**
       若 `logvar=-2`，正确标准差是 `exp(-1)`，不是 `-2`。标准差不能为负。

    2. **使用 `exp(logvar)` 作为标准差。**
       `exp(logvar)` 是方差 \(\sigma^2\)，会多平方一次缩放程度。

    3. **认为重参数化取消了随机性。**
       随机性仍在 \(\epsilon\) 中；改变随机种子仍会产生不同 \(z\)。
    """)
    return


@app.cell
def _(exercise_block, mo):
    mo.vstack(
        [
            mo.md("## 11. 分层练习"),
            exercise_block(
                ("重参数化究竟把随机性从哪里移动到了哪里？", "从“直接由带参数分布采样 z”移动到固定标准分布 epsilon；mu 和 sigma 只参与普通可微变换。"),
                (r"若 logvar=0，variance 和 std 分别是多少？", r"variance \(=e^0=1\)，std \(=e^{0/2}=1\)。"),
                ("解释 `torch.sum(..., dim=-1).mean()` 的两个 reduction。", "先沿最后一个 latent 维度为每个样本求 KL，再沿 batch 对样本取平均，最终得到标量。"),
                ("固定 logvar=0，改变 mu；再固定 mu=0，改变 logvar。比较两种偏离 prior 的方式。", "mu 偏离 0 会产生二次惩罚；logvar 偏离 0 表示方差偏离 1，两边都会增加 KL，但曲线并不对称。"),
            ),
        ],
        gap=0.8,
    )
    return


@app.cell
def _(CHAPTERS, chapter_footer):
    chapter_footer(
        CHAPTERS[6],
        [
            "重参数化把随机性放入与参数无关的 epsilon。",
            "logvar 是 log(sigma^2)，std = exp(0.5 * logvar)。",
            "对角 Gaussian KL 可以解析计算，并与 Monte Carlo 交叉验证。",
            "shape、reduction 轴和方差定义是从公式落到代码的关键。",
        ],
    )
    return


if __name__ == "__main__":
    app.run()
