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
        intuition_and_rigor,
        mo,
        np,
        plt,
    )


@app.cell
def _(CHAPTERS, chapter_header):
    chapter_header(CHAPTERS[2], duration="60–90 分钟")
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 1. 本章为什么存在

    上一章的 encoder 为每个输入给出一个点 \(z\)。问题是：这些点之间可能有大片空洞，
    我们也不知道随机生成时应该从哪里取点。

    VAE 的关键转变是：

    > 不再说“这个输入对应 latent 点 1.7”，而是说
    > “这个输入对应一个以 1.7 为中心、带有一定不确定性的分布”。

    分布给我们两样东西：哪些位置更可能，以及怎样随机采样。

    ## 2. 你已经知道什么

    - 第 1 章：latent 是压缩表示。
    - 一个确定的点只有一个位置；一个分布描述一组可能位置。

    **回忆问题：** 为什么只知道训练样本的 latent 点，还不足以规定随机生成方式？
    """)
    return


@app.cell
def _(intuition_and_rigor):
    intuition_and_rigor(
        r"""
        把确定 latent 点想成地图上的一枚图钉；把 latent distribution 想成图钉周围的一团雾。
        雾最浓的地方最可能，离中心越远越不可能。方差控制雾扩散得有多宽。
        """,
        r"""
        随机变量 \(Z\) 不是“不断变化的普通变量”，而是一次随机试验的数值结果。
        连续随机变量通过概率密度 \(p(z)\) 描述。区间概率由密度曲线下面积给出：

        \[
        P(a\le Z\le b)=\int_a^b p(z)\,dz,\qquad
        \int_{-\infty}^{\infty}p(z)\,dz=1.
        \]

        单点的密度值可以大于 1；真正必须位于 \([0,1]\) 的是区间概率。
        """,
    )
    return


@app.cell
def _(derivation_map, mo):
    mo.vstack(
        [
            mo.md("## 3–5. 即时数学与概率模型"),
            derivation_map(["选择 latent z", "根据 z 生成 x", "考虑所有可能 z", "得到数据的边缘概率 p(x)"]),
        ],
        gap=0.8,
    )
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### 条件概率与边缘概率

    \(p_\theta(x\mid z)\) 表示“已经知道 \(z\) 时，生成 \(x\) 的可能性”。
    但观察数据时，我们并不知道实际使用了哪个 \(z\)，所以要把所有可能的 \(z\)
    都考虑进去：

    \[
    p_\theta(x)=\int p_\theta(x\mid z)p(z)\,dz.
    \]

    这相当于：

    1. 用 prior \(p(z)\) 衡量每个 latent 位置本来有多可能；
    2. 用 likelihood \(p_\theta(x\mid z)\) 衡量该位置生成 \(x\) 的能力；
    3. 对所有 latent 位置做加权汇总。

    离散情况下积分变成加法。例如有两个 latent 状态：

    \[
    p(x)=p(x\mid z=0)p(z=0)+p(x\mid z=1)p(z=1).
    \]

    ### 维度与定义域

    - \(z\) 可以是标量或向量；标准 VAE 常令 \(z\in\mathbb R^k\)。
    - \(p(z)\ge 0\)，且总积分为 1。
    - \(p_\theta(x\mid z)\) 对固定 \(z\) 必须是关于 \(x\) 的合法分布。
    - \(p_\theta(x)\) 是积分后的边缘分布，不再显式依赖 \(z\)。
    """)
    return


@app.cell
def _(mo):
    mu_slider = mo.ui.slider(-3.0, 3.0, step=0.1, value=0.0, show_value=True, label="均值 mu")
    sigma_slider = mo.ui.slider(0.2, 2.0, step=0.1, value=1.0, show_value=True, label="标准差 sigma")
    return mu_slider, sigma_slider


@app.cell
def _(COLORS, mo, mu_slider, np, plt, sigma_slider):
    _mu = float(mu_slider.value)
    _sigma = float(sigma_slider.value)
    _x = np.linspace(-6, 6, 500)
    _density = np.exp(-0.5 * ((_x - _mu) / _sigma) ** 2) / (_sigma * np.sqrt(2 * np.pi))
    _area = float(np.trapz(_density, _x))
    _rng = np.random.default_rng(7)
    _samples = _mu + _sigma * _rng.standard_normal(500)

    _fig, (_ax1, _ax2) = plt.subplots(1, 2, figsize=(10, 4))
    _ax1.plot(_x, _density, color=COLORS["prior"], linewidth=2)
    _ax1.fill_between(_x, _density, alpha=0.2, color=COLORS["prior"])
    _ax1.axvline(_mu, color=COLORS["model"], linestyle="--", label="mu")
    _ax1.set_title(f"Gaussian density，数值面积 ≈ {_area:.4f}")
    _ax1.legend()
    _ax2.hist(_samples, bins=25, density=True, color=COLORS["data"], alpha=0.7)
    _ax2.set_title("固定随机种子的 500 次采样")

    mo.vstack(
        [
            mo.md(
                fr"""
                ## 7. 交互实验：点变成一团概率雾

                当前 \(Z\sim\mathcal N({_mu:.1f},{_sigma:.1f}^2)\)。

                - 改变均值：整团分布左右移动；
                - 改变标准差：分布变宽或变窄；
                - 曲线高度不是概率，曲线下的总面积才是 1。
                """
            ),
            mo.hstack([mu_slider, sigma_slider], widths="equal"),
            _fig,
        ]
    )
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 8–10. 代码映射、验证与反例

    ```python
    # epsilon 来自标准正态 N(0, 1)。
    epsilon = rng.standard_normal(500)

    # 乘 sigma 控制宽度，加 mu 控制中心：
    # z = mu + sigma * epsilon
    samples = mu + sigma * epsilon
    ```

    数值积分得到的面积接近 1，支持 Gaussian 密度已归一化这一结论；
    这不是替代解析积分证明。

    ### 常见错误

    **把密度 \(p(z)\) 当成“点 \(z\) 发生的概率”。**

    对连续变量，单个精确点的概率通常为 0。密度描述的是某个位置附近概率积累得有多快，
    区间概率才由曲线面积给出。
    """)
    return


@app.cell
def _(exercise_block, mo):
    mo.vstack(
        [
            mo.md("## 11. 分层练习"),
            exercise_block(
                ("均值和标准差分别控制 Gaussian 的什么？", "均值控制中心位置，标准差控制分散宽度；标准差必须大于 0。"),
                ("若两个 latent 状态先验各为 0.5，生成 x 的概率分别为 0.2 和 0.8，求 p(x)。", r"\(p(x)=0.2\times0.5+0.8\times0.5=0.5\)。"),
                ("为什么采样代码先生成 epsilon，再乘 sigma、加 mu？", "标准正态容易采样；仿射变换把它移动并缩放成目标 Gaussian，同时为后续重参数化铺路。"),
                ("保持 mu 不变，逐步增大 sigma。观察曲线峰值和样本范围，并解释为什么总面积仍接近 1。", "分布变宽时峰值下降，因为相同总概率被摊到更宽区间；归一化常数随 sigma 调整，使总面积保持 1。"),
            ),
        ],
        gap=0.8,
    )
    return


@app.cell
def _(CHAPTERS, chapter_footer):
    chapter_footer(
        CHAPTERS[2],
        [
            "随机变量用分布描述可能取值，而不是只给一个确定点。",
            "Gaussian 的均值控制中心，标准差控制宽度。",
            "边缘概率 p(x) 要汇总所有 latent z 的贡献。",
            "VAE 的生成方向是先采样 z，再通过 p(x|z) 生成 x。",
        ],
    )
    return


if __name__ == "__main__":
    app.run()
