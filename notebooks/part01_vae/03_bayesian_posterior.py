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
    chapter_header(CHAPTERS[3], duration="60–90 分钟")
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 本章为什么存在

    生成模型沿着 \(z\rightarrow x\) 工作：先选 latent 原因，再得到观测结果。
    训练 encoder 时却要解决相反问题：已经看见 \(x\)，怎样反推 \(z\)？

    例如，两台机器都会生产红球，但比例不同。你拿到一个红球后，应该怎样判断它更可能来自哪台机器？
    这就是 posterior inference。

    ## 你已经知道什么

    - prior \(p(z)\)：观察数据之前，对 latent 原因的看法；
    - likelihood \(p(x\mid z)\)：给定原因后，看到结果的可能性；
    - 边缘概率 \(p(x)\)：把所有原因的贡献加起来。

    **回忆问题：** 为什么 likelihood 高的机器不一定拥有最高 posterior？
    """)
    return


@app.cell
def _(intuition_and_rigor):
    intuition_and_rigor(
        r"""
        Bayes rule 是“先验观点经过新证据修正”的计算规则。
        posterior 同时考虑两件事：某个原因原本常不常见，以及它产生当前证据有多擅长。
        """,
        r"""
        由条件概率定义，

        \[
        p(z,x)=p(x\mid z)p(z)=p(z\mid x)p(x).
        \]

        当 \(p(x)>0\) 时，两边相除得到

        \[
        p(z\mid x)=\frac{p(x\mid z)p(z)}{p(x)}.
        \]

        分母 \(p(x)\) 与具体候选 \(z\) 无关，负责让所有 posterior 概率重新归一化为 1。
        """,
    )
    return


@app.cell
def _(derivation_map, mo):
    mo.vstack(
        [
            mo.md("## Bayes rule 推导地图"),
            derivation_map(["写出联合概率", "用两种顺序分解联合概率", "令两式相等", "除以 evidence p(x)", "得到 posterior"]),
        ],
        gap=0.8,
    )
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### 一个可手算的例子

    - 机器 A 被选择的 prior：\(p(A)=0.8\)；
    - 机器 B 被选择的 prior：\(p(B)=0.2\)；
    - A 生产红球的概率：\(p(R\mid A)=0.3\)；
    - B 生产红球的概率：\(p(R\mid B)=0.9\)。

    先算红球的总概率：

    \[
    p(R)=0.3\times0.8+0.9\times0.2=0.42.
    \]

    再算红球来自 B 的 posterior：

    \[
    p(B\mid R)=\frac{0.9\times0.2}{0.42}\approx0.429.
    \]

    B 更擅长生产红球，但它原本很少被选择，所以 posterior 并没有达到 0.9。

    ### VAE 中的困难

    \[
    p_\theta(z\mid x)
    =\frac{p_\theta(x\mid z)p(z)}
    {\int p_\theta(x\mid z)p(z)\,dz}.
    \]

    当 decoder 是神经网络且 \(z\) 维度较高时，这个积分通常无法在可接受的
    计算量内精确求出（英文常称 **intractable**）。
    VAE 因此引入可计算的 approximate posterior \(q_\phi(z\mid x)\)。
    """)
    return


@app.cell
def _(mo):
    prior_b = mo.ui.slider(0.01, 0.99, step=0.01, value=0.20, show_value=True, label="prior p(B)")
    red_a = mo.ui.slider(0.01, 0.99, step=0.01, value=0.30, show_value=True, label="likelihood p(R|A)")
    red_b = mo.ui.slider(0.01, 0.99, step=0.01, value=0.90, show_value=True, label="likelihood p(R|B)")
    return prior_b, red_a, red_b


@app.cell
def _(COLORS, mo, np, plt, prior_b, red_a, red_b):
    _p_b = float(prior_b.value)
    _p_a = 1.0 - _p_b
    _joint_a = _p_a * float(red_a.value)
    _joint_b = _p_b * float(red_b.value)
    _evidence = _joint_a + _joint_b
    _posterior = np.array([_joint_a, _joint_b]) / _evidence

    _fig, _ax = plt.subplots(figsize=(7, 4))
    _x = np.arange(2)
    _ax.bar(_x - 0.18, [_p_a, _p_b], width=0.36, label="prior", color=COLORS["muted"])
    _ax.bar(_x + 0.18, _posterior, width=0.36, label="看到红球后的 posterior", color=COLORS["data"])
    _ax.set_xticks(_x, ["机器 A", "机器 B"])
    _ax.set_ylim(0, 1)
    _ax.set_ylabel("概率")
    _ax.legend()

    mo.vstack(
        [
            mo.md(
                fr"""
                ## 交互实验：证据如何修改 prior

                当前 \(p(B\mid R)={_posterior[1]:.3f}\)，
                \(p(A\mid R)={_posterior[0]:.3f}\)，两者之和为 **{_posterior.sum():.3f}**。
                """
            ),
            mo.hstack([prior_b, red_a, red_b], widths="equal"),
            _fig,
        ]
    )
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 检查、代码映射和反例

    ```python
    # joint_b 对应 p(R, B) = p(R | B) p(B)
    joint_b = likelihood_red_given_b * prior_b

    # evidence 汇总所有可能原因产生红球的联合概率。
    evidence = joint_a + joint_b

    # posterior 对应 Bayes rule。
    posterior_b = joint_b / evidence
    ```

    **定义域检查：** evidence 必须大于 0，否则“已经观察到红球”本身就是模型认为不可能的事件，
    条件概率没有定义。

    ### 常见错误：只看 likelihood

    直接把 \(p(R\mid B)\) 当成 \(p(B\mid R)\) 是典型的条件方向颠倒。
    前者问“B 多常产生红球”，后者问“红球多可能来自 B”，问题不同。
    """)
    return


@app.cell
def _(exercise_block, mo):
    mo.vstack(
        [
            mo.md("## 分层练习"),
            exercise_block(
                ("用一句话区分 prior、likelihood 和 posterior。", "Prior 是看数据前对原因的看法；likelihood 是给定原因后证据出现的可能性；posterior 是看见证据后更新的原因概率。"),
                ("若 p(A)=p(B)=0.5，p(R|A)=0.2，p(R|B)=0.8，求 p(B|R)。", r"\(p(R)=0.2\times0.5+0.8\times0.5=0.5\)，所以 \(p(B|R)=0.4/0.5=0.8\)。"),
                ("已知 `joint = prior * likelihood`，补全 `evidence = ____` 与 `posterior = ____`，再写一个归一化断言。", "`evidence = joint.sum()`，`posterior = joint / evidence`，并检查 `assert np.isclose(posterior.sum(), 1.0)`。这里要求 `evidence > 0`；它对应 Bayes rule 的分母。"),
                ("让 B 的 likelihood 很高但 prior 很低，观察 posterior。解释哪一个因素最终占主导。", "没有固定答案；posterior 由 prior 与 likelihood 的乘积共同决定，应比较两台机器的 joint weight，而不是单独看其中一个量。"),
            ),
        ],
        gap=0.8,
    )
    return


@app.cell
def _(CHAPTERS, chapter_footer):
    chapter_footer(
        CHAPTERS[3],
        [
            "Bayes rule 把 prior 与 likelihood 合并成 posterior。",
            "posterior 与 likelihood 的条件方向不同。",
            "evidence p(x) 负责汇总并归一化所有 latent 原因。",
            "VAE 的真实 posterior 通常因高维积分不可计算，所以需要 q_phi(z|x)。",
        ],
    )
    return


if __name__ == "__main__":
    app.run()
