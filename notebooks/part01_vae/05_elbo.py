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
    from src.math_checks import kl_discrete
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
        kl_discrete,
        mo,
        np,
        plt,
    )


@app.cell
def _(CHAPTERS, chapter_header):
    chapter_header(CHAPTERS[5], duration="90–120 分钟")
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 1. 本章为什么存在

    理想目标是让训练数据拥有高概率，即最大化 \(\log p_\theta(x)\)。
    但第 3 章已经看到：

    \[
    p_\theta(x)=\int p_\theta(x\mid z)p(z)\,dz
    \]

    对高维神经网络通常无法直接计算。ELBO 的作用不是“随便换一个 loss”，而是构造一个：

    - 可以计算和优化；
    - 永远不超过 \(\log p_\theta(x)\)；
    - 越接近它越好的下界。

    ## 2. 你已经知道什么

    - Bayes rule；
    - KL divergence 非负；
    - \(q_\phi(z\mid x)\) 用来近似真实 posterior。

    **回忆问题：** 若能写成“目标 = 可计算量 + 非负量”，可计算量与目标是什么关系？
    """)
    return


@app.cell
def _(intuition_and_rigor):
    intuition_and_rigor(
        r"""
        想象真正的山顶被雾挡住，无法直接测量。ELBO 是一个能站上去测量的观景台。
        观景台不会高过山顶；当 approximate posterior 足够准确时，观景台升到山顶。
        """,
        r"""
        对任意满足支持集条件的 \(q_\phi(z\mid x)\)，有恒等式

        \[
        \log p_\theta(x)
        =\mathcal L_{\mathrm{ELBO}}(x)
        +D_{KL}\!\left(q_\phi(z\mid x)\|p_\theta(z\mid x)\right).
        \]

        因 KL 非负，所以 \(\mathcal L_{\mathrm{ELBO}}(x)\le\log p_\theta(x)\)。
        等号成立当且仅当 approximate posterior 与真实 posterior 相同（几乎处处）。
        """,
    )
    return


@app.cell
def _(derivation_map, mo):
    mo.vstack(
        [
            mo.md("## 3–5. ELBO 推导地图"),
            derivation_map(["从 posterior KL 出发", "展开对数比", "代入 Bayes rule", "把 log p(x) 移出期望", "整理成 log evidence = ELBO + gap"]),
        ],
        gap=0.8,
    )
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### 路径一：从 posterior KL 出发

    \[
    \begin{aligned}
    D_{KL}(q_\phi(z\mid x)\|p_\theta(z\mid x))
    &=\mathbb E_q\left[
    \log\frac{q_\phi(z\mid x)}{p_\theta(z\mid x)}
    \right]\\
    &\overset{\text{Bayes}}=
    \mathbb E_q\left[
    \log\frac{q_\phi(z\mid x)p_\theta(x)}
    {p_\theta(x,z)}
    \right]\\
    &=\log p_\theta(x)
    -\mathbb E_q\left[
    \log\frac{p_\theta(x,z)}{q_\phi(z\mid x)}
    \right].
    \end{aligned}
    \]

    定义最后的期望为 ELBO：

    \[
    \mathcal L_{\mathrm{ELBO}}
    =\mathbb E_q\left[
    \log\frac{p_\theta(x,z)}{q_\phi(z\mid x)}
    \right].
    \]

    使用 \(p_\theta(x,z)=p_\theta(x\mid z)p(z)\) 展开：

    \[
    \mathcal L_{\mathrm{ELBO}}
    =\underbrace{\mathbb E_q[\log p_\theta(x\mid z)]}_{\text{重构/似然项}}
    -\underbrace{D_{KL}(q_\phi(z\mid x)\|p(z))}_{\text{正则项}}.
    \]

    ### 关于符号

    上式是需要**最大化**的 ELBO。代码常最小化负 ELBO：

    \[
    L_{\text{train}}
    =L_{\text{reconstruction}}+L_{\text{KL}}.
    \]

    两者相差整体负号。混淆“最大化目标”和“最小化 loss”会导致符号错误。
    """)
    return


@app.cell
def _(mo):
    q_z1 = mo.ui.slider(0.01, 0.99, step=0.01, value=0.50, show_value=True, label="q(z=1 | x=1)")
    return (q_z1,)


@app.cell
def _(COLORS, kl_discrete, mo, np, plt, q_z1):
    _prior = np.array([0.5, 0.5])
    _likelihood_x1 = np.array([0.1, 0.8])
    _joint = _prior * _likelihood_x1
    _evidence = float(_joint.sum())
    _posterior = _joint / _evidence
    _q = np.array([1.0 - q_z1.value, q_z1.value])

    _expected_log_likelihood = float(np.sum(_q * np.log(_likelihood_x1)))
    _prior_kl = kl_discrete(_q, _prior)
    _elbo = _expected_log_likelihood - _prior_kl
    _log_evidence = float(np.log(_evidence))
    _gap = kl_discrete(_q, _posterior)

    _fig, _ax = plt.subplots(figsize=(8, 4))
    _ax.bar(
        ["ELBO", "posterior KL gap", "log p(x)"],
        [_elbo, _gap, _log_evidence],
        color=[COLORS["data"], COLORS["model"], COLORS["success"]],
    )
    _ax.axhline(0, color="black", linewidth=0.8)
    _ax.set_title("恒等式检查：ELBO + gap = log p(x)")

    mo.vstack(
        [
            mo.md(
                fr"""
                ## 7、9. 交互验证：让 q 靠近真实 posterior

                这个二元 latent 模型可精确计算：

                - 真实 \(p(z=1\mid x=1)={_posterior[1]:.4f}\)；
                - 当前 ELBO \(={_elbo:.4f}\)；
                - posterior KL gap \(={_gap:.4f}\)；
                - \(\log p(x)={_log_evidence:.4f}\)；
                - 恒等式误差 \(={abs((_elbo + _gap) - _log_evidence):.2e}\)。

                把 \(q(z=1\mid x)\) 调到真实 posterior 附近，gap 会接近 0，ELBO 会贴近 evidence。
                """
            ),
            q_z1,
            _fig,
        ]
    )
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 6、8、10. 检查、代码映射与错误案例

    ```python
    # q 的 shape 是 [num_latent_states]。
    expected_log_likelihood = np.sum(q * np.log(likelihood_x))

    # KL(q || prior) 的方向不能交换。
    regularizer = kl_discrete(q, prior)

    # 数学上最大化 ELBO；训练代码通常最小化 -ELBO。
    elbo = expected_log_likelihood - regularizer
    training_loss = -elbo
    ```

    ### 检查

    - 所有期望都必须注明对哪个分布求：这里是 \(q_\phi(z\mid x)\)；
    - ELBO、KL、log evidence 都是标量；
    - `log(likelihood)` 要求 likelihood 正；
    - “等价目标”要说明是差一个负号、常数还是正比例。

    ### 常见错误

    把 `reconstruction + KL` 直接称为 ELBO。严格地说，它通常是**负 ELBO**
    或其 Monte Carlo 估计。命名不清会让最大化、最小化和符号全部混乱。
    """)
    return


@app.cell
def _(exercise_block, mo):
    mo.vstack(
        [
            mo.md("## 11. 分层练习"),
            exercise_block(
                ("为什么 ELBO 是下界？", "因为 log p(x) 等于 ELBO 加上一个非负的 posterior KL，所以 ELBO 不可能超过 log p(x)。"),
                ("若 log p(x)=-1.2，posterior KL=0.3，ELBO 是多少？", r"由恒等式得 ELBO \(=-1.2-0.3=-1.5\)。"),
                ("训练代码为什么通常写 reconstruction_loss + kl_loss？", "代码执行最小化，因此使用负 ELBO；负的 expected log likelihood 成为 reconstruction loss，减 KL 变为加正 KL。"),
                ("拖动 q，找出 ELBO 最大的位置，并与真实 posterior 比较。", "ELBO 在 q 等于真实 posterior 时达到 log evidence；滑块离开该位置时 posterior KL gap 增大。"),
            ),
        ],
        gap=0.8,
    )
    return


@app.cell
def _(CHAPTERS, chapter_footer):
    chapter_footer(
        CHAPTERS[5],
        [
            "ELBO 是 log evidence 的可优化下界。",
            "log p(x) = ELBO + approximate posterior 到真实 posterior 的 KL gap。",
            "ELBO 分成 expected log likelihood 与 prior KL 两部分。",
            "数学最大化 ELBO，工程代码通常最小化负 ELBO。",
        ],
    )
    return


if __name__ == "__main__":
    app.run()
