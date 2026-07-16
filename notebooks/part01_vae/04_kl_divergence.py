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
def _(CHAPTERS, chapter_header, course_styles):
    course_styles()
    chapter_header(CHAPTERS[4], duration="75–105 分钟")
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 1. 本章为什么存在

    假设真实天气分布认为“下雨”的概率是 80%，而某个近似预报只给 30%。
    仅说“相差 50 个百分点”还不够，因为概率错误的代价取决于事件在真实分布下有多常发生。

    VAE 需要让 approximate posterior \(q_\phi(z\mid x)\) 接近另一个分布。
    我们需要一种衡量**整个概率分布差异**的量，这就是 KL divergence。

    ## 2. 你已经知道什么

    - 概率向量中的每项非负，总和为 1。
    - 对数满足 \(\log(a/b)=\log a-\log b\)。
    - 第 3 章：条件概率的方向不能随意交换。

    **回忆问题：** 如果一个近似分布把真实可能发生的事件写成概率 0，会有什么危险？
    """)
    return


@app.cell
def _(intuition_and_rigor):
    intuition_and_rigor(
        r"""
        把 \(q\) 看成你用来编码或下注的“错误世界观”，把 \(p\) 看成真实世界。
        KL 衡量：真实事件按照 \(p\) 发生时，使用 \(q\) 的错误概率判断平均会付出多少额外代价。
        """,
        r"""
        对离散分布，若 \(p\) 对 \(q\) 绝对连续，即 \(p_i>0\Rightarrow q_i>0\)，则

        \[
        D_{KL}(p\|q)=\sum_i p_i\log\frac{p_i}{q_i}.
        \]

        若存在 \(p_i>0,q_i=0\)，则 \(D_{KL}(p\|q)=+\infty\)。
        KL 非负，且在满足常规条件时仅当 \(p=q\) 时为 0；但它不是距离，因为通常不对称。
        """,
    )
    return


@app.cell
def _(derivation_map, mo):
    mo.md("## 3–5. 从单次惊讶到平均差异")
    derivation_map(["比较同一事件的 p_i 与 q_i", "取对数比 log(p_i/q_i)", "按真实概率 p_i 加权", "对所有事件求和", "得到平均额外信息"])
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### 期望就是概率加权平均

    普通平均把每个数看得一样重要；期望按照事件发生概率加权：

    \[
    \mathbb E_{X\sim p}[f(X)]=\sum_i p_i f(i).
    \]

    令 \(f(i)=\log(p_i/q_i)\)，就得到 KL。

    ### 为什么 KL 不对称？

    \(D_{KL}(p\|q)\) 的平均权重来自 \(p\)，而 \(D_{KL}(q\|p)\) 的平均权重来自 \(q\)。
    两个问题分别问“用 q 近似 p”和“用 p 近似 q”，关注的事件区域不同。

    ### 非负性的严格路线

    因为 \(-\log u\) 是凸函数，由 Jensen inequality：

    \[
    \mathbb E_p\left[-\log\frac{q(X)}{p(X)}\right]
    \ge -\log\mathbb E_p\left[\frac{q(X)}{p(X)}\right]
    \]

    注意期望只遍历 \(p_i>0\) 的事件，因此

    \[
    \mathbb E_p\left[\frac{q(X)}{p(X)}\right]
    =\sum_{i:p_i>0}q_i\le 1.
    \]

    于是

    \[
    D_{KL}(p\|q)
    \ge-\log\sum_{i:p_i>0}q_i
    \ge-\log1=0.
    \]

    如果 \(q\) 在 \(p\) 的支持集之外没有概率质量，中间的和才恰好等于 1。
    例如 \(p=(1,0),q=(0.5,0.5)\) 时，这个和是 0.5，而不是 1；
    此时 \(D_{KL}(p\|q)=\log2>0\)。定理仍然成立，但不能把隐藏的
    “两者支持集相同”当成默认条件。Jensen 将在第 5 章配合图形完整复用。

    ### 检查

    - \(p_i,q_i\) 是标量概率；
    - 比值进入对数前必须为正；
    - KL 最终是标量；
    - Jensen 证明中的求和只发生在 \(p\) 的支持集；
    - VAE 中必须始终写清是 \(KL(q_\phi(z|x)\|p(z))\) 还是反方向。
    """)
    return


@app.cell
def _(mo):
    p_rain = mo.ui.slider(0.01, 0.99, step=0.01, value=0.80, show_value=True, label="p(下雨)")
    q_rain = mo.ui.slider(0.01, 0.99, step=0.01, value=0.30, show_value=True, label="q(下雨)")
    return p_rain, q_rain


@app.cell
def _(COLORS, kl_discrete, mo, np, p_rain, plt, q_rain):
    _p = np.array([p_rain.value, 1.0 - p_rain.value])
    _q = np.array([q_rain.value, 1.0 - q_rain.value])
    _kl_pq = kl_discrete(_p, _q)
    _kl_qp = kl_discrete(_q, _p)

    _fig, (_ax1, _ax2) = plt.subplots(1, 2, figsize=(10, 4))
    _positions = np.arange(2)
    _ax1.bar(_positions - 0.18, _p, width=0.36, label="p", color=COLORS["data"])
    _ax1.bar(_positions + 0.18, _q, width=0.36, label="q", color=COLORS["model"])
    _ax1.set_xticks(_positions, ["下雨", "不下雨"])
    _ax1.set_ylim(0, 1)
    _ax1.legend()

    _q_grid = np.linspace(0.01, 0.99, 200)
    _curve = [
        kl_discrete(_p, np.array([_value, 1.0 - _value])) for _value in _q_grid
    ]
    _ax2.plot(_q_grid, _curve, color=COLORS["prior"])
    _ax2.axvline(_p[0], color=COLORS["success"], linestyle="--", label="q=p")
    _ax2.scatter([_q[0]], [_kl_pq], color=COLORS["danger"], zorder=3)
    _ax2.set_xlabel("q(下雨)")
    _ax2.set_ylabel("D_KL(p || q)")
    _ax2.legend()

    mo.vstack(
        [
            mo.md(
                fr"""
                ## 7. 交互实验：方向真的会改变结果

                \[
                D_{{KL}}(p\|q)={_kl_pq:.4f},\qquad
                D_{{KL}}(q\|p)={_kl_qp:.4f}.
                \]

                除非分布恰好具有特殊对称关系，两者通常不同。
                """
            ),
            mo.hstack([p_rain, q_rain], widths="equal"),
            _fig,
        ]
    )
    return


@app.cell
def _(kl_discrete, mo, np):
    _boundary_value = kl_discrete(np.array([1.0, 0.0]), np.array([0.0, 1.0]))
    mo.md(
        fr"""
        ## 8–10. 代码、数值验证与错误案例

        ```python
        # 只在 p_i > 0 的位置计算，因为极限 0 * log(0/q) 约定为 0。
        mask = p > 0

        # 但在计算前必须检查：p_i > 0 且 q_i == 0 时 KL 是 +∞。
        if np.any((p > 0) & (q == 0)):
            return float("inf")

        kl = np.sum(p[mask] * np.log(p[mask] / q[mask]))
        ```

        本项目修正后的边界测试：

        \[
        KL([1,0]\|[0,1])={_boundary_value}.
        \]

        ### 错误实现

        若使用 `(p > 0) & (q > 0)` 直接过滤，互不重叠的分布会把所有危险项删除，
        错算成 0。这不仅是数值小问题，而是违背了 KL 的支持集定义。
        """
    )
    return


@app.cell
def _(exercise_block, mo):
    mo.md("## 11. 分层练习")
    exercise_block(
        ("为什么 KL 不是普通几何距离？", "它通常不对称，而且不满足普通距离要求的对称性；方向决定由哪个分布提供期望权重。"),
        (r"计算 \(p=(0.5,0.5),q=(0.25,0.75)\) 的 KL，保留公式即可。", r"\(0.5\log(0.5/0.25)+0.5\log(0.5/0.75)\)。使用自然对数时约为 0.1438。"),
        ("为什么不能简单过滤 q=0 的位置？", "若该位置 p>0，真实世界可能发生而近似分布宣称绝不发生，对数比发散，KL 应为 +∞。"),
        ("固定 p，移动 q。先预测 KL 最小点，再观察曲线是否支持预测。", "最小点应在 q=p；曲线提供数值支持，非负性的严格理由来自 Jensen inequality。"),
    )
    return


@app.cell
def _(CHAPTERS, chapter_footer):
    chapter_footer(
        CHAPTERS[4],
        [
            "期望是按概率加权的平均。",
            "KL 衡量用 q 近似 p 的平均对数比代价。",
            "KL 有方向、非负，但不是对称距离。",
            "p>0 而 q=0 时 KL 为正无穷，代码不能静默过滤。",
        ],
    )
    return


if __name__ == "__main__":
    app.run()
