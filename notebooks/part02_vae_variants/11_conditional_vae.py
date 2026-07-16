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
    chapter_header(CHAPTERS[11], duration="70–100 分钟")
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 1. 本章为什么存在

    普通 VAE 能生成“像数据”的样本，却不能直接要求“生成数字 7”或“生成红色物体”。
    Conditional VAE 把控制变量 y 同时交给 encoder 和 decoder：

    \[
    q_\phi(z|x,y),\qquad p_\theta(x|z,y).
    \]

    ## 2. 你已经知道什么

    - 第 2–3 章：条件概率的竖线表示已知信息。
    - 第 7 章：z 表示未直接观测的变化因素。
    - **回忆：** 若 decoder 不收到 y，它如何知道用户要求的类别？
    """)
    return


@app.cell
def _(intuition_and_rigor):
    intuition_and_rigor(
        "z 像“写法和风格”，y 像“要写哪个数字”。同一个风格 z 配不同 y，可以生成不同类别；同一个 y 配不同 z，可以生成同类中的变化。",
        r"""
        常见生成模型为 \(p(y)p(z|y)p_\theta(x|z,y)\)，简化实现也常取
        \(p(z)=N(0,I)\)。条件 ELBO：
        \[
        \log p(x|y)\ge E_{q(z|x,y)}\log p_\theta(x|z,y)
        -KL(q_\phi(z|x,y)\|p(z|y)).
        \]
        若 prior 不依赖 y，则最后一项写成 KL 到 p(z)。
        """,
    )
    return


@app.cell
def _(derivation_map, mo):
    mo.vstack(
        [
            mo.md("## 3–6. 条件信息进入哪里"),
            derivation_map(["观测 x 与条件 y", "encoder 推断 z", "重参数化采样", "decoder 接收 z 与 y", "最大化 conditional ELBO"]),
            mo.md(r"""
            One-hot y shape `[B,num_classes]`；与 x 或 hidden 沿 feature 轴拼接。
            条件独立假设不是“x 与 y 无关”，而是具体概率图中声明：给定哪些父节点后，
            某变量不再依赖其他变量。使用前必须画清模型，不可凭直觉删条件。

            ### 条件 ELBO 从哪里来？

            对固定的条件 y，从恒等式开始：

            \[
            \log p_\theta(x|y)
            =
            \log\int p_\theta(x,z|y)dz.
            \]

            乘除可计算的 \(q_\phi(z|x,y)\)，再用 Jensen：

            \[
            \log p_\theta(x|y)
            \ge
            E_q\left[
            \log\frac{p_\theta(x,z|y)}{q_\phi(z|x,y)}
            \right].
            \]

            若概率图规定
            \(p_\theta(x,z|y)=p_\theta(x|z,y)p(z|y)\)，便得到

            \[
            E_q\log p_\theta(x|z,y)
            -KL(q_\phi(z|x,y)\|p(z|y)).
            \]

            如果另行假设 \(Z\) 与 \(Y\) 在 prior 下独立，才可以把 \(p(z|y)\)
            简化为 \(p(z)\)。这是模型选择，不是由“conditional VAE”四个字自动推出。

            一个常见概率图是 \(Y\to X,\ Z\to X\)，并让 encoder 近似后验读取
            \(X,Y\)。它表达“给定 z,y 后生成 x”，并不表达 x 与 y 无关。
            """),
        ],
        gap=0.8,
    )
    return


@app.cell
def _(mo):
    condition = mo.ui.dropdown(options={"类别 A": 0, "类别 B": 1}, value="类别 A", label="条件 y")
    style = mo.ui.slider(-2.5, 2.5, value=0.0, step=0.1, show_value=True, label="latent style z")
    return condition, style


@app.cell
def _(COLORS, condition, mo, np, plt, style):
    _rng = np.random.default_rng(7)
    _centers = np.array([[-1.5, 0.0], [1.5, 0.0]])
    _data_a = _centers[0] + 0.35 * _rng.standard_normal((300, 2))
    _data_b = _centers[1] + 0.35 * _rng.standard_normal((300, 2))
    _direction = np.array([0.0, 0.55])
    _generated = _centers[condition.value] + style.value * _direction
    _fig, _axes = plt.subplots(1, 2, figsize=(9, 3.8))
    _axes[0].scatter(_data_a[:, 0], _data_a[:, 1], s=6, alpha=0.3, color=COLORS["data"], label="y=A")
    _axes[0].scatter(_data_b[:, 0], _data_b[:, 1], s=6, alpha=0.3, color=COLORS["model"], label="y=B")
    _axes[0].scatter(*_generated, s=100, marker="*", color=COLORS["danger"])
    _axes[0].legend(fontsize=8)
    _z = np.linspace(-2.5, 2.5, 100)
    _axes[1].plot(_z, _centers[0,1] + _z*_direction[1], label="A: change z", color=COLORS["data"])
    _axes[1].plot(_z, _centers[1,1] + _z*_direction[1], label="B: change z", color=COLORS["model"])
    _axes[1].set_xlabel("z")
    _axes[1].set_ylabel("生成样本纵坐标")
    _axes[1].legend(fontsize=8)
    _fig.tight_layout()
    mo.vstack([mo.md(fr"""## 7、9. 条件与 style 分离的 toy 实验

    当前生成点=`{_generated}`。切换 y 改变类别中心；移动 z 改变类内风格。
    真实 CVAE 未必自动得到如此干净的语义分离，这只是设计目标。"""),
    mo.hstack([condition, style], widths="equal"), _fig])
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 8. 代码映射

    ```python
    # y_onehot: [B, C]；x: [B, input_dim]。
    encoder_input = torch.cat([x, y_onehot], dim=-1)
    mu, logvar = encoder(encoder_input)
    z = reparameterize(mu, logvar)

    # decoder 必须再次收到 y，否则生成时条件无法生效。
    decoder_input = torch.cat([z, y_onehot], dim=-1)
    x_hat = decoder(decoder_input)
    ```

    ## 10. 错误与反例

    - 只给 encoder 条件：训练可能推断更容易，但 decoder 生成时无法控制。
    - 未声明 prior 是否依赖 y，就在 \(p(z)\) 与 \(p(z|y)\) 之间随意切换。
    - 把类别整数直接当连续大小：类别 2 不一定是类别 1 的两倍，应使用 embedding/one-hot。
    - 训练时使用真实 y、生成时遗漏 y：输入 shape 和概率问题同时出错。

    原始资料：[Sohn et al., Learning Structured Output Representation using Deep Conditional Generative Models](https://papers.nips.cc/paper/5775-learning-structured-output-representation-using-deep-conditional-generative-models)
    """)
    return


@app.cell
def _(exercise_block, mo):
    mo.vstack(
        [
            mo.md("## 11. 分层练习"),
            exercise_block(
                ("y 与 z 分别承担什么角色？", "y 提供显式控制条件，z 表示在给定条件下仍未指定的变化。"),
                ("B=32、C=10，one-hot y shape 是什么？", "`[32,10]`。"),
                ("为什么 decoder 也需要 y？", "生成时只有 z 和目标条件，decoder 必须读取 y 才能改变输出类别。"),
                ("固定 z 切换 y，再固定 y 移动 z。", "前者应主要改变条件类别，后者应展示类内变化；真实模型需实验验证。"),
            ),
        ],
        gap=0.8,
    )
    return


@app.cell
def _(CHAPTERS, chapter_footer):
    chapter_footer(CHAPTERS[11], ["CVAE 建模条件 likelihood。", "y 与 z 分别承载显式条件和剩余变化。", "encoder 与 decoder 通常都要接收 y。", "下一章把连续 latent 改成离散 code。"])
    return


if __name__ == "__main__":
    app.run()
