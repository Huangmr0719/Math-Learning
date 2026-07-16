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
        course_styles,
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
        course_styles,
        derivation_map,
        exercise_block,
        intuition_and_rigor,
        mo,
        np,
        plt,
    )


@app.cell
def _(CHAPTERS, chapter_header, course_styles):
    course_styles()
    chapter_header(CHAPTERS[1], duration="60–90 分钟")
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 1. 本章为什么存在

    想象你要把一张 64 像素的小图片通过电话告诉朋友。逐像素朗读当然可以，
    但很慢。更聪明的方法是先概括：“这是一个稍微向右倾斜的 7”。

    Autoencoder 做的事情很相似：

    \[
    x \xrightarrow{\text{encoder}} z
    \xrightarrow{\text{decoder}} \hat{x}
    \]

    - \(x\)：原始数据；
    - \(z\)：压缩后的 latent representation；
    - \(\hat{x}\)：decoder 尝试还原出的数据。

    本章真正要追问的不是“能不能压缩”，而是：

    > 如果模型能把训练样本重构得很好，它是否已经学会了怎样生成新样本？

    答案是：**还没有。** 这是通往 VAE 的第一个缺口。

    ## 2. 你已经知道什么

    - 一个数字列表可以看成向量，例如 \((2, 3)\)。
    - 函数接收输入并产生输出，例如 \(f(x)=2x+1\)。
    - 两个数越接近，它们的差的平方通常越小。

    **回忆问题：** 为什么比较 \(x-\hat{x}\) 可以衡量重构是否准确？
    """)
    return


@app.cell
def _(intuition_and_rigor):
    intuition_and_rigor(
        r"""
        把一张图片想成空间中的一个点。图片有多少个像素，这个点就有多少个坐标。
        Encoder 像一台压缩机，把高维点压到低维；decoder 像解压机，再把低维点展开。

        latent dimension 越小，压缩越狠；但压得太狠，很多细节会丢失。
        """,
        r"""
        设输入 \(x\in\mathbb R^d\)，encoder 为
        \(f_\phi:\mathbb R^d\rightarrow\mathbb R^k\)，decoder 为
        \(g_\theta:\mathbb R^k\rightarrow\mathbb R^d\)，且通常 \(k<d\)。

        \[
        z=f_\phi(x),\qquad \hat{x}=g_\theta(z).
        \]

        训练通过最小化数据集上的平均重构误差学习参数 \(\phi,\theta\)。
        """,
    )
    return


@app.cell
def _(derivation_map, mo):
    mo.md("## 3–5. 即时数学与推导地图")
    derivation_map(
        [
            "把数据写成向量",
            "计算每个坐标的误差",
            "平方以避免正负抵消",
            "对坐标取平均",
            "对样本取平均",
        ]
    )
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### 欧氏距离与均方误差

    若 \(x=(2,3)\)，重构结果为 \(\hat{x}=(1,5)\)，两个坐标的误差是
    \(1\) 和 \(-2\)。直接相加会得到 \(-1\)，正负误差可能互相抵消。
    平方后得到 \(1\) 和 \(4\)，于是

    \[
    \operatorname{MSE}(x,\hat{x})
    =\frac{1}{d}\sum_{j=1}^{d}(x_j-\hat{x}_j)^2
    =\frac{1+4}{2}=2.5.
    \]

    **规则标签：**

    1. 坐标相减：代数；
    2. 平方：把方向不同的误差变成非负量；
    3. 求和并除以 \(d\)：算术平均。

    ### 维度检查

    - \(x,\hat{x}\in\mathbb R^d\)：长度相同的向量；
    - 每个 \((x_j-\hat{x}_j)^2\) 是标量；
    - 求和后仍是标量；
    - batch 训练时，通常先对 feature/pixel 求和或平均，再对 batch 平均。
    """)
    return


@app.cell
def _(mo):
    angle = mo.ui.slider(
        0,
        180,
        step=2,
        value=35,
        show_value=True,
        label="一维 latent 轴的角度（度）",
    )
    return (angle,)


@app.cell
def _(COLORS, angle, mo, np, plt):
    _points = np.array(
        [[-2.0, -1.2], [-1.4, -0.7], [-0.5, -0.1], [0.4, 0.4], [1.2, 0.9], [2.0, 1.4]]
    )
    _theta = np.deg2rad(angle.value)
    _direction = np.array([np.cos(_theta), np.sin(_theta)])

    # 每个二维点在一维轴上的坐标，就是点与单位方向向量的点积。
    # shape: [6, 2] @ [2] -> [6]
    _latent = _points @ _direction

    # decoder 在这个 toy example 中把一维坐标乘回方向向量。
    # shape: [6, 1] * [2] -> [6, 2]，这里发生 broadcasting。
    _reconstruction = _latent[:, None] * _direction
    _mse = float(np.mean((_points - _reconstruction) ** 2))

    _fig, _ax = plt.subplots()
    _ax.scatter(_points[:, 0], _points[:, 1], s=70, color=COLORS["data"], label="原始点 x")
    _ax.scatter(
        _reconstruction[:, 0],
        _reconstruction[:, 1],
        s=70,
        color=COLORS["model"],
        marker="x",
        label="重构点 x_hat",
    )
    _line = np.linspace(-2.8, 2.8, 2)[:, None] * _direction
    _ax.plot(_line[:, 0], _line[:, 1], color=COLORS["prior"], label="一维 latent 轴")
    for _point, _rec in zip(_points, _reconstruction):
        _ax.plot([_point[0], _rec[0]], [_point[1], _rec[1]], color=COLORS["muted"], alpha=0.5)
    _ax.set_aspect("equal")
    _ax.set_xlim(-3, 3)
    _ax.set_ylim(-2.4, 2.4)
    _ax.set_title(f"把二维数据压到一维：MSE = {_mse:.3f}")
    _ax.legend()
    mo.vstack(
        [
            mo.md(
                f"""
                ## 7. 交互实验：亲手寻找好的压缩方向

                当前角度：**{angle.value}°**，当前 MSE：**{_mse:.3f}**。

                拖动滑块。好的 latent 轴会沿着数据的主要方向，垂直误差较短；
                错误方向会让许多信息丢失。
                """
            ),
            _fig,
        ]
    )
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 8–10. 公式、代码验证与错误案例

    ```python
    # x 的 shape 是 [batch_size, input_dim]
    z = encoder(x)                 # 对应 z = f_phi(x)
    x_hat = decoder(z)             # 对应 x_hat = g_theta(z)

    # (x - x_hat) ** 2 对每个像素计算平方误差；
    # mean() 同时对 batch 和像素平均，最后得到一个标量。
    reconstruction_loss = ((x - x_hat) ** 2).mean()
    ```

    上面的二维实验用投影充当 encoder 和 decoder。它不是神经网络，
    但能精确显示“压缩方向不合适就会增加重构误差”。

    ### 如果做错会怎样？

    **误解：重构好，所以随机采样 latent 一定能生成好数据。**

    Autoencoder 只约束训练样本对应的若干 latent 点。它没有要求这些点填满空间，
    也没有规定应该从哪里采样。两个训练样本之间可能是 decoder 从未见过的空洞。
    因此“能重构”与“有明确、可采样的生成分布”是两件事。
    """)
    return


@app.cell
def _(exercise_block, mo):
    mo.md("## 11. 分层练习")
    exercise_block(
        (
            "用自己的话解释 encoder、latent 和 decoder 各自做什么。",
            "Encoder 压缩输入，latent 保存压缩表示，decoder 根据表示重建输入。关键是 latent 目前只是点，还不是具有明确采样规则的概率分布。",
        ),
        (
            r"计算 \(x=(0,2)\)、\(\hat{x}=(1,4)\) 的 MSE。",
            r"平方误差为 \((0-1)^2=1\)、\((2-4)^2=4\)，平均后为 \((1+4)/2=2.5\)。",
        ),
        (
            "把代码中的 `mean()` 改成 `sum()`，结果的含义发生了什么变化？",
            "它变成所有样本、所有坐标误差的总和，数值会随 batch size 和输入维度增长；比较不同实验时必须统一 reduction。",
        ),
        (
            "拖动角度，先预测哪个方向 MSE 最小，再用图验证。记录预测与结果是否一致。",
            "数据大致沿右上方向排列，所以约 30°–40° 的轴较好。探索题重在先预测，再根据误差线和 MSE 修正判断。",
        ),
    )
    return


@app.cell
def _(CHAPTERS, chapter_footer):
    chapter_footer(
        CHAPTERS[1],
        [
            "Autoencoder 学习的是确定性的压缩和重构映射。",
            "MSE 是坐标平方误差的平均，最终是一个标量。",
            "低维表示会丢失信息，latent dimension 与重构能力存在权衡。",
            "重构良好不等于 latent space 具有可采样的概率结构。",
        ],
    )
    return


if __name__ == "__main__":
    app.run()
