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
    chapter_header(CHAPTERS[13], duration="100–135 分钟")
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 本章为什么存在

    对角 Gaussian 只能表达一个椭圆形峰，真实 posterior 可能弯曲、多峰或强相关。
    Normalizing flow 从简单样本 \(z_0\) 出发，连续应用可逆变换：

    \[
    z_K=f_K\circ\cdots\circ f_1(z_0).
    \]

    ## 你已经知道什么

    - 第 6 章：仿射变换可改变 Gaussian 中心和宽度。
    - 概率总量必须守恒。
    - **回忆：** 把坐标轴拉伸两倍后，同一概率质量为何密度要降低？
    """)
    return


@app.cell
def _(intuition_and_rigor):
    intuition_and_rigor(
        "把概率密度画在橡皮布上。橡皮布被拉长时，同一团概率被摊到更大面积，单位面积密度必须下降；压缩时密度上升。",
        r"""
        若 \(z=f(u)\) 可逆且可微，Jacobian
        \(J_f(u)=\partial f/\partial u\)。变量替换公式：
        \[
        p_Z(z)=p_U(u)|\det J_f(u)|^{-1},\quad u=f^{-1}(z).
        \]
        等价的 log 形式：
        \[
        \log p_Z(z)=\log p_U(u)-\log|\det J_f(u)|.
        \]
        determinant 为 0 表示该点的线性近似压扁了某个方向，inverse function
        theorem 不能保证附近存在光滑逆映射。Normalizing flow 通常使用更强条件：
        变换是光滑双射，且 Jacobian 在使用密度公式的区域内非奇异。
        """,
    )
    return


@app.cell
def _(derivation_map, mo):
    mo.vstack(
        [
            mo.md("## Change of variables"),
            derivation_map(["小区间概率守恒", "新旧区间长度由导数相连", "一维得到除以 |f'|", "多维用 Jacobian determinant", "多个 flow 的 log-det 相加"]),
            mo.md(r"""
            一维小区间：
            \[
            p_U(u)\,du=p_Z(z)\,dz,\quad dz=f'(u)du
            \Rightarrow p_Z(z)=p_U(u)/|f'(u)|.
            \]
            多维 determinant 描述局部体积缩放。shape 检查：单样本 Jacobian `[D,D]`，
            determinant 是标量；batch 时每个样本各有一个 log-det。
            """),
        ],
        gap=0.8,
    )
    return


@app.cell
def _(mo):
    warp = mo.ui.slider(-0.98, 1.5, value=0.6, step=0.02, show_value=True, label="warp a in z=u+a*tanh(u)")
    return (warp,)


@app.cell
def _(COLORS, mo, np, plt, warp):
    _u = np.linspace(-4,4,1200)
    _z = _u + warp.value*np.tanh(_u)
    _derivative = 1 + warp.value/(np.cosh(_u)**2)
    _p_u = np.exp(-0.5*_u**2)/np.sqrt(2*np.pi)
    _p_z = _p_u/np.abs(_derivative)
    _mass_u = np.trapz(_p_u, _u)
    _mass_z = np.trapz(_p_z, _z)
    _fig, _axes = plt.subplots(1,2,figsize=(10,3.8))
    _axes[0].plot(_u,_z,color=COLORS["model"], label="z=f(u)")
    _axes[0].plot(_u,_u,"--",color=COLORS["muted"], label="不变换 z=u")
    _axes[0].set_xlabel("u"); _axes[0].set_ylabel("z=f(u)")
    _axes[0].legend(fontsize=8)
    _axes[1].plot(_u,_p_u,label="base p(u)",color=COLORS["data"])
    _axes[1].plot(_z,_p_z,label="transformed p(z)",color=COLORS["prior"])
    _axes[1].set_xlabel("对应坐标 u 或 z")
    _axes[1].set_ylabel("概率密度")
    _axes[1].legend(fontsize=8)
    _fig.tight_layout()
    mo.vstack([mo.md(fr"""## 密度守恒实验

    最小导数={_derivative.min():.3f}；base mass={_mass_u:.6f}，
    transformed mass={_mass_z:.6f}。本例中导数始终为正，所以 f 严格递增并具有
    光滑逆映射。注意“可逆”和“导数处处非零”不是同一句话：\(f(u)=u^3\)
    全局可逆，但 \(f'(0)=0\)；它不满足 normalizing flow 常用的光滑非奇异条件。"""), warp, _fig])
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 代码映射

    ```python
    z = u + a * torch.tanh(u)

    # 一维 Jacobian 就是普通导数；多维 flow 需计算 determinant 或特殊结构。
    derivative = 1 + a * (1 - torch.tanh(u).pow(2))
    log_p_z = log_p_u - torch.log(torch.abs(derivative))
    ```

    删除 log-det correction 后，变换后的“密度”通常不再积分为 1。

    ## 错误与反例

    - 忘记绝对值：orientation reversal 会得到负“体积”。
    - determinant=0 仍机械套用光滑变量替换公式：局部逆可能不光滑，密度可能出现奇点。
    - 把“Jacobian 非奇异是 flow 的常用条件”误说成“所有可逆函数的必要条件”。
    - 把 elementwise derivative 当多维 determinant：只有对角 Jacobian 才能直接相乘。

    原始资料：[Rezende & Mohamed, Variational Inference with Normalizing Flows](https://arxiv.org/abs/1505.05770)
    """)
    return


@app.cell
def _(exercise_block, mo):
    mo.vstack(
        [
            mo.md("## 分层练习"),
            exercise_block(
                ("为什么拉伸区域密度下降？", "同一概率质量分布到更大体积，单位体积的概率密度必须下降。"),
                ("若 z=3u，p_z 与 p_u 的关系？", r"\(p_Z(z)=p_U(z/3)/3\)。"),
                ("补全一维变量变换：`log_pz = log_pu - ____`。若导数可能为负，代码中为什么必须有绝对值？", "填写 `torch.log(torch.abs(dz_du))`。密度由局部体积缩放的绝对大小决定；导数符号只表示方向翻转，不能让概率密度为负。"),
                ("将 a 调到接近 -1，观察导数。", "中心处导数接近 0，密度尖锐且数值不稳定，说明可逆性边界。"),
            ),
        ],
        gap=0.8,
    )
    return


@app.cell
def _(CHAPTERS, chapter_footer):
    chapter_footer(CHAPTERS[13], ["可逆变换提升 posterior 表达力。", "Jacobian determinant 修正局部体积变化。", "概率守恒可用积分数值验证。", "下一部分放弃一次 latent 解码，改为从噪声逐步恢复数据。"])
    return


if __name__ == "__main__":
    app.run()
