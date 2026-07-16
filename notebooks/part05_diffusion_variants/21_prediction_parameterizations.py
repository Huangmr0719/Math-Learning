import marimo

__generated_with = "0.23.13"
app = marimo.App(width="medium")


@app.cell
def _():
    import sys as _sys
    from pathlib import Path as _Path
    _root=_Path(__file__).resolve().parents[2]
    if str(_root) not in _sys.path:_sys.path.insert(0,str(_root))
    import marimo as mo
    import numpy as np
    from src.teaching import CHAPTERS,chapter_footer,chapter_header,derivation_map,exercise_block,intuition_and_rigor
    from src.visualization import COLORS,configure_matplotlib
    configure_matplotlib(); import matplotlib.pyplot as plt
    return CHAPTERS,COLORS,chapter_footer,chapter_header,derivation_map,exercise_block,intuition_and_rigor,mo,np,plt


@app.cell
def _(CHAPTERS,chapter_header):
    chapter_header(CHAPTERS[21],duration="85–115 分钟")
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 1. 本章为什么存在

    同一个 noisy state 可由信号和噪声描述。模型可以预测 \(\epsilon\)、\(x_0\)、
    score，或 velocity \(v\)。它们不是四个无关目标，而是同一二维线性坐标系的不同基。

    ## 2. 你已经知道什么

    \[
    x_t=\alpha_t x_0+\sigma_t\epsilon,\qquad \alpha_t^2+\sigma_t^2=1.
    \]
    本章用 \(\alpha_t=\sqrt{\bar\alpha_t}\)、\(\sigma_t=\sqrt{1-\bar\alpha_t}\)；
    注意这与 DDPM 单步 alpha 记号不同。
    """)
    return


@app.cell
def _(intuition_and_rigor):
    intuition_and_rigor(
        "已知一杯饮料由果汁和水混合，可以预测加了多少水，也可以预测原来有多少果汁；两者通过混合比例换算。",
        r"""
        定义 \(v_t=\alpha_t\epsilon-\sigma_t x_0\)。则
        \[
        \begin{bmatrix}x_t\\v_t\end{bmatrix}
        =
        \begin{bmatrix}\alpha_t&\sigma_t\\-\sigma_t&\alpha_t\end{bmatrix}
        \begin{bmatrix}x_0\\\epsilon\end{bmatrix}.
        \]
        该矩阵是旋转矩阵，逆等于转置：
        \(x_0=\alpha_tx_t-\sigma_tv_t\)，
        \(\epsilon=\sigma_tx_t+\alpha_tv_t\)。
        """,
    )
    return


@app.cell
def _(derivation_map,mo):
    mo.vstack(
        [
            mo.md("## 3–6. 参数化换算与 SNR"),
            derivation_map(["写 xt 混合式","定义 v","组成 2×2 旋转矩阵","转置求逆","得到 x0 与 epsilon","score 再做时间缩放"]),
            mo.md(r"""
            \[
            \mathrm{SNR}(t)=\alpha_t^2/\sigma_t^2,\qquad
            score=-\epsilon/\sigma_t
            \]
            （这里是已知 \(x_0\) 时 \(q(x_t|x_0)\) 的 conditional Gaussian score）。
            真正生成所需的 marginal score 是
            \(\nabla_{x_t}\log p_t(x_t)\)；训练数据平均后，最优噪声预测与它建立联系，
            不能把单个样本的真实 epsilon 直接称为 marginal score。

            SNR 是信号方差与噪声方差之比：

            - \(\mathrm{SNR}\gg1\)：接近干净端，信号占主导；
            - \(\mathrm{SNR}\ll1\)：接近噪声端，噪声占主导；
            - \(\log\mathrm{SNR}=2\log\alpha_t-2\log\sigma_t\) 把跨越许多数量级的
              比值变成较易观察的加减尺度。

            当 \(\sigma_t\to0\) 时 score 换算含除零风险；
            当 \(\alpha_t\to0\) 时从 epsilon 恢复 x0 会放大误差。参数化影响数值条件和 loss 权重。

            上面的 2×2 矩阵只有在 \(\alpha_t^2+\sigma_t^2=1\) 的
            variance-preserving 参数化下才是旋转矩阵；更一般噪声路径仍可线性换算，
            但逆矩阵不一定等于转置。
            """),
        ],
        gap=0.8,
    )
    return


@app.cell
def _(mo):
    angle=mo.ui.slider(0.05,1.52,value=0.7,step=0.05,show_value=True,label="noise angle")
    v_prediction=mo.ui.slider(-2,2,value=0.3,step=0.1,show_value=True,label="v prediction")
    return angle,v_prediction


@app.cell
def _(COLORS,angle,mo,np,plt,v_prediction):
    _a=np.cos(angle.value); _s=np.sin(angle.value); _xt=1.0; _v=v_prediction.value
    _x0=_a*_xt-_s*_v; _eps=_s*_xt+_a*_v
    _xt_check=_a*_x0+_s*_eps; _v_check=_a*_eps-_s*_x0
    _fig,_ax=plt.subplots(figsize=(7,4))
    _ax.quiver([0,0,0],[0,0,0],[_xt,_x0,_eps],[0,_v,0],angles="xy",scale_units="xy",scale=1,color=[COLORS["data"],COLORS["model"],COLORS["prior"]])
    _ax.set_xlim(-2.5,2.5);_ax.set_ylim(-2.5,2.5);_ax.set_aspect("equal")
    _ax.set_title("线性坐标换算")
    mo.vstack([mo.md(fr"""## 7、9. 等价公式随机点检查

    恢复 \(x_0={_x0:.4f}\)、\(\epsilon={_eps:.4f}\)；
    重构误差 max=`{max(abs(_xt_check-_xt),abs(_v_check-_v)):.2e}`。"""),mo.hstack([angle,v_prediction],widths="equal"),_fig])
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 8. 代码

    ```python
    # alpha、sigma 已 reshape 到能与 xt 广播的 shape。
    x0_hat = alpha * xt - sigma * v_hat
    epsilon_hat = sigma * xt + alpha * v_hat
    score_hat = -epsilon_hat / sigma.clamp_min(1e-5)
    ```

    ## 10. 错误与反例

    - 把 \(\alpha_t\) 当第 15 章单步 alpha：必须先声明记号。
    - score 与 epsilon 不加缩放直接互换。
    - 把 conditional score \(-\epsilon/\sigma_t\) 与 marginal score 不加说明地混为一谈。
    - 接近边界时间仍直接除以极小 alpha/sigma。

    参考：[Salimans & Ho, Progressive Distillation](https://arxiv.org/abs/2202.00512)
    """)
    return


@app.cell
def _(exercise_block,mo):
    mo.vstack(
        [
            mo.md("## 11. 分层练习"),
            exercise_block(
                ("为什么称为不同参数化？","它们表达同一 noisy state 中的信号与噪声，但选择不同预测坐标和损失尺度。"),
                ("alpha=.8、sigma=.6、xt=1、v=0，求 x0 与 epsilon。","x0=.8，epsilon=.6。"),
                ("为什么 clamp sigma？","防止接近干净端时除以接近 0 导致数值爆炸。"),
                ("拖动 angle，观察同一 xt、v 的换算。","时间改变坐标旋转角，因此同一数值预测对应不同 x0 与 epsilon。"),
            ),
        ],
        gap=0.8,
    )
    return


@app.cell
def _(CHAPTERS,chapter_footer):
    chapter_footer(CHAPTERS[21],["epsilon、x0、score、v 可相互换算。","SNR 描述信号与噪声比例。","不同参数化改变数值条件和训练权重。","下一章把条件信息用于 guidance。"])
    return


if __name__=="__main__":app.run()
