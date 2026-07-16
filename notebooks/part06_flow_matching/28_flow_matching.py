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
    from src.math_checks import conditional_squared_error
    from src.teaching import CHAPTERS,chapter_footer,chapter_header,derivation_map,exercise_block,intuition_and_rigor
    from src.visualization import COLORS,configure_matplotlib
    configure_matplotlib();import matplotlib.pyplot as plt
    return CHAPTERS,COLORS,chapter_footer,chapter_header,conditional_squared_error,derivation_map,exercise_block,intuition_and_rigor,mo,np,plt


@app.cell
def _(CHAPTERS,chapter_header):
    chapter_header(CHAPTERS[28],duration="105–145 分钟")
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 1. 本章为什么存在

    CNF 最大似然训练需要计算 divergence。Flow Matching 选择一族已知的
    **条件概率路径**（conditional probability path），直接回归每条条件路径的速度：
    \[
    L_{\rm CFM}=E_{t,x_0,x_1}\|v_\theta(x_t,t)-u_t(x_t|x_0,x_1)\|^2.
    \]

    ## 2. 你已经知道什么

    - 第 27 章：velocity field 决定分布运输。
    - 第 4 章：期望是按概率加权的平均数。
    - 第 1 章：MSE 衡量预测与目标的平方差。
    - **回忆：** 同一位置可能由多对起点终点经过，模型应输出哪一个速度？

    > **符号切换提醒：** DDPM 中 \(x_0\) 一直表示干净数据；本部分采用
    > Flow Matching 常见记号，\(x_0\) 表示 source/base noise，
    > \(x_1\) 表示 target/data。二者不要混用。
    """)
    return


@app.cell
def _(intuition_and_rigor):
    intuition_and_rigor(
        "老师随机给出许多“起点—终点”路线，在路线中间标注瞬时速度。学生只看到当前位置和时间；当多条路线在此交会时，MSE 会让学生学到这些速度的平均。",
        r"""
        对平方损失，固定 \(X_t=x\) 后最优函数为
        \[
        v^*(x,t)=E[U_t|X_t=x].
        \]
        右侧称为**条件期望**：只在已经知道 \(X_t=x\) 的那些可能路线中，
        对目标速度 \(U_t\) 做概率加权平均。满足可积性与适当正则条件时，
        这个条件平均正是产生边缘概率路径的 velocity field。
        数值训练只能验证例子；严格结论来自下面的平方误差正交分解。
        """,
    )
    return


@app.cell
def _(derivation_map,mo):
    mo.md("## 3–6. Conditional path 如何产生 marginal field")
    derivation_map(["采样 source x0","采样 target x1","采样时间 t","构造 conditional xt","计算已知 conditional velocity","MSE 回归","条件期望得到 marginal velocity"])
    mo.md(r"""
    最简单线性插值：
    \[
    x_t=(1-t)x_0+tx_1,\qquad u_t=x_1-x_0.
    \]
    这里单条样本速度恒定，但模型看到的 \(v(x,t)\) 是给定位置后的条件平均。
    shape：x0、x1、xt、u、prediction 均 `[B,D]`；t reshape 为 `[B,1]` 广播。

    ### 数学暂停站：为什么 MSE 会学到条件平均？

    先固定时间 \(t\) 和位置 \(X_t=x\)。令
    \(\mu(x,t)=E[U_t\mid X_t=x]\)，模型在该位置输出任意候选值 \(a\)。
    使用恒等式

    \[
    U_t-a=(U_t-\mu)+(\mu-a)
    \]

    展开平方并取条件期望：

    \[
    \begin{aligned}
    E[(U_t-a)^2\mid X_t=x]
    &=E[(U_t-\mu)^2\mid X_t=x]\\
    &\quad+2(\mu-a)E[U_t-\mu\mid X_t=x]\\
    &\quad+(\mu-a)^2\\
    &=\operatorname{Var}(U_t\mid X_t=x)+(\mu-a)^2.
    \end{aligned}
    \]

    中间交叉项为 0，因为条件平均的定义给出
    \(E[U_t-\mu\mid X_t=x]=0\)。第一项与预测 \(a\) 无关，第二项只在
    \(a=\mu\) 时达到最小值 0。因此 MSE 的最优预测就是条件期望。

    向量情形把平方换成 \(\|U_t-a\|^2\)，结论相同；要求
    \(E[\|U_t\|^2]<\infty\)，否则 MSE 本身可能没有有限值。
    """)
    return


@app.cell
def _(mo):
    velocity_guess=mo.ui.slider(-3,5,value=0,step=.1,show_value=True,label="候选速度 a")
    return (velocity_guess,)


@app.cell
def _(COLORS,conditional_squared_error,mo,np,plt,velocity_guess):
    _targets=np.array([-2.0,4.0])
    _probabilities=np.array([.25,.75])
    _conditional_mean=float(np.sum(_targets*_probabilities))
    _grid=np.linspace(-3,5,200)
    _losses=np.array([
        conditional_squared_error(_targets,_probabilities,_value)
        for _value in _grid
    ])
    _current=conditional_squared_error(
        _targets,_probabilities,velocity_guess.value
    )
    _fig,_ax=plt.subplots(figsize=(7,3.8))
    _ax.plot(_grid,_losses,color=COLORS["model"])
    _ax.axvline(_conditional_mean,color=COLORS["success"],linestyle="--",label="conditional mean")
    _ax.scatter([velocity_guess.value],[_current],color=COLORS["danger"],zorder=3,label="current prediction")
    _ax.set_xlabel("prediction a");_ax.set_ylabel("expected squared error");_ax.legend(fontsize=8)
    _fig.tight_layout()
    mo.vstack([mo.md(fr"""### 条件期望的手算与交互验证

    假设在同一个 \((x,t)\) 处，目标速度有两种可能：
    \(U=-2\) 的条件概率为 0.25，\(U=4\) 的条件概率为 0.75。

    条件平均为
    \[
    \mu=0.25\times(-2)+0.75\times4={_conditional_mean:.1f}.
    \]

    当前预测 \(a={velocity_guess.value:.1f}\)，期望平方误差为
    {_current:.3f}。拖动预测值，最低点始终位于条件平均；这是解析结论的
    数值展示，不是对一般定理的证明。"""),velocity_guess,_fig])
    return


@app.cell
def _(mo):
    flow_t=mo.ui.slider(0,1,value=.5,step=.05,show_value=True,label="time t")
    sample_pair=mo.ui.slider(0,7,value=0,step=1,show_value=True,label="pair index")
    return flow_t,sample_pair


@app.cell
def _(COLORS,flow_t,mo,np,plt,sample_pair):
    _x0=np.array([[-2,-1],[-2,1],[-1.5,0],[-2,.5],[-1,-1.5],[-1,1.5],[-2,0],[-1.5,.8]],float)
    _x1=np.array([[2,1],[2,-1],[1.5,0],[2,-.5],[1,1.5],[1,-1.5],[2,0],[1.5,-.8]],float)
    _xt=(1-flow_t.value)*_x0+flow_t.value*_x1;_u=_x1-_x0;_i=sample_pair.value
    _fig,_axes=plt.subplots(1,2,figsize=(9,3.8))
    for _j in range(len(_x0)):_axes[0].plot([_x0[_j,0],_x1[_j,0]],[_x0[_j,1],_x1[_j,1]],color=COLORS["muted"],alpha=.4)
    _axes[0].scatter(_xt[:,0],_xt[:,1],color=COLORS["data"]);_axes[0].quiver(_xt[:,0],_xt[:,1],_u[:,0],_u[:,1],angles="xy",scale_units="xy",scale=5,color=COLORS["model"])
    _axes[0].set_aspect("equal")
    _axes[1].plot(np.linspace(0,1,20),np.tile(np.linalg.norm(_u[_i]),20),color=COLORS["prior"]);_axes[1].set_title(f"pair {_i} speed magnitude")
    _fig.tight_layout()
    mo.vstack([mo.md(fr"""## 7、9. 条件路径动画帧

    pair {_i}: \(x_t={_xt[_i]}\)，conditional velocity={_u[_i]}。
    拖动 t 观察粒子沿直线移动；速度不随 t 变是线性 path 的特性。"""),mo.hstack([flow_t,sample_pair],widths="equal"),_fig])
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 8. 训练代码

    ```python
    x0 = sample_source(batch_size)       # [B,D]
    x1 = sample_data(batch_size)         # [B,D]
    t = torch.rand(batch_size, 1)        # [B,1]
    xt = (1-t)*x0 + t*x1
    target_velocity = x1 - x0
    prediction = model(xt, t)
    loss = F.mse_loss(prediction, target_velocity)
    ```

    ## 10. 错误与反例

    - 把样本速度 \(x_1-x_0\) 与 marginal velocity 当作处处相同。
    - 把 DDPM 的 \(x_0=\) 数据记号直接搬来：本章 \(x_0\) 是 source noise。
    - 忘记模型只看 xt,t，训练时偷偷输入 x1 会改变生成问题。
    - 将数值 MSE 下降称为 continuity equation 的证明。

    原始资料：[Lipman et al., Flow Matching for Generative Modeling](https://arxiv.org/abs/2210.02747)
    """)
    return


@app.cell
def _(exercise_block,mo):
    mo.md("## 11. 分层练习")
    exercise_block(
        ("为什么最优模型是条件平均速度？","固定 xt,t 后，期望平方误差可分解为不可约条件方差，加上预测与条件平均之差的平方；后者在预测等于条件平均时为 0。"),
        ("x0=-2、x1=3、t=.4，xt 和速度？","xt=0，速度=5。"),
        ("t 为什么 reshape 为 [B,1]？","让每个样本的标量时间沿 D 个 feature 广播。"),
        ("改变 pair 与 t，比较位置和速度。","线性 path 中位置随 t 变，单对样本速度保持不变。"),
    )
    return


@app.cell
def _(CHAPTERS,chapter_footer):
    chapter_footer(CHAPTERS[28],["Flow Matching 直接回归 conditional velocity。","MSE 最优解是给定 xt,t 的条件期望。","线性 path 的单样本速度为 x1-x0。","下一章研究 pairing/coupling 如何改变路径难度。"])
    return


if __name__=="__main__":app.run()
