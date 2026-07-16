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
    configure_matplotlib();import matplotlib.pyplot as plt
    return CHAPTERS,COLORS,chapter_footer,chapter_header,derivation_map,exercise_block,intuition_and_rigor,mo,np,plt


@app.cell
def _(CHAPTERS,chapter_header):
    chapter_header(CHAPTERS[27],duration="105–140 分钟")
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 1. 本章为什么存在

    第 26 章说明 ODE 可以运输概率分布。Continuous Normalizing Flow 直接定义
    \[
    \frac{dx_t}{dt}=v_\theta(x_t,t),
    \]
    从简单 base distribution 流向数据分布，并同时追踪密度变化。

    ## 2. 你已经知道什么

    - 第 13 章：离散可逆变换用 log determinant 修正密度。
    - 第 25 章：ODE solver 沿速度场推进状态。
    - **回忆：** 流体局部向外发散时，密度应上升还是下降？
    """)
    return


@app.cell
def _(intuition_and_rigor):
    intuition_and_rigor(
        "把概率看成可压缩的人群。速度箭头向外散开，人群变稀；箭头向内汇聚，人群变密。divergence 就是局部“散开还是汇聚”的刻度。",
        r"""
        Flow map \(\phi_t(x_0)\) 是 ODE 从初值到时刻 t 的解。
        密度满足 continuity equation：
        \[
        \partial_t p_t+\nabla\cdot(p_tv_t)=0.
        \]
        沿单条轨迹：
        \[
        \frac d{dt}\log p_t(x_t)=-\nabla\cdot v_t(x_t).
        \]
        多维 divergence 是 Jacobian trace，而非 determinant。
        """,
    )
    return


@app.cell
def _(derivation_map,mo):
    mo.md("## 3–6. 从概率守恒到瞬时变量替换")
    derivation_map(["小体积内概率守恒","流入减流出得到 continuity equation","沿粒子轨迹用链式法则","展开 divergence product","抵消密度梯度项","得到 d log p/dt=-div v"])
    mo.md(r"""
    \[
    \nabla\cdot v=\sum_i\partial v_i/\partial x_i.
    \]
    对 batch 状态 `[B,D]`，每个样本得到一个 divergence 标量 `[B]`。
    CNF 最大似然需同时积分状态与 log density，计算 divergence 可能昂贵。

    ### 从一维概率守恒开始

    先看一条直线。区间 \([a,b]\) 内的概率为
    \(\int_a^b p_t(x)dx\)。它的变化等于左边流入减去右边流出：

    \[
    \frac d{dt}\int_a^b p_t(x)dx
    =p_t(a)v_t(a)-p_t(b)v_t(b)
    =-\int_a^b\partial_x(p_tv_t)dx.
    \]

    因为这对任意区间都成立，所以
    \[
    \partial_t p_t+\partial_x(p_tv_t)=0.
    \]
    多维时把一维导数换成 divergence，得到 continuity equation。

    沿满足 \(\dot x_t=v_t(x_t)\) 的粒子轨迹，用链式法则：

    \[
    \frac d{dt}\log p_t(x_t)
    =\frac{\partial_t p_t}{p_t}
     +\nabla\log p_t\cdot v_t.
    \]

    再展开
    \(\nabla\cdot(p_tv_t)=v_t\cdot\nabla p_t+p_t\nabla\cdot v_t\)，
    代入 continuity equation 后，梯度项抵消，得到
    \[
    \frac d{dt}\log p_t(x_t)=-\nabla\cdot v_t(x_t).
    \]

    该推导要求密度和向量场足够光滑，并且 ODE 在所考虑区间内存在唯一解。
    生成通常从 \(p_0=\) base noise 积分到 \(p_1=\) data；计算数据 likelihood
    时则常从数据反向积分回可计算密度的 base，并同步累计 log-density change。
    """)
    return


@app.cell
def _(mo):
    expansion=mo.ui.slider(-1.5,1.5,value=.5,step=.1,show_value=True,label="linear field a in v(x)=a x")
    time=mo.ui.slider(0,1,value=.7,step=.05,show_value=True,label="time t")
    return expansion,time


@app.cell
def _(COLORS,expansion,mo,np,plt,time):
    _rng=np.random.default_rng(7);_x0=_rng.standard_normal(1000);_xt=np.exp(expansion.value*time.value)*_x0
    _theory_var=np.exp(2*expansion.value*time.value)
    _log_density_change=-expansion.value*time.value
    _fig,_axes=plt.subplots(1,2,figsize=(9,3.8))
    _axes[0].hist(_x0,bins=35,density=True,alpha=.4,label="p0",color=COLORS["data"])
    _axes[0].hist(_xt,bins=35,density=True,alpha=.4,label="pt",color=COLORS["model"]);_axes[0].legend(fontsize=8)
    _grid=np.linspace(-2,2,20);_axes[1].quiver(_grid,np.zeros_like(_grid),expansion.value*_grid,np.zeros_like(_grid),angles="xy",scale_units="xy",scale=1,color=COLORS["prior"])
    _axes[1].set_ylim(-.5,.5);_axes[1].set_title("v(x)=a x")
    _fig.tight_layout()
    mo.vstack([mo.md(fr"""## 7、9. 解析 ODE 验证

    \(x_t=e^{{at}}x_0\)。理论 variance={_theory_var:.4f}，
    样本 variance={_xt.var():.4f}；单维 log-density change={_log_density_change:.4f}。"""),mo.hstack([expansion,time],widths="equal"),_fig])
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 8. 增广 ODE 代码

    ```python
    def augmented_dynamics(t, state):
        x, log_p = state
        velocity = vector_field(x, t)             # [B,D]
        divergence = jacobian_trace(velocity, x)  # [B]
        d_log_p = -divergence
        return velocity, d_log_p
    ```

    ## 10. 错误与反例

    - 用 determinant 代替 divergence：CNF 是瞬时 trace。
    - divergence 正时密度增加：符号相反，向外散开使 log density 降低。
    - ODE 有唯一解不检查 Lipschitz/数值条件。

    原始资料：[Chen et al., Neural Ordinary Differential Equations](https://arxiv.org/abs/1806.07366)
    """)
    return


@app.cell
def _(exercise_block,mo):
    mo.md("## 11. 分层练习")
    exercise_block(
        ("divergence 正代表什么？","局部体积向外膨胀，因此沿轨迹的 log density 下降。"),
        ("二维 v=ax，divergence？","2a，因为两个对角偏导各为 a。"),
        ("为什么 log_p shape 是 [B]？","每个 batch 样本对应一个标量密度值。"),
        ("把 a 调为负，观察分布。","粒子汇聚、方差下降、密度升高。"),
    )
    return


@app.cell
def _(CHAPTERS,chapter_footer):
    chapter_footer(CHAPTERS[27],["CNF 用 ODE flow map 运输分布。","Continuity equation 表达概率守恒。","log density 沿轨迹由负 divergence 决定。","下一章绕开昂贵密度追踪，直接监督 velocity。"])
    return


if __name__=="__main__":app.run()
