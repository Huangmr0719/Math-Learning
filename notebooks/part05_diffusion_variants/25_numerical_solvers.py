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
    chapter_header(CHAPTERS[25],duration="95–125 分钟")
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 本章为什么存在

    一次 denoiser 调用很昂贵。若生成轨迹可看成 ODE
    \(dx/dt=v(x,t)\)，采样就是数值积分问题：怎样用更少步仍跟准曲线？

    ## 你已经知道什么

    - 导数是局部斜率，ODE 指定每个位置的运动速度。
    - DDIM 确定性采样已经像离散轨迹。
    - **回忆：** 用当前斜率走很大一步为什么容易偏离弯曲轨迹？
    """)
    return


@app.cell
def _(intuition_and_rigor):
    intuition_and_rigor(
        "Euler 像只看脚下道路方向就直走一步；Heun 先试走到终点，再把起点和试走终点的方向平均，能更早发现道路正在转弯。",
        r"""
        Euler:
        \(x_{n+1}=x_n+h f(x_n,t_n)\)，全局误差通常 \(O(h)\)。
        Heun:
        \(k_1=f(x_n,t_n)\)，
        \(k_2=f(x_n+hk_1,t_n+h)\)，
        \(x_{n+1}=x_n+\frac h2(k_1+k_2)\)，全局误差通常 \(O(h^2)\)。
        阶数结论要求足够光滑与稳定条件。
        """,
    )
    return


@app.cell
def _(derivation_map,mo):
    mo.vstack(
        [
            mo.md("## 从面积到数值积分"),
            derivation_map(["ODE 给速度","小时间内位移≈速度×时间","Euler 用左端速度","Heun 预测终点速度","两端平均","比较解析解误差"]),
            mo.md(r"""
            测试方程 \(x'=x,x(0)=1\) 的解析解 \(e^t\)。
            状态 x 可为 `[B,C,H,W]`；时间 h 是标量并广播。反向生成时 h 可能为负，
            不能擅自取绝对值，否则时间方向翻转。

            ### 局部误差和全局误差不是一回事

            对足够光滑的真实解做 Taylor 展开：

            \[
            x(t+h)=x(t)+hx'(t)+\frac{h^2}{2}x''(\xi).
            \]

            Euler 只保留前两项，所以“假设起点完全正确，单走一步”的局部截断误差
            是 \(O(h^2)\)。但从 0 走到固定终点大约需要 \(1/h\) 步，前面每步的误差
            还会传播，因此稳定条件下累计成 \(O(h)\) 的全局误差。

            Heun 的局部截断误差通常是 \(O(h^3)\)，稳定累计后全局误差为 \(O(h^2)\)。
            “二阶”通常指全局误差阶，而不是每一步只错 \(h^2\)。

            公平比较还要使用 NFE（number of function evaluations，网络调用次数）：
            Euler 每步 1 次，Heun 每步 2 次。10 步 Heun 与 20 步 Euler 都约为 20 NFE。
            """),
        ],
        gap=0.8,
    )
    return


@app.cell
def _(mo):
    solver_steps=mo.ui.slider(2,50,value=8,step=1,show_value=True,label="solver steps")
    return (solver_steps,)


@app.cell
def _(COLORS,mo,np,plt,solver_steps):
    _h=1/solver_steps.value;_e=[1.0];_heun=[1.0]
    for _ in range(solver_steps.value):
        _e.append(_e[-1]+_h*_e[-1])
        _k1=_heun[-1];_k2=_heun[-1]+_h*_k1
        _heun.append(_heun[-1]+_h*.5*(_k1+_k2))
    _t=np.linspace(0,1,solver_steps.value+1);_exact=np.exp(_t)
    _err_e=abs(_e[-1]-np.e);_err_h=abs(_heun[-1]-np.e)
    _fig,_axes=plt.subplots(1,2,figsize=(9,3.8))
    _axes[0].plot(_t,_exact,label="exact",color=COLORS["success"])
    _axes[0].plot(_t,_e,"o-",label="Euler",color=COLORS["data"])
    _axes[0].plot(_t,_heun,"s-",label="Heun",color=COLORS["model"]);_axes[0].legend(fontsize=8)
    _ns=np.arange(2,51);_eerrs=np.abs((1+1/_ns)**_ns-np.e)
    _hvals=1/_ns;_hsol=(1+_hvals+_hvals**2/2)**_ns;_herrs=np.abs(_hsol-np.e)
    _axes[1].loglog(_ns,_eerrs,label="Euler",color=COLORS["data"]);_axes[1].loglog(_ns,_herrs,label="Heun",color=COLORS["model"]);_axes[1].legend(fontsize=8)
    _fig.tight_layout()
    mo.vstack([mo.md(fr"""## 解析轨迹对比

    Euler final error={_err_e:.3e}；Heun final error={_err_h:.3e}。
    这是平滑 toy ODE 的证据，不保证真实 learned vector field 同样稳定。"""),solver_steps,_fig])
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 逐行代码

    ```python
    # Euler：一次模型调用。
    velocity = model(x, t)
    x_next = x + step_size * velocity

    # Heun：两次调用，用起点和预测终点速度平均。
    k1 = model(x, t)
    x_predict = x + step_size * k1
    k2 = model(x_predict, t + step_size)
    x_next = x + 0.5 * step_size * (k1 + k2)
    ```

    ## 错误与反例

    - 高阶必然更快：Heun 每步两次网络调用，应按 NFE 比较。
    - 把局部 \(O(h^2)\) 误说成 Euler 的全局误差阶。
    - 忽略反向时间的负步长。
    - 只在一个步数比较，无法观察收敛阶。

    参考：[Lu et al., DPM-Solver](https://arxiv.org/abs/2206.00927)
    """)
    return


@app.cell
def _(exercise_block,mo):
    mo.vstack(
        [
            mo.md("## 分层练习"),
            exercise_block(
                ("Euler 的核心近似？","在一个小时间段内，把速度当作起点速度保持不变。"),
                ("x=2、h=.1、f=x，Euler 下一步？","2+0.1×2=2.2。"),
                ("Euler 跑 20 步、每步 1 次模型调用；Heun 跑 10 步、每步 2 次。写出两者 `NFE`，并说明为什么这样比较更公平。","两者都是 `NFE=20`。不同 solver 每步调用模型次数不同，因此只比较步数会掩盖真实计算成本；固定 NFE 后再比较误差，才能更接近等预算比较。"),
                ("减少 steps，比较两种误差。","Heun 通常在同一步数下更准，但需两倍调用。"),
            ),
        ],
        gap=0.8,
    )
    return


@app.cell
def _(CHAPTERS,chapter_footer):
    chapter_footer(CHAPTERS[25],["采样可视为数值积分。","Euler 一阶，Heun 用预测校正达到二阶。","质量应结合误差和 NFE 比较。","下一章把 diffusion 写成连续时间 SDE/ODE。"])
    return


if __name__=="__main__":app.run()
