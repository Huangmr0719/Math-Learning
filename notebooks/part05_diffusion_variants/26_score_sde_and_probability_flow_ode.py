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
    from src.math_checks import ornstein_uhlenbeck_variance,simulate_ornstein_uhlenbeck
    from src.teaching import CHAPTERS,chapter_footer,chapter_header,derivation_map,exercise_block,intuition_and_rigor
    from src.visualization import COLORS,configure_matplotlib
    configure_matplotlib();import matplotlib.pyplot as plt
    return CHAPTERS,COLORS,chapter_footer,chapter_header,derivation_map,exercise_block,intuition_and_rigor,mo,np,ornstein_uhlenbeck_variance,plt,simulate_ornstein_uhlenbeck


@app.cell
def _(CHAPTERS,chapter_header):
    chapter_header(CHAPTERS[26],duration="110–150 分钟")
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 1. 本章为什么存在

    DDPM 使用离散 t。把步长缩小到连续极限，可用 stochastic differential equation：
    \[
    dX_t=f(X_t,t)dt+g(t)dW_t.
    \]
    更惊人的是，存在一个确定性 probability flow ODE，拥有相同时间边缘分布。

    ## 2. 你已经知道什么

    - 第 17 章：score 是 \(\nabla_x\log p_t(x)\)。
    - 第 25 章：ODE 给每个状态确定速度。
    - Gaussian 增量的方差会相加，标准差则不能直接相加。
    - **回忆：** 随机轨迹不同，为什么群体分布仍可能相同？
    """)
    return


@app.cell
def _(intuition_and_rigor):
    intuition_and_rigor(
        "SDE 像人群既随风移动又随机散步；ODE 像给每个人安排确定路线。个人轨迹不同，但如果速度场设计正确，每一时刻的人群密度可以完全相同。",
        r"""
        对状态无关的扩散系数 \(g(t)\)，forward SDE 为
        \(dX_t=f(X_t,t)dt+g(t)dW_t\)。
        若仍使用原时间变量 \(t\)，reverse-time SDE 必须从 \(T\) 积分到 0，
        drift 为 \(f-g^2\nabla_x\log p_t\)。此时数值步长 \(dt<0\)。
        也可以另设反向时间 \(s=T-t\) 并使用正步长，但 drift 符号必须随变量替换重写。
        Probability flow ODE:
        \[
        \frac{dX_t}{dt}
        =f(X_t,t)-\frac12g(t)^2\nabla_x\log p_t(X_t).
        \]
        在足够光滑等正则条件下，forward SDE 与 probability flow ODE 在同一
        时间 \(t\) 共享边缘分布，但轨迹分布不同；不能说每个样本轨迹相同。
        """,
    )
    return


@app.cell
def _(derivation_map,mo):
    mo.vstack(
        [
            mo.md("## 3–6. ODE、SDE 与密度演化"),
            derivation_map(["离散 Gaussian 增量","dt→0 得 Brownian noise","Fokker–Planck 描述密度","reverse drift 加 score","将扩散项改写为密度相关 drift","得到 probability flow ODE"]),
            mo.md(r"""
            Brownian increment \(dW\) 的标准差是 \(\sqrt{dt}\)，不是 dt。
            更具体地，在一个小时间段 \(\Delta t\) 内，
            \[
            \Delta W\sim\mathcal N(0,\Delta t),\qquad
            X_{n+1}=X_n+f(X_n,t_n)\Delta t+g(t_n)\sqrt{\Delta t}\epsilon_n.
            \]
            这里 \(\epsilon_n\sim\mathcal N(0,1)\)，并且不同时间步独立。

            Fokker–Planck：
            \[
            \partial_t p=-\nabla\cdot(fp)+\frac12g^2\Delta p.
            \]
            将 \(\Delta p=\nabla\cdot(p\nabla\log p)\) 代入，可得到 probability flow drift。
            这一步要求 \(g\) 不依赖状态 \(x\)、密度足够光滑且边界行为合适；
            更一般的矩阵值、状态相关 diffusion 还会出现额外项。
            """),
        ],
        gap=0.8,
    )
    return


@app.cell
def _(mo):
    stochasticity=mo.ui.slider(0,1,value=.5,step=.05,show_value=True,label="toy diffusion g")
    particles=mo.ui.slider(100,2000,value=600,step=100,show_value=True,label="particles")
    return particles,stochasticity


@app.cell
def _(COLORS,mo,ornstein_uhlenbeck_variance,particles,plt,simulate_ornstein_uhlenbeck,stochasticity):
    _dt=.01
    _t,_means,_vars,_x=simulate_ornstein_uhlenbeck(
        particles=particles.value,
        steps=100,
        dt=_dt,
        diffusion=stochasticity.value,
        seed=7,
    )
    _theory=ornstein_uhlenbeck_variance(
        _t,
        diffusion=stochasticity.value,
    )
    _fig,_axes=plt.subplots(1,2,figsize=(9,3.8))
    _axes[0].hist(_x,bins=35,density=True,color=COLORS["data"],alpha=.6);_axes[0].set_title("SDE particle marginal")
    _axes[1].plot(_t,_vars,color=COLORS["model"],label="sample variance");_axes[1].plot(_t,_theory, "--",color=COLORS["success"],label="OU theory")
    _axes[1].legend(fontsize=8);_fig.tight_layout()
    mo.vstack([mo.md(fr"""## 7、9. Euler–Maruyama 验证

    我们验证的是
    \(dX_t=-0.5X_tdt+g\,dW_t,\ X_0=0\)。它的理论方差为
    \(g^2(1-e^{{-t}})\)。

    最终样本均值={_x.mean():.4f}，样本方差={_x.var():.4f}，
    理论方差={_theory[-1]:.4f}。

    更新式必须保留旧状态：
    `x = x + (-0.5*x)*dt + g*sqrt(dt)*epsilon`。
    若漏掉开头的 `x +`，模拟的就不再是这条 SDE；若把 `sqrt(dt)` 误写成
    `dt`，噪声方差会随步长错误消失。"""),mo.hstack([stochasticity,particles],widths="equal"),_fig])
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 8. 代码

    ```python
    # x、drift、随机噪声 shape 相同，例如 [particles] 或 [B,C,H,W]。
    # 旧状态 x 必须保留；下面对应 ΔX = fΔt + g√Δt ε。
    x = x + drift(x,t)*dt + diffusion(t)*sqrt(dt)*randn_like(x)

    # Probability flow ODE：没有随机项，但需要 score。
    velocity = drift(x,t) - 0.5*diffusion(t)**2*score_model(x,t)
    x = ode_solver_step(x, velocity, dt)
    ```

    ## 10. 错误与反例

    - 漏写更新式开头的 `x +`：每一步都近似重新生成状态，不再模拟原 SDE。
    - 把 Brownian 增量写成 dt：应为标准差 sqrt(dt)。
    - 说 ODE 与 SDE 每条轨迹相同：只有 marginals 相同。
    - 一边声称从 \(T\) 积到 0，一边仍使用正 `dt` 且不改 drift：时间约定互相冲突。

    原始资料：[Song et al., Score-Based Generative Modeling through SDEs](https://arxiv.org/abs/2011.13456)
    """)
    return


@app.cell
def _(exercise_block,mo):
    mo.vstack(
        [
            mo.md("## 11. 分层练习"),
            exercise_block(
                ("ODE 与 SDE 的路径差别？","ODE 给定初值后确定，SDE 还受 Brownian 随机增量影响。"),
                ("dt=.01，Brownian increment 标准差？","sqrt(.01)=.1。"),
                ("为何 probability flow ODE 仍需 score？","它用 score 把 SDE 的扩散效应改写为确定性密度运输 drift。"),
                ("减小 particles，观察方差曲线。","Monte Carlo 抖动增大，但理论趋势不变。"),
            ),
        ],
        gap=0.8,
    )
    return


@app.cell
def _(CHAPTERS,chapter_footer):
    chapter_footer(CHAPTERS[26],["SDE 同时包含 drift 与 Brownian noise。","Reverse dynamics 依赖 score。","Probability flow ODE 与 SDE 共享边缘分布。","既然 ODE 能运输分布，下一部分直接学习 velocity field。"])
    return


if __name__=="__main__":app.run()
