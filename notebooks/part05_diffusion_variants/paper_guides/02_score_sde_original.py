import marimo

__generated_with = "0.23.13"
app = marimo.App(width="medium")


@app.cell
def _():
    import sys as _sys
    from pathlib import Path as _Path

    _root = _Path(__file__).resolve().parents[3]
    if str(_root) not in _sys.path:
        _sys.path.insert(0, str(_root))

    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    from src.teaching import (
        PAPERS,
        PAPER_GUIDES,
        derivation_map,
        exercise_block,
        paper_evidence_block,
        paper_guide_footer,
        paper_guide_header,
    )
    from src.visualization import COLORS, configure_matplotlib

    configure_matplotlib()
    return (
        COLORS,
        PAPERS,
        PAPER_GUIDES,
        derivation_map,
        exercise_block,
        mo,
        np,
        paper_evidence_block,
        paper_guide_footer,
        paper_guide_header,
        plt,
    )


@app.cell
def _(PAPER_GUIDES, paper_guide_header):
    GUIDE = PAPER_GUIDES["score_sde_original"]
    paper_guide_header(GUIDE, duration="120–180 分钟，可分两次完成")
    return (GUIDE,)


@app.cell
def _(mo):
    mo.callout(
        mo.md(r"""
        ## 先解决一个语言陷阱｜“等价 ODE”到底等价什么

        论文说 probability flow ODE 与 SDE 有相同的时间边缘分布。它没有说：

        - 同一个初值会走出同一条路径；
        - ODE 路径等于 SDE 路径的平均；
        - 任意近似 score 和任意 solver 都能保持精确等价。

        本导读的核心任务，是从密度演化方程证明“marginals 相同”的准确含义，
        再用一维可解模型看到：粒子轨迹明显不同，直方图却可以一致。
        """),
        kind="warn",
    )
    return


@app.cell
def _(PAPERS, paper_evidence_block):
    paper_evidence_block(
        PAPERS["score_sde"],
        title="一张图统一 forward SDE、reverse SDE 与 probability flow ODE",
        original_quote="We also derive an equivalent neural ODE that samples from the same distribution as the SDE.",
        chinese_translation="作者构造一个确定性神经 ODE，使它与 SDE 在每个时刻拥有相同的边缘分布。",
        figure_path="assets/paper_figures/diffusion_variants/score_sde_figure2_overview.webp",
        figure_number="原文 Figure 2",
        figure_caption="左半从数据经 forward SDE 到 prior；右半从 prior 经 reverse SDE 或 probability flow ODE 回到数据。",
        legend=(
            "横轴从左到中：时间 0→T，数据密度逐渐扩散成 prior；从中到右：生成时反向积分 T→0。",
            "彩色锯齿轨迹：SDE 粒子，包含 Brownian 随机性。",
            "白色平滑轨迹：probability flow ODE 粒子，给定初值后确定。",
            "背景亮度与两侧密度曲线：群体边缘密度；相同背景不表示彩色线与白线逐条重合。",
            "上方公式中的 d w-bar 表示反向时间 Wiener process；时间方向必须结合积分区间理解。",
        ),
        learning_goal="用自己的话区分“路径随机性”和“群体密度”，并沿图指出 score 出现在两条反向动力学中的位置。",
        evidence_boundary="Figure 2 是统一框架示意；严格的 marginal 等价来自 Fokker–Planck/continuity equation 推导，不是由画面相似证明。",
    )
    return


@app.cell
def _(derivation_map, mo):
    mo.vstack(
        [
            mo.md("## 三条动力学｜先固定时间约定"),
            derivation_map(
                [
                    "写 forward Itô SDE",
                    "定义时刻 t 的边缘密度 p_t",
                    "用 score ∇ log p_t 写 reverse-time drift",
                    "明确 reverse SDE 从 T 积分到 0",
                    "写 probability flow ODE",
                    "比较两者的密度演化而非单条轨迹",
                ]
            ),
            mo.md(r"""
            对状态无关扩散系数 \(g(t)\) 的 forward SDE：
            \[
            dX_t=f(X_t,t)\,dt+g(t)\,dW_t.
            \]

            使用同一个时间变量 \(t\)，reverse-time SDE 写成
            \[
            dX_t=
            [f(X_t,t)-g(t)^2\nabla_x\log p_t(X_t)]\,dt
            +g(t)\,d\bar W_t,
            \]
            但积分方向是 \(t:T\to0\)，所以数值实现中的 \(dt<0\)。

            Probability flow ODE 为
            \[
            \frac{dX_t}{dt}
            =f(X_t,t)-\frac12g(t)^2\nabla_x\log p_t(X_t).
            \]

            reverse SDE 的 score 系数是 \(1\)，ODE 是 \(1/2\)。少掉的一半不是近似：
            ODE 没有扩散项，因此必须把恰好一半的 score 漂移放入确定速度，才能复制
            SDE 的密度演化。
            """),
        ],
        gap=0.8,
    )
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 严格推导｜为什么 probability flow ODE 共享 marginals

    以下推导假设密度足够光滑、\(g(t)\) 不依赖状态，且边界项满足所需衰减条件。

    Forward SDE 的 Fokker–Planck equation 是
    \[
    \partial_t p_t
    =-\nabla\cdot(f p_t)+\frac12g(t)^2\Delta p_t.
    \]

    ODE 的速度场取
    \[
    v(x,t)=f(x,t)-\frac12g(t)^2\nabla\log p_t(x).
    \]

    确定性流的 continuity equation 为
    \[
    \partial_t p_t=-\nabla\cdot(vp_t).
    \]

    代入 \(v\)，并使用
    \(p_t\nabla\log p_t=\nabla p_t\)：
    \[
    \begin{aligned}
    -\nabla\cdot(vp_t)
    &=-\nabla\cdot(f p_t)
      +\frac12g(t)^2\nabla\cdot(p_t\nabla\log p_t)\\
    &=-\nabla\cdot(f p_t)+\frac12g(t)^2\Delta p_t.
    \end{aligned}
    \]

    右侧与 SDE 的 Fokker–Planck equation 相同。若两者初始密度相同且该密度演化问题
    解唯一，则每个时刻的边缘密度相同。证明对象是 \(p_t\)，不是粒子路径。
    """)
    return


@app.cell
def _(mo):
    toy_time = mo.ui.slider(0.0, 1.0, value=0.7, step=0.05, show_value=True, label="观察时间 t")
    toy_diffusion = mo.ui.slider(0.2, 1.4, value=0.8, step=0.1, show_value=True, label="扩散强度 g")
    toy_seed = mo.ui.slider(0, 30, value=7, step=1, show_value=True, label="随机 seed")
    return toy_diffusion, toy_seed, toy_time


@app.cell
def _(COLORS, mo, np, plt, toy_diffusion, toy_seed, toy_time):
    _rng = np.random.default_rng(toy_seed.value)
    _count = 30000
    _variance0 = .35
    _t = toy_time.value
    _g = toy_diffusion.value
    _x0 = _rng.normal(0.0, np.sqrt(_variance0), _count)

    _sde_noise = _rng.normal(0.0, np.sqrt(max(_t, 0.0)), _count)
    _x_sde = _x0 + _g * _sde_noise
    _scale = np.sqrt((_variance0 + _g**2 * _t) / _variance0)
    _x_ode = _scale * _x0
    _theory_variance = _variance0 + _g**2 * _t

    _fig, _axes = plt.subplots(1, 2, figsize=(10, 3.8))
    _bins = np.linspace(-3.2, 3.2, 70)
    _axes[0].hist(
        _x_sde, bins=_bins, density=True, histtype="step", linewidth=2,
        color=COLORS["data"], label="SDE 样本边缘",
    )
    _axes[0].hist(
        _x_ode, bins=_bins, density=True, histtype="step", linewidth=2,
        color=COLORS["success"], label="ODE 样本边缘",
    )
    _grid = np.linspace(-3.2, 3.2, 400)
    _density = np.exp(-.5 * _grid**2 / _theory_variance) / np.sqrt(2 * np.pi * _theory_variance)
    _axes[0].plot(_grid, _density, "--", color=COLORS["danger"], label="理论 Gaussian")
    _axes[0].set(title="群体边缘分布", xlabel="x", ylabel="density")
    _axes[0].legend(fontsize=8)

    _pick = np.arange(20)
    _axes[1].scatter(_x0[_pick], _x_sde[_pick], color=COLORS["data"], label="SDE：随机终点")
    _axes[1].scatter(_x0[_pick], _x_ode[_pick], color=COLORS["success"], marker="x", label="ODE：确定终点")
    _axes[1].set(title="同一初值的粒子终点", xlabel="初值 x(0)", ylabel="x(t)")
    _axes[1].legend(fontsize=8)
    _fig.tight_layout()

    _sde_variance = float(np.var(_x_sde))
    _ode_variance = float(np.var(_x_ode))
    mo.vstack(
        [
            mo.hstack([toy_time, toy_diffusion, toy_seed], widths="equal"),
            _fig,
            mo.md(
                f"""
                **图例与学习任务**

                - 左图蓝线=SDE 经验边缘，绿线=ODE 经验边缘，红虚线=理论密度；
                - 右图蓝点=SDE 随机终点，绿色叉=ODE 确定终点；
                - 理论方差={_theory_variance:.4f}，SDE 样本方差={_sde_variance:.4f}，
                  ODE 样本方差={_ode_variance:.4f}。

                左图接近说明当前 Monte Carlo 支持边缘等价；右图不重合则直接反驳
                “逐路径相同”。有限样本相符不是一般证明，一般证明在上一节。
                """
            ),
        ],
        gap=0.8,
    )
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 一维 toy 的 ODE 公式从哪里来

    本实验选择
    \[
    dX_t=g\,dW_t,\qquad X_0\sim\mathcal N(0,\sigma_0^2).
    \]
    所以
    \[
    p_t=\mathcal N(0,\sigma_0^2+g^2t),\qquad
    \nabla_x\log p_t(x)=-\frac{x}{\sigma_0^2+g^2t}.
    \]

    Probability flow ODE 为
    \[
    \frac{dx}{dt}
    =-\frac12g^2\nabla_x\log p_t(x)
    =\frac{g^2}{2(\sigma_0^2+g^2t)}x.
    \]

    积分得到
    \[
    x(t)=x(0)\sqrt{\frac{\sigma_0^2+g^2t}{\sigma_0^2}},
    \]
    因而 ODE 把初始方差精确缩放为
    \(\sigma_0^2+g^2t\)，与 SDE 相同；但 ODE 中 \(x(t)\) 是 \(x(0)\) 的
    确定倍数，SDE 还增加独立 Brownian noise。
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 公式到代码｜时间方向不能靠猜负号

        # forward-time probability flow velocity
        score = score_model(x, t)            # shape 与 x 相同
        velocity = f(x, t) - 0.5 * g(t)**2 * score

        # 反向生成：时间网格从 T 递减到 epsilon，所以 dt < 0
        x = x + velocity * dt

        # reverse SDE 的 drift 使用完整 g² score，并另外加入 Brownian 增量
        reverse_drift = f(x, t) - g(t)**2 * score
        x = x + reverse_drift * dt + g(t) * sqrt(abs(dt)) * noise

    这里的 reverse SDE 写法沿用论文“同一 t、反向积分”的约定。若改用正向增加的
    新变量 \(\tau=T-t\)，漂移符号需要整体变量替换。代码审查必须同时看公式、时间网格
    和 dt 符号，不能孤立地问 score 前面是加还是减。
    """)
    return


@app.cell
def _(mo):
    sde_claim = mo.ui.radio(
        options={
            "ODE 与 SDE 对每个初值给出同一路径": "path",
            "精确 score 下两者可共享每时刻边缘密度": "marginal",
            "ODE 不需要 score network": "no_score",
        },
        label="论文中的 equivalent 指什么？",
    )
    return (sde_claim,)


@app.cell
def _(mo, sde_claim):
    if sde_claim.value is None:
        _message, _kind = "先选择，再同时看原文 Figure 2 的白线、彩色线与背景密度。", "warn"
    elif sde_claim.value == "marginal":
        _message, _kind = "正确。密度演化相同；轨迹机制分别是确定性 ODE 与随机 SDE。", "success"
    else:
        _message, _kind = "不正确。两者路径不同，而且 probability flow ODE 仍显式依赖 score。", "danger"
    mo.callout(mo.md(_message), kind=_kind)
    return


@app.cell
def _(mo):
    mo.callout(
        mo.md(r"""
        ## 错误与反例

        1. **把 ODE 与 SDE 的 individual trajectories 说成相同。**  
           原文只保证在精确条件下共享 marginals。

        2. **在 ODE 中使用完整 \(g^2 score\)。**  
           Probability flow ODE 的系数是 \(1/2\)；否则密度扩散项会多一倍。

        3. **忘记 reverse-time 的积分方向。**  
           论文公式配合 \(T\to0\)；若代码 dt 为正，必须重新推导变量变换。

        4. **把近似网络与离散 solver 当成理论精确系统。**  
           score error 和 truncation error 都会破坏精确 marginal 等价。

        5. **由两张相似直方图宣称证明。**  
           直方图是有限样本核验；证明依赖 Fokker–Planck 与 continuity equation。
        """),
        kind="danger",
    )
    return


@app.cell
def _(exercise_block, mo):
    mo.vstack(
        [
            mo.md("## 学习检测｜从原图回到密度方程"),
            exercise_block(
                understanding=(
                    "为什么 probability flow ODE 的 score 系数是 reverse SDE 的一半？",
                    "SDE 自身的 Brownian 扩散贡献半个 g² Laplacian；ODE 没有随机扩散，必须把这部分通过速度场的 divergence 复制出来，因此使用一半 g² score。",
                ),
                calculation=(
                    r"若 \(p_t=N(0,4)\)，\(g=2\)，在 \(x=1\) 处的 ODE score 修正项是多少？",
                    r"score=-x/4=-1/4。速度中的修正是 \(-\frac12g^2 score=-2(-1/4)=0.5\)。若 forward drift 为零，总速度为 0.5。",
                ),
                coding=(
                    "反向时间网格 [1.0,0.8] 的 dt 是多少？Brownian 增量标准差用 sqrt(dt) 还是 sqrt(abs(dt))？",
                    "dt=0.8-1.0=-0.2。随机增量方差必须为正，使用 sqrt(abs(dt))；漂移项仍乘带符号的 dt。",
                ),
                exploration=(
                    "拖动 g 和 t，找一个左右图差异最明显、左图仍接近的设置，并解释。",
                    "g 或 t 增大时 Brownian 路径分散更明显，而 ODE 仍按初值确定缩放；只要使用精确 toy score，两者理论方差仍相同。比较直方图、样本方差和逐初值散点三类证据。",
                ),
            ),
        ],
        gap=0.8,
    )
    return


@app.cell
def _(GUIDE, paper_guide_footer):
    paper_guide_footer(
        GUIDE,
        takeaways=(
            "原文 Figure 2 中彩色 SDE 路径和白色 ODE 路径不同，背景边缘密度才是共享对象。",
            "Probability flow ODE 的半系数可由 Fokker–Planck 与 continuity equation 严格推出。",
            "reverse-time 公式必须与 T→0 的积分方向一起阅读。",
            "精确等价依赖真实 score、正则条件和连续动力学；神经网络与数值 solver 会引入误差。",
        ),
        bridge=(
            "下一部分 Flow Matching 将不再通过 score 间接构造 probability flow velocity，"
            "而是直接回归一条概率路径的条件速度场。"
        ),
    )
    return


if __name__ == "__main__":
    app.run()
