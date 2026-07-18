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
    GUIDE = PAPER_GUIDES["flow_matching_rectified_flow"]
    paper_guide_header(GUIDE, duration="140–200 分钟，建议分两次完成")
    return (GUIDE,)


@app.cell
def _(mo):
    mo.callout(
        mo.md(
            r"""
            ## 先拆开四个对象

            - **Coupling（耦合）**：怎样联合抽取起点 \(X_0\) 与终点 \(X_1\)。
            - **Conditional path（条件路径）**：固定隐藏条件后，中间样本怎样生成。
            - **Conditional velocity（条件速度）**：知道隐藏配对时的瞬时速度。
            - **Marginal velocity（边缘速度）**：模型只看到 \(X_t,t\) 时能学到的平均速度。

            Flow Matching 给出从条件对象训练边缘 CNF 的一般框架；Rectified Flow 选择线性
            插值，并进一步研究 ODE 诱导的新 coupling 与 reflow。共享回归形式不代表
            两篇论文的全部方法和定理都相同。
            """
        ),
        kind="warn",
    )
    return


@app.cell
def _(PAPERS, paper_evidence_block):
    paper_evidence_block(
        PAPERS["flow_matching"],
        title="Diffusion path 与 OT conditional path 改变了什么",
        original_quote="Flow Matching is compatible with a general family of Gaussian probability paths.",
        chinese_translation="Flow Matching 可以配合一族高斯概率路径；扩散路径只是其中一种选择。",
        figure_path="assets/paper_figures/flow_matching/flow_matching_figures2_3.webp",
        figure_number="原文 Figure 2–3",
        figure_caption="Figure 2 比较条件场，Figure 3 比较单样本轨迹；课程保留原页公式上下文。",
        legend=(
            "Figure 2 左半是 diffusion path 的 conditional score，右半是 OT path 的 conditional vector field。",
            "每组小图从 t=0、1/3、2/3 到 1；黑点 x1 是固定目标，背景是条件密度。",
            "原图说明蓝色表示较大幅值、红色表示较小幅值；黑线段表示局部方向。",
            "Figure 3 左侧 diffusion paths 可弯曲或 overshoot，右侧 OT conditional paths 为直线。",
            "这里的 OT 是条件 Gaussian 间的 displacement map，不保证最终 marginal field 是全局 OT 解。",
        ),
        learning_goal="先比较同一时刻的箭头方向，再比较整条轨迹，说明 path choice 如何改变回归目标。",
        evidence_boundary="这些二维图不证明 OT path 对所有数据、网络、coupling 和计算预算都优于 diffusion path。",
    )
    return


@app.cell
def _(derivation_map, mo):
    mo.vstack(
        [
            mo.md("## 核心证明一｜条件速度为什么能生成边缘路径"),
            derivation_map(
                [
                    "写条件 continuity equation",
                    "按条件变量积分",
                    "交换微分与积分",
                    "累加 probability flux",
                    "除以边缘密度",
                    "得到边缘 continuity equation",
                ]
            ),
            mo.md(
                r"""
                用 \(Z\) 表示终点或端点对。假设每条条件路径满足
                \[
                \partial_t p_t(x\mid z)+
                \nabla\cdot\bigl(p_t(x\mid z)u_t(x\mid z)\bigr)=0.
                \]
                边缘密度为
                \[
                p_t(x)=\int p_t(x\mid z)q(z)\,dz.
                \]
                在可交换微分与积分、边界条件合适时：
                \[
                \begin{aligned}
                \partial_t p_t(x)
                &=-\nabla\cdot\int p_t(x\mid z)u_t(x\mid z)q(z)\,dz\\
                &=-\nabla\cdot\bigl(p_t(x)u_t(x)\bigr),
                \end{aligned}
                \]
                其中
                \[
                u_t(x)=\int u_t(x\mid z)
                \frac{p_t(x\mid z)q(z)}{p_t(x)}\,dz
                =E[u_t(X_t\mid Z)\mid X_t=x].
                \]

                这就是边缘速度：在相同可观察位置和时间上，对隐藏路径速度按后验权重求平均。
                比值要求 \(p_t(x)>0\)；零密度区域没有训练样本要求模型唯一取值。
                """
            ),
        ],
        gap=0.8,
    )
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 核心证明二｜CFM 与不可计算的 FM 为什么有相同梯度

    记 \(U=u_t(X_t\mid Z)\)，\(m(X_t,t)=E[U\mid X_t,t]\)。对预测
    \(v_\theta(X_t,t)\)，平方误差满足正交分解：
    \[
    \begin{aligned}
    E\|v_\theta-U\|^2
    &=E\|v_\theta-m+m-U\|^2\\
    &=E\|v_\theta-m\|^2+E\|U-m\|^2.
    \end{aligned}
    \]
    交叉项为零，因为 \(E[U-m\mid X_t,t]=0\)。第二项是条件速度的不可约方差，
    与 \(\theta\) 无关。因此 CFM 与直接回归边缘速度的 FM 目标相差参数无关常数，
    具有相同总体梯度和最优预测。

    相同的是总体期望梯度；有限 minibatch 的随机梯度、方差和优化轨迹不必逐步相同。
    网络容量不足时，得到的是当前函数类中的最佳投影。
    """)
    return


@app.cell
def _(mo):
    regression_seed = mo.ui.slider(0, 30, value=6, step=1, show_value=True, label="回归实验 seed")
    sample_count = mo.ui.slider(200, 5000, value=1500, step=100, show_value=True, label="样本数")
    return regression_seed, sample_count


@app.cell
def _(COLORS, mo, np, plt, regression_seed, sample_count):
    _rng = np.random.default_rng(regression_seed.value)
    _x = _rng.normal(size=sample_count.value)
    _hidden_noise = _rng.choice([-1.5, 1.5], size=sample_count.value)
    _u_conditional = 2.0 * _x + _hidden_noise
    _u_marginal = 2.0 * _x
    _theta = np.linspace(-1.0, 5.0, 180)
    _prediction = _theta[:, None] * _x[None, :]
    _loss_cfm = np.mean((_prediction - _u_conditional[None, :]) ** 2, axis=1)
    _loss_fm = np.mean((_prediction - _u_marginal[None, :]) ** 2, axis=1)
    _difference = _loss_cfm - _loss_fm

    _fig, _axes = plt.subplots(1, 2, figsize=(10, 3.8))
    _axes[0].plot(_theta, _loss_cfm, color=COLORS["model"], label="条件目标 CFM")
    _axes[0].plot(_theta, _loss_fm, "--", color=COLORS["success"], label="边缘目标 FM")
    _axes[0].axvline(2.0, linestyle=":", color=COLORS["danger"], label="总体最优 θ=2")
    _axes[0].set(xlabel="模型参数 θ", ylabel="empirical MSE", title="两条损失相差近似常数")
    _axes[0].legend(fontsize=8)
    _axes[1].plot(_theta, _difference, color=COLORS["prior"], label="CFM − FM")
    _axes[1].axhline(2.25, linestyle="--", color=COLORS["danger"], label="理论条件方差 2.25")
    _axes[1].set(xlabel="θ", ylabel="loss difference", title="参数无关项的数值核验")
    _axes[1].legend(fontsize=8)
    _fig.tight_layout()

    mo.vstack(
        [
            mo.hstack([regression_seed, sample_count], widths="equal"),
            _fig,
            mo.md(
                f"""
                **图例与任务：**左图蓝实线是条件标签损失、绿虚线是边缘标签损失，
                红点线是真实总体最优参数；右图紫线是两损失之差，红虚线是理论条件方差。
                当前差值随 θ 的波动范围为 {float(np.ptp(_difference)):.4f}。

                增大样本数应使紫线更接近水平。有限样本相符是数值核验，不替代上一节证明。
                """
            ),
        ],
        gap=0.8,
    )
    return


@app.cell
def _(PAPERS, paper_evidence_block):
    paper_evidence_block(
        PAPERS["rectified_flow"],
        title="交叉的直线监督为什么不会成为交叉的 ODE 轨迹",
        original_quote="The trajectories are rewired at the intersection points to avoid the crossing.",
        chinese_translation="ODE 在交点处重新连接个体轨迹，避免违反解唯一性要求的不交叉性质。",
        figure_path="assets/paper_figures/flow_matching/rectified_flow_figure2.webp",
        figure_number="原文 Figure 2",
        figure_caption="从交叉线性插值到 rewiring，再到 reflow 后更直的 coupling。",
        legend=(
            "紫色点云 π0 是 source，红色点云 π1 是 target；四个小图的端点分布不变。",
            "浅蓝或浅绿线是当前 coupling 的 linear interpolation；深色线是 learned ODE trajectories。",
            "(a) 监督直线交叉；(b) ODE 在交点附近 rewiring，保持轨迹不交叉。",
            "(c) 用第一次 ODE 端点的新 coupling 再做插值；(d) 第二次 rectification 得到更直的流。",
            "X0、X1 是原配对，Z0、Z1 是 ODE 诱导的新配对；reflow 会改变 coupling。",
        ),
        learning_goal="按 (a) 到 (d) 说清监督直线、learned ODE trajectory 与 induced coupling 如何轮换。",
        evidence_boundary="二维示意不证明一次 reflow 达到最优传输，也不保证有限网络下每条轨迹都变直。",
    )
    return


@app.cell
def _(mo):
    crossing_time = mo.ui.slider(0.0, 1.0, value=0.5, step=0.02, show_value=True, label="插值时间 t")
    pairing_mode = mo.ui.dropdown(
        options={"交叉 coupling": "cross", "不交叉 coupling": "straight"},
        value="交叉 coupling",
        label="端点配对",
    )
    return crossing_time, pairing_mode


@app.cell
def _(COLORS, crossing_time, mo, np, pairing_mode, plt):
    _x0 = np.array([[-1.0, -1.0], [-1.0, 1.0]])
    _x1 = (
        np.array([[1.0, 1.0], [1.0, -1.0]])
        if pairing_mode.value == "cross"
        else np.array([[1.0, -1.0], [1.0, 1.0]])
    )
    _velocity = _x1 - _x0
    _xt = (1.0 - crossing_time.value) * _x0 + crossing_time.value * _x1
    _mean_velocity = np.mean(_velocity, axis=0)

    _fig, _axes = plt.subplots(1, 2, figsize=(10, 3.9))
    for _i, _color in enumerate((COLORS["data"], COLORS["success"])):
        _axes[0].plot(
            [_x0[_i, 0], _x1[_i, 0]],
            [_x0[_i, 1], _x1[_i, 1]],
            color=_color,
            label=f"条件路径 {_i + 1}",
        )
        _axes[0].scatter(*_xt[_i], color=_color, s=70)
        _axes[0].quiver(*_xt[_i], *_velocity[_i], angles="xy", scale_units="xy", scale=4, color=_color)
    _axes[0].scatter(_x0[:, 0], _x0[:, 1], color=COLORS["prior"], marker="o", label="source π0")
    _axes[0].scatter(_x1[:, 0], _x1[:, 1], color=COLORS["danger"], marker="s", label="target π1")
    _axes[0].set(xlim=(-1.3, 1.3), ylim=(-1.3, 1.3), title="隐藏配对下的条件路径", xlabel="x₁", ylabel="x₂")
    _axes[0].legend(fontsize=7)

    _axes[1].quiver(0, 0, *_velocity[0], angles="xy", scale_units="xy", scale=4, color=COLORS["data"], label="条件速度 1")
    _axes[1].quiver(0, 0, *_velocity[1], angles="xy", scale_units="xy", scale=4, color=COLORS["success"], label="条件速度 2")
    _axes[1].quiver(0, 0, *_mean_velocity, angles="xy", scale_units="xy", scale=4, color=COLORS["danger"], width=.018, label="条件平均")
    _axes[1].set(xlim=(-.2, 1.0), ylim=(-.8, .8), title="同一位置只能输出一个 ODE 速度", xlabel="velocity x₁", ylabel="velocity x₂")
    _axes[1].legend(fontsize=7)
    _fig.tight_layout()
    _separation = float(np.linalg.norm(_xt[0] - _xt[1]))

    mo.vstack(
        [
            mo.hstack([pairing_mode, crossing_time], widths="equal"),
            _fig,
            mo.md(
                f"""
                **图例与任务：**左图紫圆是 source、红方是 target，蓝绿线和箭头是条件路径与速度；
                右图蓝绿箭头是隐藏标签，红箭头是条件平均。当前中间点距离为 {_separation:.3f}。

                在交叉 coupling、t=0.5 时，两条路径来到同一点却要求不同速度；只看位置和时间的
                ODE 网络只能给一个值，MSE 最优解是条件平均。这解释 rewiring，但不是全部理论证明。
                """
            ),
        ],
        gap=0.8,
    )
    return


@app.cell
def _(derivation_map, mo):
    mo.vstack(
        [
            mo.md("## Rectification 与 OT｜相邻，但不等同"),
            derivation_map(
                [
                    "从任意 coupling 采样端点",
                    "用线性插值构造监督",
                    "学习条件平均速度",
                    "解 ODE 得到新端点",
                    "形成新的确定性 coupling",
                    "可选 reflow 再训练",
                ]
            ),
            mo.md(
                r"""
                Rectified Flow 的基本目标是
                \[
                \min_v\int_0^1 E\|X_1-X_0-v(X_t,t)\|^2dt,
                \qquad X_t=(1-t)X_0+tX_1.
                \]
                它与简单线性 CFM 具有相同的条件期望结构；Rectified Flow 进一步研究
                rectification operator：learned ODE 的端点 \((Z_0,Z_1)\) 形成新 coupling，
                再递归 reflow。

                在原论文精确条件下，一类凸成本满足
                \[
                E[c(Z_1-Z_0)]\le E[c(X_1-X_0)].
                \]
                这是“凸运输成本不增加”，不是“一步得到全局二次 OT 唯一解”。有限网络、
                经验风险、minibatch 和离散 solver 都会造成理论算子与实现之间的误差。
                """
            ),
        ],
        gap=0.8,
    )
    return


@app.cell
def _(mo):
    relation_claim = mo.ui.radio(
        options={
            "线性 CFM 与 Rectified Flow 目标相近，但 reflow 是额外机制": "correct",
            "Rectified Flow 每次训练都精确求解全局 OT": "ot",
            "条件直线相交时 learned ODE 也必须相交": "cross",
        },
        label="哪一种关系表述准确？",
    )
    return (relation_claim,)


@app.cell
def _(mo, relation_claim):
    if relation_claim.value is None:
        _message, _kind = "先回答，再回看原图中的浅色直线和深色 ODE 轨迹。", "warn"
    elif relation_claim.value == "correct":
        _message, _kind = "正确。共享回归结构不等于全部方法与定理相同。", "success"
    else:
        _message, _kind = "不正确。确定性 ODE 轨迹不能同刻交叉；rectification 也不等于一次精确 OT。", "danger"
    mo.callout(mo.md(_message), kind=_kind)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 公式到代码｜先声明 source、data 与时间方向

        x0 = sample_source(batch_size)              # [B, D]，source noise
        x1 = sample_data(batch_size)                # [B, D]，target data
        t = torch.rand(batch_size, 1)               # [B, 1]
        x_t = (1.0 - t) * x0 + t * x1              # [B, D]
        u_conditional = x1 - x0                     # [B, D]
        v = model(x_t, t)                           # [B, D]
        loss = ((v - u_conditional) ** 2).mean()

    Flow Matching 原文常用 \(x_1\) 表示数据条件，并允许末端保留很小的
    \(\sigma_{\min}>0\)；diffusion 代码常用 \(x_0\) 表示干净数据。移植公式前必须
    声明 source、data 和时间方向，不能只凭变量名猜含义。
    """)
    return


@app.cell
def _(mo):
    mo.callout(
        mo.md(
            """
            ## 错误与反例

            1. 把单样本条件速度当成模型处处应输出的速度：模型看不到隐藏配对。
            2. 把 conditional OT path 称为全局数据 OT：原文明确提醒二者不能直接画等号。
            3. 认为直线监督必然产生直线 ODE：交叉监督会被平均并 rewiring。
            4. 把 reflow 当 distillation：reflow 可以改变 coupling。
            5. 把成本不增写成有限实现必然改善 FID：理论成本与感知指标不是同一对象。
            """
        ),
        kind="danger",
    )
    return


@app.cell
def _(exercise_block, mo):
    mo.vstack(
        [
            mo.md("## 学习检测｜从条件标签恢复可生成的边缘速度"),
            exercise_block(
                understanding=(
                    "为什么交叉监督路径不会强迫确定性 ODE 在同一时刻真正交叉？",
                    "同一位置和时间上向量场只能给一个速度；平方损失选择隐藏条件速度的条件平均并重新连接路径。",
                ),
                calculation=(
                    "某位置有等概率速度 (2,2) 与 (2,-2)，最优边缘速度和最小平方范数 MSE 是多少？",
                    "条件平均是 (2,0)；到两个标签的平方距离均为 4，所以按样本平均的平方范数 MSE 是 4。",
                ),
                coding=(
                    "若 t 的 shape 是 [B]，怎样修改 x_t=(1-t)*x0+t*x1 才能广播到 [B,D]？",
                    "先写 t = t[:, None] 得到 [B,1]，再检查 assert x_t.shape == x0.shape == x1.shape。",
                ),
                exploration=(
                    "比较两种 coupling，说明‘线更直’与‘边缘分布正确’为何是两个判断。",
                    "合法 coupling 只保证两端 marginals；中间直度、learned ODE 误差和最终端点分布仍需分别检查。",
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
            "条件 probability flux 边缘化后得到正确的边缘速度与 continuity equation。",
            "CFM 与 FM 平方损失相差参数无关的条件方差项。",
            "线性监督可以交叉；确定性 ODE 学到条件平均并 rewiring。",
            "Rectified Flow 的 reflow 重构 coupling；成本不增不能简写成一步精确 OT。",
        ),
        bridge="课程主线至此闭合：现在可以从概率路径、局部监督、边缘动力学、数值求解和证据边界比较三类生成模型。",
    )
    return


if __name__ == "__main__":
    app.run()
