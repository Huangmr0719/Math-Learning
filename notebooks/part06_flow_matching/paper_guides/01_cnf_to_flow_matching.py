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
    GUIDE = PAPER_GUIDES["flow_matching_lineage"]
    paper_guide_header(GUIDE, duration="110–160 分钟，可分两次完成")
    return (GUIDE,)


@app.cell
def _(mo):
    mo.callout(
        mo.md(
            """
            ## 先抓住论文谱系中的三个不同问题

            1. **Neural ODE**：能否让神经网络直接描述连续时间的导数？
            2. **FFJORD**：CNF 的 Jacobian trace 在高维怎样便宜地估计？
            3. **Flow Matching**：能否不在训练内反复求解当前 ODE，直接监督速度场？

            三篇论文依次改变“模型表示”“密度计算”和“训练信号”。Flow Matching
            继承 CNF 的生成机制，却不等于 FFJORD 的另一种 trace estimator。
            """
        ),
        kind="info",
    )
    return


@app.cell
def _(PAPERS, paper_evidence_block):
    paper_evidence_block(
        PAPERS["neural_ode"],
        title="从有限层跳跃到连续向量场",
        original_quote="Instead of specifying a discrete sequence of hidden layers, we parameterize the derivative of the hidden state.",
        chinese_translation="作者不再逐层指定状态更新，而是让神经网络给出状态关于时间的导数。",
        figure_path="assets/paper_figures/flow_matching/neural_ode_figure1.webp",
        figure_number="原文 Figure 1",
        figure_caption="左图是有限个残差层，右图是由 ODE 向量场连续运输状态；圆点表示函数求值位置。",
        legend=(
            "横轴表示一维状态位置；纵轴 Depth 在右图中可理解为连续时间。",
            "黑色圆点是网络或向量场被计算的位置，不是训练数据点。",
            "黑色粗轨迹是状态演化；淡色箭头表示局部更新方向。",
            "左图只在整数层更新；右图的 solver 可在需要的位置评估向量场。",
        ),
        learning_goal="指出残差更新 h(k+1)=h(k)+f(h(k))Δt 如何在 Δt 变小时过渡到 dh/dt=f(h,t)。",
        evidence_boundary="这张概念图说明离散层与连续动力学的表示差异；它不证明 ODE 网络一定更准确、更省时或更容易训练。",
    )
    return


@app.cell
def _(derivation_map, mo):
    mo.vstack(
        [
            mo.md("## 第一跳｜瞬时变量替换为什么出现 trace"),
            derivation_map(
                [
                    "从离散变换的 log|det J| 出发",
                    "令一步接近 I+ΔtJv",
                    "使用 log det(I+ΔtA)=Δt tr(A)+o(Δt)",
                    "除以 Δt 并取极限",
                    "得到 d log p/dt=-tr(Jv)",
                ]
            ),
            mo.md(
                r"""
                对 ODE \(\dot x_t=v_\theta(x_t,t)\)，一个极短时间步是
                \[
                x_{t+\Delta t}=x_t+\Delta t\,v_\theta(x_t,t)+o(\Delta t).
                \]
                小步映射的 Jacobian 为 \(I+\Delta tJ_v+o(\Delta t)\)。离散变量替换给出
                \[
                \log p_{t+\Delta t}(x_{t+\Delta t})-\log p_t(x_t)
                =-\log\left|\det(I+\Delta tJ_v+o(\Delta t))\right|.
                \]
                使用矩阵极限公式后：
                \[
                \frac{d}{dt}\log p_t(x_t)
                =-\operatorname{tr}\!\left(\frac{\partial v_\theta}{\partial x}\right)
                =-\nabla\cdot v_\theta(x_t,t).
                \]

                **维度检查：**\(x,v\in\mathbb R^D\)，\(J_v\in\mathbb R^{D\times D}\)，
                trace 和每个样本的 log-density change 都是标量。推导要求向量场足够光滑，
                且 ODE 解在区间内存在并唯一。
                """
            ),
        ],
        gap=0.8,
    )
    return


@app.cell
def _(PAPERS, paper_evidence_block):
    paper_evidence_block(
        PAPERS["ffjord"],
        title="一张密度图读懂 CNF 在运输什么",
        original_quote="We use Hutchinson's trace estimator to give a scalable unbiased estimate of the log-density.",
        chinese_translation="作者用 Hutchinson trace estimator 构造可扩展的无偏 log-density 估计。",
        figure_path="assets/paper_figures/flow_matching/ffjord_figure1.webp",
        figure_number="原文 Figure 1",
        figure_caption="底部简单 base density 沿连续动力学运输成顶部多峰 target density。",
        legend=(
            "底部 p(z(t0)) 是单峰 base density；顶部 p(z(t1)) 是多峰 target density。",
            "彩色背景表示时间和位置上的密度：黄色较高、紫色较低。",
            "白色流线与箭头是粒子运动方向，纵轴从 t=0 指向 t=1。",
            "曲线与背景描述群体密度；单条白线描述一个状态轨迹。",
        ),
        learning_goal="沿一条白线说明：状态由 ODE 更新，而 log density 由负 divergence 同步累计。",
        evidence_boundary="Figure 1 是一维机制示意，不证明高维 estimator 方差很小，也不证明有限模型可以拟合任意密度。",
    )
    return


@app.cell
def _(mo):
    probes = mo.ui.slider(1, 200, value=20, step=1, show_value=True, label="Hutchinson probes")
    matrix_dim = mo.ui.slider(2, 80, value=20, step=2, show_value=True, label="Jacobian 维度 D")
    trace_seed = mo.ui.slider(0, 30, value=7, step=1, show_value=True, label="随机 seed")
    return matrix_dim, probes, trace_seed


@app.cell
def _(COLORS, matrix_dim, mo, np, plt, probes, trace_seed):
    _rng = np.random.default_rng(trace_seed.value)
    _d = matrix_dim.value
    _matrix = _rng.normal(size=(_d, _d)) / np.sqrt(_d)
    _true_trace = float(np.trace(_matrix))
    _eps = _rng.choice([-1.0, 1.0], size=(200, _d))
    _samples = np.einsum("bi,ij,bj->b", _eps, _matrix, _eps)
    _running = np.cumsum(_samples) / np.arange(1, 201)
    _estimate = float(_running[probes.value - 1])

    _fig, _axes = plt.subplots(1, 2, figsize=(10, 3.8))
    _axes[0].plot(np.arange(1, 201), _running, color=COLORS["model"], label="累计 Hutchinson 估计")
    _axes[0].axhline(_true_trace, linestyle="--", color=COLORS["danger"], label="精确 trace")
    _axes[0].axvline(probes.value, linestyle=":", color=COLORS["prior"], label="当前 probes")
    _axes[0].set(xlabel="probe 数量", ylabel="trace estimate", title="无偏不等于单次精确")
    _axes[0].legend(fontsize=8)
    _dimensions = np.arange(2, 202, 2)
    _axes[1].plot(_dimensions, _dimensions**2, color=COLORS["danger"], label="显式 Jacobian：D²")
    _axes[1].plot(_dimensions, _dimensions, color=COLORS["success"], label="一次向量乘积：D")
    _axes[1].set(xlabel="状态维度 D", ylabel="toy operation scale", yscale="log", title="复杂度趋势示意")
    _axes[1].legend(fontsize=8)
    _fig.tight_layout()

    mo.vstack(
        [
            mo.hstack([matrix_dim, probes, trace_seed], widths="equal"),
            _fig,
            mo.md(
                f"""
                **图例与任务：**左图蓝线是累计样本均值，红虚线是真实 np.trace(A)，
                紫点线是当前 probe 数；真实值为 {_true_trace:.4f}，当前估计为 {_estimate:.4f}。
                右图只是维度趋势，不是实测运行时间。

                固定 seed 增加 probes，再更换 seed。应观察“平均意义下趋近”，而不是期待
                单次估计精确。该实验不能证明有限 solver tolerance 下的完整 likelihood
                estimator 仍严格无偏。
                """
            ),
        ],
        gap=0.8,
    )
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 第二跳｜Hutchinson 恒等式只做了哪一步

    令随机向量 \(\epsilon\) 满足
    \(E[\epsilon]=0\)、\(E[\epsilon\epsilon^\top]=I\)。对任意方阵 \(A\)：
    \[
    E[\epsilon^\top A\epsilon]
    =\operatorname{tr}(A E[\epsilon\epsilon^\top])
    =\operatorname{tr}(A).
    \]
    取 \(A=J_v\)，自动微分可计算 \(\epsilon^\top J_v\epsilon\)，而不显式构造 Jacobian。

        velocity = vector_field(x, t)                       # [B, D]
        epsilon = sample_rademacher_like(x)                 # [B, D]
        scalar = (velocity * epsilon).sum()                 # 标量
        vjp = autograd.grad(scalar, x)[0]                   # [B, D]
        trace_estimate = (vjp * epsilon).sum(-1)            # [B]
        dlogp_dt = -trace_estimate                          # [B]

    “无偏”针对随机 probe 对固定 Jacobian 的期望；估计方差、ODE 离散误差、网络误差和
    训练误差仍然存在。
    """)
    return


@app.cell
def _(mo):
    bottleneck = mo.ui.dropdown(
        options={
            "需要连续状态表示": "representation",
            "高维 trace 太贵": "trace",
            "训练内反复解 ODE 太慢": "training",
        },
        value="训练内反复解 ODE 太慢",
        label="当前研究瓶颈",
    )
    return (bottleneck,)


@app.cell
def _(bottleneck, mo):
    _answers = {
        "representation": ("Neural ODE", "用 vθ(x,t) 参数化连续时间导数。"),
        "trace": ("FFJORD", "用 Hutchinson estimator 近似 Jacobian trace。"),
        "training": ("Flow Matching", "从可采样条件路径构造速度监督，训练时不必先模拟当前 CNF。"),
    }
    _paper, _reason = _answers[bottleneck.value]
    mo.callout(mo.md(f"**对应论文坐标：{_paper}。** {_reason}"), kind="success")
    return


@app.cell
def _(derivation_map, mo):
    mo.vstack(
        [
            mo.md("## 第三跳｜Flow Matching 绕开的不是 ODE 生成"),
            derivation_map(
                [
                    "选择可采样 probability path",
                    "写 conditional path",
                    "得到 conditional velocity",
                    "用 MSE 训练 vθ",
                    "推理时解 learned ODE",
                ]
            ),
            mo.md(
                r"""
                Flow Matching 使用
                \[
                \mathcal L_{\mathrm{CFM}}(\theta)
                =E\left[\|v_\theta(X_t,t)-u_t(X_t\mid X_1)\|^2\right].
                \]
                条件路径和速度可以逐样本采样，所以训练不需要先用当前模型跑完整 ODE。
                但生成阶段仍要求解 \(\dot X_t=v_\theta(X_t,t)\)。

                因而 **simulation-free training** 不是“完全不用 solver”。速度 MSE 也不会
                自动给出 likelihood；若要评估归一化密度，仍需积分 divergence。
                """
            ),
        ],
        gap=0.8,
    )
    return


@app.cell
def _(mo):
    lineage_claim = mo.ui.radio(
        options={
            "Flow Matching 训练和生成都不需要 ODE solver": "none",
            "Flow Matching 避免训练内模拟当前 CNF，生成仍需求解 ODE": "correct",
            "FFJORD 与 Flow Matching 使用同一个训练目标": "same",
        },
        label="哪一句准确？",
    )
    return (lineage_claim,)


@app.cell
def _(lineage_claim, mo):
    if lineage_claim.value is None:
        _message, _kind = "先作答：区分训练监督与生成机制。", "warn"
    elif lineage_claim.value == "correct":
        _message, _kind = "正确。simulation-free 修饰训练目标，不会删除生成 ODE。", "success"
    else:
        _message, _kind = "不准确。FFJORD 是 likelihood 路线；Flow Matching 改用条件速度回归。", "danger"
    mo.callout(mo.md(_message), kind=_kind)
    return


@app.cell
def _(mo):
    mo.callout(
        mo.md(
            """
            ## 错误诊断

            - 把 trace 当 determinant：CNF 使用瞬时 trace，离散 flow 使用整步 log-determinant。
            - 把 Hutchinson 无偏理解成零方差：单次 probe 可以偏离真实 trace。
            - 把 simulation-free 写成 sampling-free：生成仍需积分 ODE。
            - 从一维密度图推出高维性能：原图解释机制，不替代 benchmark。
            """
        ),
        kind="danger",
    )
    return


@app.cell
def _(exercise_block, mo):
    mo.vstack(
        [
            mo.md("## 学习检测｜把论文名称还原成所解决的瓶颈"),
            exercise_block(
                understanding=(
                    "为什么 Flow Matching 属于 CNF 路线，却不以 likelihood 作为默认训练入口？",
                    "它仍学习生成 ODE 的向量场，但用条件路径提供速度回归监督；默认训练损失不积分 log density。",
                ),
                calculation=(
                    r"若二维速度场 \(v(x)=(3x_1,-2x_2)\)，沿轨迹的 \(d\log p/dt\) 是多少？",
                    "divergence=3-2=1，因此 dlogp_dt = -1；它是每个样本的标量。",
                ),
                coding=(
                    "删除 trace_estimate = (vjp * epsilon).sum(-1) 中的 sum(-1) 会出现什么 shape 错误？",
                    "结果保留 [B,D]，但每个样本的 log_p 与 dlogp_dt 应是 [B]。",
                ),
                exploration=(
                    "拖动 probe 数和 seed，怎样判断 estimator 方差，而不是只看一次误差？",
                    "固定矩阵比较多个 seed 在相同 probe 数下的分散程度；probe 增多通常更稳定，但单条曲线不是方差界。",
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
            "Neural ODE 把离散层更新提升为连续向量场，并导出瞬时变量替换。",
            "FFJORD 用 Hutchinson estimator 降低 trace 计算负担，但引入估计方差。",
            "Flow Matching 改变训练监督；生成阶段仍求解 learned ODE。",
            "模型表示、密度计算和训练目标是三个不同设计层。",
        ),
        bridge="下一篇把 Flow Matching 与 Rectified Flow 放进同一组符号，区分条件路径、边缘速度、coupling 与 reflow。",
    )
    return


if __name__ == "__main__":
    app.run()
