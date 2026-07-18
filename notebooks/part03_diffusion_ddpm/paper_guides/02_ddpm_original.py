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

    from src.math_checks import (
        ddpm_mean_from_noise,
        ddpm_posterior_mean_variance,
        diffusion_coefficients,
        q_sample_from_x0,
    )
    from src.teaching import (
        PAPERS,
        PAPER_GUIDES,
        derivation_map,
        exercise_block,
        intuition_and_rigor,
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
        ddpm_mean_from_noise,
        ddpm_posterior_mean_variance,
        derivation_map,
        diffusion_coefficients,
        exercise_block,
        intuition_and_rigor,
        mo,
        np,
        paper_evidence_block,
        paper_guide_footer,
        paper_guide_header,
        plt,
        q_sample_from_x0,
    )


@app.cell
def _(PAPER_GUIDES, paper_guide_header):
    GUIDE = PAPER_GUIDES["ddpm_original"]
    paper_guide_header(GUIDE, duration="130–190 分钟，建议分两次完成")
    return (GUIDE,)


@app.cell
def _(PAPERS, mo):
    _paper = PAPERS["ddpm"]
    _followup = PAPERS["improved_ddpm"]
    mo.vstack(
        [
            mo.callout(
                mo.md(r"""
                ## 这次不是复习五章，而是重建一篇论文

                第 14–18 章按教学顺序拆开了 Markov 链、闭式加噪、高斯后验、
                变分目标和最小实现。论文把这些内容压缩在十几条公式与两段算法中。

                你的任务是回答：**每个等号依赖什么、论文真正优化什么、代码索引怎样
                对齐式号，以及哪些结论是后续论文才补上的。**
                """),
                kind="info",
            ),
            mo.md(
                fr"""
                ## 一手资料与版本

                - 主论文：[{_paper.title}]({_paper.url})
                - 中文题名：{_paper.chinese_title}
                - 作者与版本：{_paper.authors} · {_paper.venue_year}
                - 后续核对：[{_followup.title}]({_followup.url})

                本导读提供**教学性意译**，不代替论文原文。式号默认指 Ho et al.
                的 arXiv:2006.11239；不同版本若版式变化，应以公式内容而非页码定位。

                ### 完成标准

                1. 从式 (2) 归纳推出式 (4)；
                2. 从高斯乘积得到式 (6)–(7) 的 posterior；
                3. 说明式 (8)、式 (10) 与式 (12) 的关系；
                4. 把算法 1–2 写成带 shape 与索引说明的教学代码；
                5. 准确区分完整变分下界、加权噪声 MSE 与 \(L_{{simple}}\)。
                """
            ),
        ],
        gap=0.8,
    )
    return


@app.cell
def _(PAPERS, paper_evidence_block):
    paper_evidence_block(
        PAPERS["ddpm"],
        title="两条方向相反的链如何放进同一张图？",
        original_quote="Our best results are obtained by training on a weighted variational bound.",
        chinese_translation="作者明确把最佳样本结果与重新加权的变分训练目标联系起来，而不是宣称 simple loss 数值等于完整 ELBO。",
        figure_path="assets/paper_figures/ddim/ho_figure2_graphical_model.webp",
        figure_number="原文 Figure 2",
        figure_caption="上方实线是从 x_T 到 x_0 的生成链 pθ；下方虚线是从 x_0 到 x_T 的固定扩散链 q。",
        legend=(
            "灰色节点 x_T…x_0：同一数据空间在不同噪声时刻的随机变量。",
            "上方实箭头 pθ(x_{t-1}|x_t)：生成时学习的反向转移。",
            "下方虚箭头 q(x_t|x_{t-1})：训练时固定的 forward transition。",
            "小图只示意噪声程度；它不是每个时刻的定量方差曲线。",
        ),
        learning_goal="沿两种箭头分别复述训练已知什么、生成未知什么，并说出为什么 posterior 训练时额外条件于 x_0。",
        evidence_boundary="Figure 2 定义图模型方向，不证明 simple loss、样本质量或采样速度。",
    )
    return


@app.cell
def _(mo):
    reading_mode = mo.ui.radio(
        options={
            "第一次：图景 → 式号 → 算法": "first",
            "推导：式 (2) → (4) → (7) → (12)": "math",
            "研究：主张 → 证据 → 局限 → 后续": "research",
        },
        value="第一次：图景 → 式号 → 算法",
        label="选择本次阅读路线",
    )
    return (reading_mode,)


@app.cell
def _(mo, reading_mode):
    _routes = {
        "first": "先看两条链和交互加噪，再读推导，最后逐行对照训练与采样。",
        "math": "重点检查累计乘积、完成平方、方差选择以及噪声参数化的代数替换。",
        "research": "重点区分论文的数学等式、CIFAR-10 等实验观察，以及 Improved DDPM 的后续改进。",
    }
    mo.callout(
        mo.vstack([reading_mode, mo.md(_routes[reading_mode.value])], gap=0.5),
        kind="neutral",
    )
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 摘要教学性意译｜先抓研究承诺

    原文摘要可以拆成三层：

    1. 作者提出一类用变分推断训练的潜变量生成模型；
    2. 其设计与去噪得分匹配存在新的联系；
    3. 在当时的无条件图像生成基准上获得高质量样本，但对数似然仍有改进空间。

    这里不能把第三点改写成“DDPM 在所有指标上都最好”。论文自己区分了样本质量与
    likelihood；后续 Improved DDPM 才进一步研究 learned variance、混合目标、
    更少采样步数和 likelihood。
    """)
    return


@app.cell
def _(derivation_map, mo):
    mo.vstack(
        [
            mo.md("## 原文路线图｜式号不是孤岛"),
            derivation_map(
                [
                    "式 (1)：定义可学习的 reverse chain",
                    "式 (2)：定义固定的 forward chain",
                    "式 (4)：闭式采样任意 x_t",
                    "式 (5)：negative ELBO 分解为逐步项",
                    "式 (6)–(7)：已知 x_0 时的真实 posterior",
                    "式 (8)：同方差 Gaussian KL 化为均值 MSE",
                    "式 (10)–(12)：用 epsilon 参数化并得到加权 MSE",
                    "式 (14)：去权重的 simple objective",
                    "算法 1 训练；算法 2 反向生成",
                ]
            ),
        ],
        gap=0.8,
    )
    return


@app.cell
def _(intuition_and_rigor):
    intuition_and_rigor(
        r"""
        forward chain 像老师自动生成练习题：从干净答案开始，每次加入少量已知噪声。
        reverse chain 像学生学习纠错。训练时老师知道原答案，因此能构造任意难度的题；
        考试生成时没有原答案，只能从纯噪声开始连续纠错。
        """,
        r"""
        Forward process 是固定分布
        \[
        q(x_{1:T}\mid x_0)=\prod_{t=1}^T
        q(x_t\mid x_{t-1}),\quad
        q(x_t\mid x_{t-1})=\mathcal N(\sqrt{\alpha_t}x_{t-1},\beta_tI),
        \]
        其中 \(\alpha_t=1-\beta_t\)。

        Reverse model 是联合分布
        \[
        p_\theta(x_{0:T})=p(x_T)\prod_{t=1}^T
        p_\theta(x_{t-1}\mid x_t).
        \]

        两条链的条件方向相反；forward 的 \(q\) 由设计者固定，reverse 的
        \(p_\theta\) 才由数据学习。训练能访问 \(x_0\)，生成不能。
        """,
    )
    return


@app.cell
def _(mo):
    schedule_steps = mo.ui.slider(
        5, 100, value=40, step=5, show_value=True, label="总步数 T"
    )
    selected_time = mo.ui.slider(
        1, 40, value=15, step=1, show_value=True, label="观察时刻 t"
    )
    schedule_end = mo.ui.slider(
        0.005, 0.12, value=0.06, step=0.005, show_value=True, label="末端 beta"
    )
    sample_seed = mo.ui.slider(
        0, 30, value=7, step=1, show_value=True, label="噪声 seed"
    )
    return sample_seed, schedule_end, schedule_steps, selected_time


@app.cell
def _(
    COLORS,
    diffusion_coefficients,
    mo,
    np,
    plt,
    q_sample_from_x0,
    sample_seed,
    schedule_end,
    schedule_steps,
    selected_time,
):
    _total = schedule_steps.value
    _t = min(selected_time.value, _total)
    _betas = np.linspace(1e-4, schedule_end.value, _total)
    _, _alpha_bars = diffusion_coefficients(_betas)
    _alpha_bar = float(_alpha_bars[_t - 1])
    _rng = np.random.default_rng(sample_seed.value)
    _angles = np.linspace(0, 2 * np.pi, 500, endpoint=False)
    _x0 = np.column_stack([2 * np.cos(_angles), 2 * np.sin(_angles)])
    _x0 += .04 * _rng.standard_normal(_x0.shape)
    _noise = _rng.standard_normal(_x0.shape)
    _xt = q_sample_from_x0(_x0, _alpha_bar, _noise)

    _fig, _axes = plt.subplots(1, 3, figsize=(11, 3.6))
    _axes[0].scatter(_x0[:, 0], _x0[:, 1], s=7, alpha=.5, color=COLORS["data"], label="原始数据 x₀")
    _axes[0].set_title("数据 x₀")
    _axes[1].scatter(_xt[:, 0], _xt[:, 1], s=7, alpha=.45, color=COLORS["model"], label="闭式样本 xₜ")
    _axes[1].set_title(fr"直接采样 x$_{{{_t}}}$")
    _axes[2].plot(np.arange(1, _total + 1), _alpha_bars, color=COLORS["prior"], label="alpha_bar 曲线")
    _axes[2].scatter([_t], [_alpha_bar], color=COLORS["danger"], zorder=3, label="当前时刻")
    _axes[2].set(xlabel="t", ylabel="alpha_bar_t", title="累计信号保留率")
    for _ax in _axes[:2]:
        _ax.set_xlim(-3.5, 3.5)
        _ax.set_ylim(-3.5, 3.5)
        _ax.set_aspect("equal")
        _ax.legend(fontsize=8)
    _axes[2].legend(fontsize=8)
    _fig.tight_layout()

    mo.vstack(
        [
            mo.hstack(
                [schedule_steps, selected_time, schedule_end, sample_seed],
                widths="equal",
            ),
            _fig,
            mo.md(
                fr"""
                **图例与观察任务：**左图蓝点是数据，中央橙点是由同一数据闭式加噪得到的
                \(x_t\)，右图紫线是累计信号保留率、红点是当前时刻。拖动 beta 末值和 t，
                观察“曲线下降”与“结构消失”是否同步，而不是把颜色理解为概率大小。

                当前实际使用 \(t={_t}\)，\(\bar\alpha_t={_alpha_bar:.4f}\)。
                当总步数小于 UI 中选择的时刻时，本图把 \(t\) 截到 \(T\)；正式代码
                应让时间控件联动，或显式验证 \(1\le t\le T\)。这是公式诊断图，
                不用于比较不同论文的 FID 或采样速度。
                """
            ),
        ],
        gap=0.8,
    )
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 式 (4) 严格推导｜为什么可以跳过前面的所有加噪步

    假设
    \[
    x_{t-1}=\sqrt{\bar\alpha_{t-1}}x_0+
    \sqrt{1-\bar\alpha_{t-1}}\,\epsilon_{t-1},
    \]
    且新噪声 \(\epsilon_t'\) 与 \(\epsilon_{t-1}\) 独立。代入一步转移：

    \[
    \begin{aligned}
    x_t
    &=\sqrt{\alpha_t}x_{t-1}+\sqrt{\beta_t}\epsilon_t'\\
    &=\sqrt{\alpha_t\bar\alpha_{t-1}}x_0
      +\sqrt{\alpha_t(1-\bar\alpha_{t-1})}\epsilon_{t-1}
      +\sqrt{\beta_t}\epsilon_t'.
    \end{aligned}
    \]

    使用累计乘积 \(\bar\alpha_t=\alpha_t\bar\alpha_{t-1}\)。后两项是独立
    零均值 Gaussian，方差相加：

    \[
    \alpha_t(1-\bar\alpha_{t-1})+\beta_t
    =1-\alpha_t\bar\alpha_{t-1}=1-\bar\alpha_t.
    \]

    因而可把两个标准 Gaussian 的线性组合重新记为一个
    \(\epsilon\sim\mathcal N(0,I)\)，得到

    \[
    q(x_t\mid x_0)=
    \mathcal N(\sqrt{\bar\alpha_t}x_0,(1-\bar\alpha_t)I).
    \]

    这一步依赖噪声独立与高斯闭包。它说明**边缘分布相同**，不是说合并后的
    \(\epsilon\) 与每一步原噪声逐个相等。
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 式 (6)–(7)｜真实 posterior 从哪里来

    已知 \(x_0\) 与 \(x_t\) 时，用 Bayes rule 忽略与 \(x_{t-1}\) 无关的比例常数：

    \[
    q(x_{t-1}\mid x_t,x_0)
    \propto q(x_t\mid x_{t-1})q(x_{t-1}\mid x_0).
    \]

    两项都是关于 \(x_{t-1}\) 的 Gaussian。把指数中的二次项完成平方，得到

    \[
    q(x_{t-1}\mid x_t,x_0)
    =\mathcal N(\tilde\mu_t(x_t,x_0),\tilde\beta_tI),
    \]

    \[
    \tilde\mu_t=
    \frac{\sqrt{\bar\alpha_{t-1}}\beta_t}{1-\bar\alpha_t}x_0+
    \frac{\sqrt{\alpha_t}(1-\bar\alpha_{t-1})}{1-\bar\alpha_t}x_t,
    \qquad
    \tilde\beta_t=
    \frac{1-\bar\alpha_{t-1}}{1-\bar\alpha_t}\beta_t.
    \]

    ### 定义域与维度

    - \(x_0,x_t,\tilde\mu_t\) 具有相同 shape；
    - \(\beta_t,\alpha_t,\bar\alpha_t,\tilde\beta_t\) 是时间标量，批训练时 reshape 后广播；
    - \(0<\beta_t<1\)，且 \(t\ge1\)；
    - 当 \(t=1\) 时 \(\bar\alpha_0=1\)，posterior variance 为 0，对应最后一步不再加噪；
    - 真实 posterior 训练时可用，生成时没有 \(x_0\)，所以需要神经网络参数化均值。
    """)
    return


@app.cell
def _(
    ddpm_mean_from_noise,
    ddpm_posterior_mean_variance,
    diffusion_coefficients,
    mo,
    np,
):
    _betas = np.linspace(1e-4, 0.02, 50)
    _alphas, _alpha_bars = diffusion_coefficients(_betas)
    _index = 27
    _x0 = np.array([1.2, -0.7, 0.4])
    _epsilon = np.array([-0.3, 0.8, 1.1])
    _xt = (
        np.sqrt(_alpha_bars[_index]) * _x0
        + np.sqrt(1.0 - _alpha_bars[_index]) * _epsilon
    )
    _posterior_mean, _posterior_variance = ddpm_posterior_mean_variance(
        _x0,
        _xt,
        alpha_t=float(_alphas[_index]),
        alpha_bar_t=float(_alpha_bars[_index]),
        alpha_bar_previous=float(_alpha_bars[_index - 1]),
        beta_t=float(_betas[_index]),
    )
    _epsilon_mean = ddpm_mean_from_noise(
        _xt,
        _epsilon,
        alpha_t=float(_alphas[_index]),
        alpha_bar_t=float(_alpha_bars[_index]),
        beta_t=float(_betas[_index]),
    )
    _mean_error = float(np.max(np.abs(_posterior_mean - _epsilon_mean)))
    mo.callout(
        mo.md(
            fr"""
            ## 可执行核验｜posterior 均值与 epsilon 参数化

            - posterior variance：{_posterior_variance:.8f}
            - 两种均值的最大绝对差：{_mean_error:.2e}

            一种写法使用 \(x_0\)，另一种把
            \(x_0=(x_t-\sqrt{{1-\bar\alpha_t}}\epsilon)/\sqrt{{\bar\alpha_t}}\)
            代回。数值一致支持当前实现；代数恒等式仍由替换与化简证明。
            """
        ),
        kind="success" if _mean_error < 1e-10 else "danger",
    )
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 式 (5) 到式 (14)｜必须区分三种目标

    需要最小化的 negative ELBO 可分为

    \[
    L_{\rm VLB}
    =L_T+\sum_{t=2}^{T}L_{t-1}+L_0,
    \]

    其中中间项

    \[
    L_{t-1}
    =\mathbb E_q
    D_{\rm KL}\!\left(
      q(x_{t-1}\mid x_t,x_0)
      \,\|\,p_\theta(x_{t-1}\mid x_t)
    \right).
    \]

    若 reverse covariance 固定为 \(\sigma_t^2I\)，两个 Gaussian 的 KL 中与
    \(\theta\) 有关的部分是均值平方误差。再把模型均值参数化为

    \[
    \mu_\theta(x_t,t)=
    \frac1{\sqrt{\alpha_t}}
    \left(x_t-\frac{\beta_t}{\sqrt{1-\bar\alpha_t}}
    \epsilon_\theta(x_t,t)\right),
    \]

    可得到带时间权重的噪声 MSE：

    \[
    L_{t-1}-C
    =
    \mathbb E\left[
      \frac{\beta_t^2}
      {2\sigma_t^2\alpha_t(1-\bar\alpha_t)}
      \|\epsilon-\epsilon_\theta(x_t,t)\|^2
    \right].
    \]

    这里 \(C\) 与 \(\theta\) 无关，但权重通常随 \(t\) 改变。原文式 (14) 的

    \[
    L_{\rm simple}
    =\mathbb E_{t,x_0,\epsilon}
    \|\epsilon-\epsilon_\theta(x_t,t)\|^2
    \]

    **主动去掉了这项权重**。因此：

    - 对固定 \(t\)，正权重不改变该条件回归的总体最优函数；
    - 共享一个有限容量网络时，跨时间重新加权会改变优化重点和实际解；
    - \(L_{\rm simple}\) 不能被称为与完整 \(L_{\rm VLB}\) 数值相等。
    """)
    return


@app.cell
def _(mo):
    objective_view = mo.ui.dropdown(
        options={
            "完整 negative ELBO": "vlb",
            "带权 epsilon MSE": "weighted",
            "simple epsilon MSE": "simple",
        },
        value="完整 negative ELBO",
        label="切换目标，检查它保留了什么",
    )
    return (objective_view,)


@app.cell
def _(mo, objective_view):
    _views = {
        "vlb": (
            "严格对应似然下界",
            "包含终点 prior、逐步 KL 与 decoder 项；可用于 bits/dim 等 likelihood 评估。",
            "不同项的估计方差与优化权衡可能不利于样本质量。",
        ),
        "weighted": (
            "固定 reverse variance 后的逐步可学习部分",
            "由同方差 Gaussian KL 与 epsilon 参数化推出，保留时间相关权重。",
            "仍需单独处理端点项与与参数无关常数。",
        ),
        "simple": (
            "原文用于高质量生成的简化代理目标",
            "随机 t、随机 epsilon，直接做未加权 MSE，代码最清楚。",
            "不是完整 VLB 的逐项相等改写，也不直接给出精确 likelihood。",
        ),
    }
    _title, _keeps, _boundary = _views[objective_view.value]
    mo.vstack(
        [
            objective_view,
            mo.callout(
                mo.md(
                    f"**{_title}**\n\n- 保留：{_keeps}\n- 结论边界：{_boundary}"
                ),
                kind="neutral",
            ),
        ],
        gap=0.6,
    )
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 算法 1｜教学版训练代码与公式逐行对照

        # x0: [B, C, H, W]；论文时间 t∈{1,...,T}。
        t = randint(low=1, high=T + 1, size=[B])

        # epsilon 与 x0 同 shape，提供监督目标。
        epsilon = randn_like(x0)

        # gather 后 reshape 为 [B, 1, 1, 1]，才能按样本广播。
        alpha_bar_t = alpha_bars[t - 1].view(B, 1, 1, 1)

        # 原文式 (4)：无需执行前面的 t-1 次加噪。
        xt = sqrt(alpha_bar_t) * x0 + sqrt(1 - alpha_bar_t) * epsilon

        # 模型输出与 epsilon 同 shape；MSE 对 batch、通道和空间轴平均。
        predicted_noise = model(xt, t)
        loss = mean((epsilon - predicted_noise) ** 2)

    论文使用一基时间 \(1,\ldots,T\)，许多代码数组使用零基索引
    \(0,\ldots,T-1\)。上面用 t - 1 访问数组，是为了明确区分数学时间与数组位置。
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 算法 2｜反向采样为何最后一步不加噪

        x = randn([B, C, H, W])  # x_T
        for t in range(T, 0, -1):
            alpha_t = alphas[t - 1]
            alpha_bar_t = alpha_bars[t - 1]
            beta_t = betas[t - 1]

            predicted_noise = model(x, t)
            mean = (
                x - beta_t * predicted_noise / sqrt(1 - alpha_bar_t)
            ) / sqrt(alpha_t)

            # t>1 才从 p_theta(x_{t-1}|x_t) 抽样；t=1 输出均值作为 x_0。
            z = randn_like(x) if t > 1 else zeros_like(x)
            x = mean + sigma_t * z

    若工程实现循环变量是零基 index，从 T-1 递减到 0，条件会相应写成
    index > 0。判断边界前必须先声明索引约定；否则“最后一步仍加噪”会污染样本。
    """)
    return


@app.cell
def _(mo):
    algorithm_check = mo.ui.radio(
        options={
            "训练必须先生成 x1,...,x(t-1)，再得到 xt": "iterate",
            "训练可用式 (4) 跳到随机 xt，采样仍需反向逐步走": "direct",
            "训练和采样都只需一步": "one",
        },
        label="哪句话准确概括算法 1 与算法 2？",
    )
    return (algorithm_check,)


@app.cell
def _(PAPERS, paper_evidence_block):
    paper_evidence_block(
        PAPERS["ddpm"],
        title="算法 1 与算法 2 为什么必须并排读？",
        original_quote="The complete sampling procedure, Algorithm 2, resembles Langevin dynamics.",
        chinese_translation="训练随机抽一个时刻并预测噪声；生成则从 x_T 开始逐步执行反向更新。",
        figure_path="assets/paper_figures/ddim/ho_algorithms_1_2.webp",
        figure_number="原文 Algorithms 1–2",
        figure_caption="左为随机时间步训练，右为从 T 到 1 的完整反向采样。",
        legend=(
            "Algorithm 1 第 3 行：t 在 1…T 中均匀抽样；不需要依次生成所有中间状态。",
            "Algorithm 1 第 4 行：epsilon 是监督目标；模型输入是一次闭式构造的 noisy sample。",
            "Algorithm 2 第 2 行：t 从 T 递减到 1；第 3 行在 t=1 令 z=0。",
            "Algorithm 2 第 4 行：均值更新与 sigma_t z 共同定义一步 reverse sampling。",
        ),
        learning_goal="逐行把论文的一基 t 映射到代码数组索引，并圈出训练可跳时刻、生成需倒序的位置。",
        evidence_boundary="算法框给出原论文程序，不保证任意代码库采用相同 variance、索引或 scheduler。",
    )
    return


@app.cell
def _(algorithm_check, mo):
    if algorithm_check.value is None:
        _message = "先选一个。重点比较训练已知 x0 与生成未知 x0。"
        _kind = "warn"
    elif algorithm_check.value == "direct":
        _message = "正确。闭式 q(x_t|x_0) 让训练随机抽时刻；原始 DDPM 采样器仍是 T 次反向 Markov 更新。"
        _kind = "success"
    else:
        _message = "不正确。式 (4) 消除了训练的顺序依赖，但原始反向生成仍需逐步执行算法 2。"
        _kind = "danger"
    mo.callout(mo.md(_message), kind=_kind)
    return


@app.cell
def _(mo):
    mo.callout(
        mo.md(r"""
        ## 错误与反例｜读 DDPM 最常见的五次偷换

        1. **把 \(\alpha_t\) 与 \(\bar\alpha_t\) 混用。**  
           前者是单步保留率，后者是从 1 到 t 的累计乘积。式 (4) 必须使用后者。

        2. **说 q(x_{t-1}|x_t) 本身可直接计算。**  
           训练中可计算的是额外条件于 \(x_0\) 的
           \(q(x_{t-1}|x_t,x_0)\)；生成时没有 \(x_0\)。

        3. **省略 simple loss 的“重新加权”。**  
           去掉随时间变化的正权重，会改变共享有限网络跨时刻的优化重点。

        4. **把固定 variance 说成 DDPM 永远的定义。**  
           原始论文讨论固定选择；Improved DDPM 进一步学习反向方差。

        5. **混淆论文时间与代码数组索引。**  
           论文 t=1 是最后一次生成 x₀ 的更新；零基循环 index=0 才对应这个位置。
        """),
        kind="danger",
    )
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 如何读实验图｜主张、证据与后续修正

    - 原始 DDPM 的样本图支持其在指定数据集和指标上的生成质量主张；
    - 表格中的 negative log-likelihood 与 FID 衡量不同方面，不能互相替代；
    - 消融实验支持作者选择 \(L_{\rm simple}\) 的经验判断，不把它升级成数学定理；
    - Nichol & Dhariwal (2021) 的 Improved DDPM 学习 reverse variance，并报告
      混合目标与更少采样步数的效果。这是后续改进，不应倒写进 2020 年原论文贡献。

    阅读论文时建议给每条结论贴标签：**定义、推导、算法选择、实验观察、后续解释**。
    同一句“有效”在五类标签中的证据强度完全不同。
    """)
    return


@app.cell
def _(exercise_block, mo):
    mo.vstack(
        [
            mo.md("## 学习检测｜闭卷重建 DDPM"),
            exercise_block(
                understanding=(
                    "为什么训练能随机抽一个 t，而原始 DDPM 生成仍要从 T 走到 1？",
                    "训练知道 x0，可用式 (4) 闭式构造任意 xt；生成没有 x0，只能用学到的 p_theta(x_{t-1}|x_t) 逐步产生前一状态。",
                ),
                calculation=(
                    r"若 \(\alpha_1=0.9,\alpha_2=0.8\)，求 \(\bar\alpha_2\) 与 \(q(x_2|x_0)\) 的方差。",
                    r"\(\bar\alpha_2=0.9\times0.8=0.72\)，所以条件方差为 \(1-0.72=0.28\)。不能误写成 \(1-\alpha_2=0.2\)。",
                ),
                coding=(
                    "解释图像 batch 中 alpha_bars[t] 为什么要 reshape 为 [B,1,1,1]。",
                    "每个样本有自己的时间标量，但同一样本的所有通道和像素共享该缩放。reshape 后广播沿 C、H、W 复制；若保留 [B]，它可能错误地与最后一轴对齐或直接 shape 报错。",
                ),
                exploration=(
                    "调大末端 beta，观察 alpha_bar 曲线和点云。怎样判断 schedule 太激进？",
                    "观察相邻时刻的信号保留率是否骤降、较早 t 是否已丢失结构，以及反向网络是否需一次修正过大。图只能提示难度；最终还需训练损失、样本质量与 likelihood 等实验。",
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
            "式 (4) 来自独立高斯噪声的方差传播与累计乘积，使训练可直接跳到任意时刻。",
            "真实 posterior 需要同时条件于 x_t 与 x_0；生成时用神经网络替代不可见的 x_0 信息。",
            "epsilon 参数化把 Gaussian 均值 KL 变成带权噪声 MSE，而 L_simple 又主动去掉时间权重。",
            "算法 1 的随机时间训练与算法 2 的逐步反向采样必须结合索引约定阅读。",
            "原始 DDPM 与 Improved DDPM 的贡献应分开归属。",
        ),
        bridge=(
            "下一章进入 DDIM：既然训练得到的是跨噪声尺度的预测场，反向生成是否一定要沿 "
            "原始随机 Markov chain 行走？这个问题将引出非 Markov 构造与确定性采样。"
        ),
    )
    return


if __name__ == "__main__":
    app.run()
