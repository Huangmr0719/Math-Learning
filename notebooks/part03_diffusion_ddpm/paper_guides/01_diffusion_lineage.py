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
    from src.math_checks import finite_difference_gradient
    from src.teaching import (
        PAPERS,
        PAPER_GUIDES,
        derivation_map,
        exercise_block,
        intuition_and_rigor,
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
        finite_difference_gradient,
        intuition_and_rigor,
        mo,
        np,
        paper_guide_footer,
        paper_guide_header,
        plt,
    )


@app.cell
def _(PAPER_GUIDES, paper_guide_header):
    GUIDE = PAPER_GUIDES["diffusion_lineage"]
    paper_guide_header(GUIDE, duration="90–140 分钟，可分两次完成")
    return (GUIDE,)


@app.cell
def _(mo):
    mo.callout(
        mo.md(r"""
        ## 先猜结局｜两条看似不同的路

        2015 年的扩散概率模型（diffusion probabilistic model）在问：

        > 已知怎样一步步破坏数据，能否学习每一个反向转移？

        2019 年的噪声条件得分网络（Noise-Conditional Score Network, NCSN）在问：

        > 不计算密度本身，能否学习“往高密度方向走”的得分场？

        2020 年的去噪扩散概率模型（Denoising Diffusion Probabilistic Model,
        DDPM）说明，在高斯加噪下，这两条路线会合。

        **阅读前预测：** 网络预测噪声 \(\epsilon\)，为什么可能等价于学习一个
        指向高概率区域的向量？先写下直觉答案，读完后再回来修改。
        """),
        kind="warn",
    )
    return


@app.cell
def _(mo):
    lineage_focus = mo.ui.radio(
        options={
            "2015｜扩散链与非平衡热力学": "diffusion",
            "2005–2019｜得分匹配与多尺度噪声": "score",
            "2020｜DDPM 的会合": "ddpm",
        },
        value="2015｜扩散链与非平衡热力学",
        label="选择一条研究路线",
    )
    return (lineage_focus,)


@app.cell
def _(PAPERS, lineage_focus, mo):
    _profiles = {
        "diffusion": (
            "diffusion_thermodynamics",
            "固定 forward transition，再学习 reverse transition",
            "有限步 Markov chain 与变分下界",
            "尚未形成后来 DDPM 的简洁噪声预测训练",
        ),
        "score": (
            "ncsn",
            "在多个噪声尺度直接学习 ∇ₓ log pσ(x)",
            "去噪得分匹配与退火 Langevin dynamics",
            "采样器与离散扩散反向链尚未写成同一套 DDPM 记号",
        ),
        "ddpm": (
            "ddpm",
            "用高斯闭式把反向均值、ELBO 与噪声预测接起来",
            "随机时间步上的简单噪声 MSE",
            "simple loss 是重新加权的代理目标，不等于完整负 ELBO",
        ),
    }
    _key, _question, _tool, _boundary = _profiles[lineage_focus.value]
    _paper = PAPERS[_key]
    mo.vstack(
        [
            lineage_focus,
            mo.md(
                f"""
                ### {_paper.chinese_title}

                **原题：** [{_paper.title}]({_paper.url})  
                **作者与版本：** {_paper.authors} · {_paper.venue_year}

                - **它解决的问题：** {_question}
                - **核心工具：** {_tool}
                - **不能越界的结论：** {_boundary}
                - **定点阅读：** {'；'.join(_paper.reading_targets)}。
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
            mo.md("## 谱系地图｜不要把 DDPM 当成凭空出现的算法"),
            derivation_map(
                [
                    "Hyvärinen 2005：定义并拟合 score",
                    "Vincent 2011：得分匹配可以写成去噪目标",
                    "Sohl-Dickstein et al. 2015：学习逐步破坏的反向链",
                    "Song & Ermon 2019：在多个噪声尺度学习 score",
                    "Ho et al. 2020：高斯扩散与 epsilon 参数化",
                    "同一网络解释为反向均值和缩放后的 score",
                ]
            ),
            mo.md(r"""
            时间顺序不等于唯一影响关系：Vincent 的去噪得分匹配早于 2015 年扩散论文，
            而 NCSN 与 DDPM 是两条在相近时期发展的路线。这里的“会合”是数学结构的会合，
            不是说某一篇论文单独发明了所有组成部分。
            """),
        ],
        gap=0.8,
    )
    return


@app.cell
def _(intuition_and_rigor):
    intuition_and_rigor(
        r"""
        想象山谷里有一层看不见的雾，雾的浓度就是概率密度。得分向量不告诉你
        “这里有多浓”，只告诉你“往哪个方向走，雾会增加得最快”。加噪把尖锐山谷
        磨平；在不同噪声尺度学习方向，就能从远处逐渐走回数据密集区。
        """,
        r"""
        对可微且为正的密度 \(p_t(x)\)，得分函数（score function）定义为

        \[
        s_t(x)=\nabla_x\log p_t(x).
        \]

        若 \(x\in\mathbb R^d\)，则 \(s_t(x)\in\mathbb R^d\)。由于
        \(\log p_t(x)=\log \tilde p_t(x)-\log Z_t\)，而归一化常数 \(Z_t\)
        与 \(x\) 无关，所以 \(\nabla_x\log Z_t=0\)。score 可以避开难算的
        归一化常数，但这不表示分部积分所需的边界条件也可忽略。
        """,
    )
    return


@app.cell
def _(mo):
    noise_level = mo.ui.slider(
        0.05, 1.5, value=0.45, step=0.05, show_value=True, label="噪声标准差 σ"
    )
    lineage_seed = mo.ui.slider(
        0, 30, value=7, step=1, show_value=True, label="重新采样 seed"
    )
    show_vectors = mo.ui.checkbox(value=True, label="显示条件 score 向量")
    return lineage_seed, noise_level, show_vectors


@app.cell
def _(COLORS, lineage_seed, mo, noise_level, np, plt, show_vectors):
    _rng = np.random.default_rng(lineage_seed.value)
    _centers = np.array([[-1.8, 0.0], [1.8, 0.0]])
    _labels = _rng.integers(0, 2, size=500)
    _x0 = _centers[_labels] + 0.16 * _rng.standard_normal((500, 2))
    _epsilon = _rng.standard_normal(_x0.shape)
    _sigma = noise_level.value
    _xt = _x0 + _sigma * _epsilon
    _conditional_score = -(_xt - _x0) / (_sigma**2)

    _fig, _axes = plt.subplots(1, 2, figsize=(10.5, 4.1))
    _axes[0].scatter(_x0[:, 0], _x0[:, 1], s=8, alpha=.45, color=COLORS["data"], label="干净样本 x₀")
    _axes[0].set_title("干净数据：两个窄分布")
    _axes[1].scatter(_xt[:, 0], _xt[:, 1], s=8, alpha=.35, color=COLORS["model"], label="加噪样本 xₜ")
    if show_vectors.value:
        _pick = np.arange(0, len(_xt), 25)
        _direction = _conditional_score[_pick]
        _length = np.linalg.norm(_direction, axis=1, keepdims=True)
        _unit = _direction / np.maximum(_length, 1e-12)
        _axes[1].quiver(
            _xt[_pick, 0], _xt[_pick, 1], _unit[:, 0], _unit[:, 1],
            color=COLORS["danger"], alpha=.7, scale=14,
            label="条件 score 方向",
        )
    _axes[1].set_title(fr"加噪数据与条件 score：$\sigma={_sigma:.2f}$")
    for _ax in _axes:
        _ax.set_xlim(-4, 4)
        _ax.set_ylim(-3, 3)
        _ax.set_aspect("equal")
        _ax.grid(alpha=.15)
        _ax.legend(fontsize=8, loc="best")
    _fig.tight_layout()

    mo.vstack(
        [
            mo.hstack([noise_level, lineage_seed, show_vectors], widths="equal"),
            _fig,
            mo.md(
                fr"""
                **图例与观察任务：**蓝点是干净样本，橙点是加噪样本，红箭头只表示
                条件 score 的单位方向。请拖动 sigma，观察点云覆盖范围和箭头局部方向，
                不要用箭头长度判断密度大小。

                红箭头使用已知配对 \((x_0,x_t)\) 计算
                \(-\frac{{x_t-x_0}}{{\sigma^2}}\)。它是
                \(q_\sigma(x_t\mid x_0)\) 的**条件得分**。真正只看到 \(x_t\)
                的网络不知道是哪一个 \(x_0\) 产生它；平方误差的最优预测是条件平均，
                这才连接到边缘密度 \(q_\sigma(x_t)\) 的得分。此图是条件 Gaussian 的
                教学实验，不是 NCSN 或 DDPM 的论文指标复现。
                """
            ),
        ],
        gap=0.8,
    )
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 关键等式｜为什么预测噪声就是预测得分的缩放版

    DDPM 第 \(t\) 步的闭式加噪为

    \[
    q(x_t\mid x_0)
    =\mathcal N\!\left(
      x_t;\sqrt{\bar\alpha_t}x_0,(1-\bar\alpha_t)I
    \right).
    \]

    令
    \[
    x_t=\sqrt{\bar\alpha_t}x_0+\sqrt{1-\bar\alpha_t}\epsilon,
    \qquad \epsilon\sim\mathcal N(0,I).
    \]

    对 \(x_t\) 求对数密度梯度：

    \[
    \begin{aligned}
    \nabla_{x_t}\log q(x_t\mid x_0)
    &=-\frac{x_t-\sqrt{\bar\alpha_t}x_0}{1-\bar\alpha_t}\\
    &=-\frac{\epsilon}{\sqrt{1-\bar\alpha_t}}.
    \end{aligned}
    \]

    因而知道 \(\epsilon\) 就知道条件得分，二者只差一个已知时间缩放和负号。

    ### 最容易被省略的一步：条件得分不等于边缘得分

    网络输入通常只有 \(x_t,t\)，没有产生它的 \(x_0\)。对平方损失

    \[
    \mathbb E\|\epsilon-\epsilon_\theta(x_t,t)\|^2
    \]

    的总体最优解是
    \(\epsilon_\theta^*(x_t,t)=\mathbb E[\epsilon\mid x_t]\)。在可交换微分与
    积分等正则条件下，Fisher identity 给出

    \[
    \nabla_{x_t}\log q_t(x_t)
    =\mathbb E[
      \nabla_{x_t}\log q(x_t\mid x_0)\mid x_t
    ]
    =-\frac{\mathbb E[\epsilon\mid x_t]}{\sqrt{1-\bar\alpha_t}}.
    \]

    所以网络学到的是**边缘分布得分的缩放版**。不能把单个训练样本的
    \(-\epsilon/\sqrt{1-\bar\alpha_t}\) 与边缘 score 逐样本直接画等号。
    """)
    return


@app.cell
def _(finite_difference_gradient, mo, np):
    _mean = np.array([0.7, -0.4])
    _variance = 0.36
    _point = np.array([1.1, 0.2])

    def _log_q(_x):
        _delta = _x - _mean
        return float(-0.5 * np.sum(_delta**2) / _variance)

    _analytic = -(_point - _mean) / _variance
    _numeric = finite_difference_gradient(_log_q, _point)
    _error = float(np.max(np.abs(_analytic - _numeric)))
    mo.callout(
        mo.md(
            f"""
            ## 可执行核验｜高斯 score 的有限差分

            - 解析梯度：{np.round(_analytic, 8)}
            - 中心有限差分：{np.round(_numeric, 8)}
            - 最大绝对误差：{_error:.2e}

            这个实验核验了当前高斯例子和代码实现；严格结论仍来自上面的微分推导。
            """
        ),
        kind="success" if _error < 1e-6 else "danger",
    )
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 论文图解｜三张图应该分别看什么

    ### Sohl-Dickstein et al. (2015) 图 1

    Swiss roll 在 forward chain 中逐步失去弯曲结构，reverse chain 再从简单分布
    恢复。这张图支持的是**作者实现的有限实验能学到反向转移**，不是热力学类比本身
    保证任意神经网络都能正确生成。

    ### Song & Ermon (2019) 的多噪声尺度

    小噪声保留数据细节，但远离数据流形时 score 难以估计；大噪声覆盖空间更广，
    却抹去细节。退火 Langevin dynamics 从大噪声尺度逐步走向小噪声尺度。

    ### Ho et al. (2020) 图 2 与算法 1–2

    算法 1 不需要真的执行整条 forward chain：随机抽 \(t\)，直接构造 \(x_t\)，
    预测本次使用的 \(\epsilon\)。算法 2 才从 \(x_T\) 开始逐步反向采样。
    “训练可随机跳时刻、生成必须走反向链”是效率上的关键差别。
    """)
    return


@app.cell
def _(mo):
    lineage_claim = mo.ui.radio(
        options={
            "每个训练样本的条件 score 就等于边缘 score": "sample",
            "平方误差最优预测先取 E[epsilon|x_t]，再对应边缘 score": "expectation",
            "score 是评价图片质量的一个标量分数": "rating",
        },
        label="哪句话准确描述 epsilon 预测与 score 的关系？",
    )
    return (lineage_claim,)


@app.cell
def _(lineage_claim, mo):
    if lineage_claim.value is None:
        _message = "先选择。注意区分条件分布、边缘分布以及向量与标量。"
        _kind = "warn"
    elif lineage_claim.value == "expectation":
        _message = "正确。条件期望是从带配对监督的条件 score 过渡到只依赖 x_t 的边缘 score 的桥。"
        _kind = "success"
    else:
        _message = "不准确。score 是与 x 同维的对数密度梯度；逐样本条件 score 还要经过条件平均。"
        _kind = "danger"
    mo.callout(mo.md(_message), kind=_kind)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 公式到代码｜一行训练样本包含三层含义

        # alpha_bar_t 可广播为 [B, 1]。
        signal_scale = torch.sqrt(alpha_bar_t)
        noise_scale = torch.sqrt(1.0 - alpha_bar_t)

        # 对应 q(x_t | x_0) 的重参数化采样。
        xt = signal_scale * x0 + noise_scale * epsilon

        # 对 batch 与 feature 平均的噪声预测目标。
        predicted_noise = model(xt, t)
        loss = ((epsilon - predicted_noise) ** 2).mean()

        # 数学参数化转换，不是第二个网络。
        predicted_score = -predicted_noise / noise_scale

    \(x_0,\epsilon,x_t,\epsilon_\theta,s_\theta\) 的 shape 相同。噪声缩放接近零时
    除法会放大数值，因此必须明确时间范围和 schedule，不能在 \(t=0\) 随意除零。
    """)
    return


@app.cell
def _(mo):
    mo.callout(
        mo.md(r"""
        ## 错误与反例

        1. **把“得分”理解成图片质量评分。**  
           score 是 \(\nabla_x\log p_t(x)\)，是与数据同维的向量场。

        2. **说 DDPM 第一次提出逐步扩散思想。**  
           2015 年论文已经建立扩散概率模型；DDPM 的贡献是参数化、训练目标与高质量实现等组合。

        3. **把 conditional score 与 marginal score 逐样本等同。**  
           前者知道 \(x_0\)，后者只依赖 \(x_t\)；二者通过条件期望联系。

        4. **把 simple MSE 称为完整 ELBO 的逐项相等形式。**  
           \(L_{\rm simple}\) 去掉时间相关权重，是重新加权的代理目标。

        5. **把论文图当成普遍证明。**  
           图与指标支持特定设置的经验主张；概率恒等式才由推导证明。
        """),
        kind="danger",
    )
    return


@app.cell
def _(exercise_block, mo):
    mo.vstack(
        [
            mo.md("## 学习检测｜从论文谱系回到数学"),
            exercise_block(
                understanding=(
                    "为什么说扩散链路线和得分路线在 DDPM 中会合？",
                    "高斯反向均值可用 epsilon 参数化；epsilon 通过已知的负时间缩放对应条件 score。对只输入 x_t 的平方误差网络，最优预测是 E[epsilon|x_t]，它再对应边缘 score。",
                ),
                calculation=(
                    r"若 \(\bar\alpha_t=0.64\)，某一维 \(\epsilon=1.5\)，条件 score 是多少？",
                    r"\(\sqrt{1-\bar\alpha_t}=0.6\)，所以条件 score 为 \(-1.5/0.6=-2.5\)。负号表示抵消本次噪声的方向。",
                ),
                coding=(
                    "补全 predicted_score 的公式，并说明 shape。",
                    "predicted_score = -predicted_noise / sqrt(1-alpha_bar_t)。分母需从 [B] reshape 为可广播的 [B,1,...]；输出与数据及噪声 shape 相同。",
                ),
                exploration=(
                    "拖动 σ，观察 score 箭头和点云。为什么 σ 很小时箭头可能很长？",
                    r"条件 score 的大小含 \(1/\sigma^2\)。同样位移在窄高斯下更不可能，对数密度下降更快、梯度更大。应同时比较覆盖范围与局部细节。",
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
            "扩散模型的历史不是单线故事：反向 Markov chain 与得分估计是两条互相照亮的路线。",
            "高斯条件 score 等于负噪声除以噪声标准差；边缘 score 还需要条件期望。",
            "DDPM 的 simple loss 是有意重新加权的噪声预测目标，不能无条件写成完整负 ELBO。",
            "论文实验、数学恒等式和后来的统一解释必须分别标注证据层级。",
        ),
        bridge=(
            "下一篇导读只盯住 Ho、Jain 与 Abbeel 的 DDPM 原文，按式 (1)–(14) "
            "和算法 1–2 重建闭式前向过程、后验、训练目标与采样器。"
        ),
    )
    return


if __name__ == "__main__":
    app.run()
