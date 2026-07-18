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
    GUIDE = PAPER_GUIDES["ddim_original"]
    paper_guide_header(GUIDE, duration="100–150 分钟，可分两次完成")
    return (GUIDE,)


@app.cell
def _(mo):
    mo.callout(
        mo.md(r"""
        ## 先做判断｜DDIM 不是“把 DDPM 的随机项删掉”

        如果只把 DDPM 更新里的随机噪声强行设为零，我们尚未说明新的过程：

        - 是否仍与训练时的噪声预测目标相容；
        - 每个 \(x_t\) 是否仍有正确的边缘分布；
        - 跨过若干时刻时应该使用什么系数。

        DDIM 的核心不是一个代码开关，而是先构造一族与 DDPM 共享边缘分布和代理训练目标的
        非马尔可夫推断过程，再从中得到确定性或随机性可调的生成过程。
        """),
        kind="warn",
    )
    return


@app.cell
def _(PAPERS, paper_evidence_block):
    paper_evidence_block(
        PAPERS["ddim"],
        title="两张图模型只改了哪些依赖关系？",
        original_quote=(
            "We generalize DDPMs via a class of non-Markovian diffusion processes "
            "that lead to the same training objective."
        ),
        chinese_translation="作者把 DDPM 的马尔可夫前向过程推广为一族非马尔可夫过程，但保留同一个训练目标。",
        figure_path="assets/paper_figures/ddim/figure1_graphical_models.webp",
        figure_number="原文 Figure 1",
        figure_caption="左：DDPM 的马尔可夫推断图；右：DDIM 的非马尔可夫推断图。",
        legend=(
            "圆节点 x₃→x₂→x₁→x₀：生成方向由噪声走向数据。",
            "左图灰色虚线 q(x_t|x_{t-1})：每一步只依赖前一状态，属于 Markov chain。",
            "右图长虚线连接 x₀：条件 q(x_{t-1}|x_t,x₀) 可直接依赖原始数据，因此推断联合分布不再要求马尔可夫。",
            "两图都使用同一个 pθ 预测反向更新；图没有声称两条联合路径分布完全相同。",
        ),
        learning_goal="只比较条件依赖结构：相同的单时刻边缘分布，不等于相同的联合轨迹分布。",
        evidence_boundary="Figure 1 是概率图定义，不是速度实验；它本身不能证明采样更快或图像更好。",
    )
    return


@app.cell
def _(derivation_map, mo):
    mo.vstack(
        [
            mo.md("## 从共享边缘到通用 DDIM 更新"),
            derivation_map(
                [
                    "保留 q(x_t|x_0) 的 Gaussian 边缘",
                    "由 epsilon_theta 估计 x_0",
                    "选择较早时刻 alpha_bar_s",
                    "把 x_s 分成预测数据、方向项与随机项",
                    "用 sigma 控制随机性",
                    "sigma=0 得确定性 DDIM",
                ]
            ),
            mo.md(r"""
            已知
            \[
            x_t=\sqrt{\bar\alpha_t}x_0+\sqrt{1-\bar\alpha_t}\epsilon,
            \]
            网络先给出
            \[
            \hat x_0=
            \frac{x_t-\sqrt{1-\bar\alpha_t}\epsilon_\theta(x_t,t)}
                 {\sqrt{\bar\alpha_t}}.
            \]

            从 \(t\) 跳到更早的 \(s<t\) 时，可写成
            \[
            x_s=
            \sqrt{\bar\alpha_s}\hat x_0+
            \sqrt{1-\bar\alpha_s-\sigma_{t\to s}^2}\,
            \epsilon_\theta(x_t,t)+
            \sigma_{t\to s}z,
            \quad z\sim\mathcal N(0,I).
            \]

            三项依次是**预测数据项、朝向当前噪声的方向项、新增随机项**。若
            \(\sigma_{t\to s}=0\)，给定 \(x_t\) 和网络后更新确定；但网络误差、
            离散步长和浮点误差仍然存在。
            """),
        ],
        gap=0.8,
    )
    return


@app.cell
def _(mo):
    ddim_eta = mo.ui.slider(0.0, 1.0, value=0.0, step=0.1, show_value=True, label="随机性 η")
    ddim_steps = mo.ui.slider(5, 100, value=20, step=5, show_value=True, label="采样步数")
    ddim_seed = mo.ui.slider(0, 30, value=7, step=1, show_value=True, label="随机 seed")
    return ddim_eta, ddim_seed, ddim_steps


@app.cell
def _(COLORS, ddim_eta, ddim_seed, ddim_steps, mo, np, plt):
    _rng = np.random.default_rng(ddim_seed.value)
    _steps = ddim_steps.value
    _eta = ddim_eta.value
    _time = np.linspace(1.0, 0.0, _steps + 1)
    _deterministic = np.column_stack([
        2.0 * np.cos(1.4 * np.pi * _time) * (1 - .25 * _time),
        2.0 * np.sin(1.4 * np.pi * _time) * (1 - .25 * _time),
    ])
    _noise = _rng.standard_normal((_steps + 1, 2))
    _noise[0] = 0
    _noise = np.cumsum(_noise, axis=0) / max(_steps, 1)
    _trajectory = _deterministic + .55 * _eta * _noise

    _fig, _axes = plt.subplots(1, 2, figsize=(10, 4))
    _axes[0].plot(
        _deterministic[:, 0], _deterministic[:, 1],
        color=COLORS["success"], linewidth=2.5, label="η=0 参考路径",
    )
    _axes[0].plot(
        _trajectory[:, 0], _trajectory[:, 1],
        color=COLORS["model"], linewidth=2, marker="o", markersize=3,
        label=f"当前路径 η={_eta:.1f}",
    )
    _axes[0].scatter(
        [_trajectory[0, 0]], [_trajectory[0, 1]],
        color=COLORS["prior"], s=65, marker="s", label="起点：高噪声",
    )
    _axes[0].scatter(
        [_trajectory[-1, 0]], [_trajectory[-1, 1]],
        color=COLORS["danger"], s=70, marker="*", label="终点：数据侧",
    )
    _axes[0].set_title("轨迹示意：η 改变路径随机性")
    _axes[0].set_aspect("equal")
    _axes[0].legend(fontsize=8, loc="best")

    _nfe = np.arange(5, 105, 5)
    _toy_error = 1.8 / _nfe + .012
    _axes[1].plot(_nfe, _toy_error, color=COLORS["data"], label="toy 离散误差")
    _axes[1].axvline(_steps, color=COLORS["danger"], linestyle="--", label=f"当前 NFE={_steps}")
    _axes[1].set(xlabel="网络调用次数 NFE", ylabel="toy 终点误差", title="步数—误差示意")
    _axes[1].legend(fontsize=8)
    _fig.tight_layout()

    mo.vstack(
        [
            mo.hstack([ddim_eta, ddim_steps, ddim_seed], widths="equal"),
            _fig,
            mo.callout(
                mo.md(r"""
                **这张教学图要看什么？**

                - 左图图例明确区分确定性参考路径、当前随机路径、噪声起点和数据终点；
                - 右图只演示“更密步长通常降低当前 toy 离散误差”，横轴是 NFE，不是论文 FID；
                - 这不是 DDIM 论文实验复现，不能用来声称现实图像质量提高。
                """),
                kind="info",
            ),
        ],
        gap=0.8,
    )
    return


@app.cell
def _(PAPERS, paper_evidence_block):
    paper_evidence_block(
        PAPERS["ddim"],
        title="采样更快的证据应怎样读？",
        original_quote="DDIMs can produce high quality samples 10× to 50× faster in terms of wall-clock time.",
        chinese_translation="作者在其硬件、数据集和实现中报告：DDIM 能以更少步骤显著缩短生成时间。",
        figure_path="assets/paper_figures/ddim/figure4_speed_quality.webp",
        figure_number="原文 Figure 4",
        figure_caption="单张 Nvidia 2080 Ti 上生成 50k 图像的时间，并展示不同采样步数的样本。",
        legend=(
            "左上蓝线：CIFAR-10；右上红线：Bedroom。纵轴 hours 与横轴 steps 都使用对数刻度。",
            "曲线上的小图：相应步数的代表样本，不是误差条。",
            "下方行：同一实验中不同采样步数的视觉结果；步数增加同时增加时间。",
            "图中的硬件、模型和 50k 样本规模属于论文实验设置，不能直接换算为当前电脑耗时。",
        ),
        learning_goal="同时读取时间成本与样本外观，不只看“快”或“好”其中一个词。",
        evidence_boundary="该图支持特定实现中的经验速度—质量折中，不证明任意 scheduler、网络或数据集都获得相同比例加速。",
    )
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 从确定性采样到 inversion｜可逆性为什么只是近似

    当 \(\eta=0\) 时，给定网络和离散时间表，采样更新是确定函数。这使我们可以尝试把
    真实图像沿相反方向送到高噪声 latent，再从该 latent 重构或编辑。

    但“确定函数”不自动等于“数值双射”：

    1. 反演使用的噪声预测通常在相邻时刻近似复用；
    2. 大步长会积累局部截断误差；
    3. classifier-free guidance 会改变向量场；
    4. 文本条件和无条件分支不一定同时重构真实图像。

    Null-text Inversion 是后续论文，不是 DDIM 原论文的一部分。它固定条件文本，
    逐时刻优化无条件文本嵌入，目标是改善大 CFG 下的真实图像重构。这里应把
    “DDIM 提供确定性轨迹”与“后续编辑方法如何校正轨迹”分开归属。
    """)
    return


@app.cell
def _(mo):
    ddim_claim = mo.ui.radio(
        options={
            "DDIM 与 DDPM 的完整联合轨迹分布相同": "joint",
            "DDIM 保留所需边缘与训练目标，但可改变联合依赖": "marginal",
            "DDIM 只是在 DDPM 代码中删除随机数": "delete",
        },
        label="哪句话最准确？",
    )
    return (ddim_claim,)


@app.cell
def _(ddim_claim, mo):
    if ddim_claim.value is None:
        _message, _kind = "先选择，再用 Figure 1 检查“边缘”和“联合路径”是否被混用。", "warn"
    elif ddim_claim.value == "marginal":
        _message, _kind = "正确。共享单时刻边缘和代理训练目标，不要求联合轨迹的条件依赖结构相同。", "success"
    else:
        _message, _kind = "不准确。DDIM 先定义非马尔可夫过程；确定性只是这族过程中的一个选择。", "danger"
    mo.callout(mo.md(_message), kind=_kind)
    return


@app.cell
def _(exercise_block, mo):
    mo.vstack(
        [
            mo.md("## 学习检测｜用原图和公式互相约束"),
            exercise_block(
                understanding=(
                    "为什么 Figure 1 能支持“换路径”，却不能支持“更快”？",
                    "Figure 1 只编码条件依赖关系，说明非马尔可夫联合构造是允许的；速度还取决于选取多少采样时刻和实际运行时间，需要 Figure 4 等实验。",
                ),
                calculation=(
                    r"若 \(\bar\alpha_s=0.81,\sigma=0.3\)，方向项系数是多少？",
                    r"方向项系数为 \(\sqrt{1-0.81-0.3^2}=\sqrt{0.10}\)。先检查定义域：\(1-\bar\alpha_s-\sigma^2\ge0\)。",
                ),
                coding=(
                    "确定性 DDIM 更新中是否还应调用 randn_like？",
                    "若 eta=0，则新增随机项系数为零，无需随机采样；但网络预测和数值计算仍然存在误差。实现还应检查时间索引与 alpha_bar 的 broadcasting。",
                ),
                exploration=(
                    "为什么 Figure 4 的加速比例不能直接套到你的机器？",
                    "论文使用特定 GPU、模型、实现和 50k 样本；本机的设备、batch、编译、内存和 scheduler 都会改变墙钟时间。可迁移的是比较方法：同时报告 NFE、硬件、样本量和质量指标。",
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
            "DDIM 先改变联合依赖结构，再得到确定性或随机性可调的采样器。",
            "共享训练目标与单时刻边缘，不等于共享完整联合轨迹分布。",
            "原文 Figure 1 解释结构，Figure 4 提供速度—质量经验；两类证据不能互相替代。",
            "确定性轨迹使 inversion 成为可能，但离散化、模型误差和 guidance 使严格可逆失效。",
        ),
        bridge=(
            "下一部分不再只问采样路径，而把整个扩散系统拆成六个可独立设计的旋钮："
            "预测什么、如何引导、在哪里扩散、用什么网络、怎样求解以及如何连续化。"
        ),
    )
    return


if __name__ == "__main__":
    app.run()
