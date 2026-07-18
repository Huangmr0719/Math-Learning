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
    from matplotlib.patches import Patch, Rectangle
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
        Patch,
        Rectangle,
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
    GUIDE = PAPER_GUIDES["diffusion_variants_landmarks"]
    paper_guide_header(GUIDE, duration="120–180 分钟，建议分三次完成")
    return (GUIDE,)


@app.cell
def _(mo):
    mo.callout(
        mo.md(r"""
        ## 先拆系统｜“扩散模型变体”不是一袋互不相关的名字

        一套扩散系统至少有六个相对独立的设计层：

        1. **预测参数化**：预测 \(\epsilon,x_0,v\) 或 score；
        2. **条件引导**：如何改变有条件与无条件分布的方向；
        3. **表示空间**：在像素还是压缩 latent 中扩散；
        4. **网络骨干**：U-Net、Transformer 或其他 denoiser；
        5. **数值求解器**：用多少 NFE、什么阶数和时间网格；
        6. **时间形式**：离散链、SDE 或 probability flow ODE。

        一篇论文可能同时改动多层，但阅读时必须逐层归因。否则很容易把质量提升全归功于
        一个名字，例如“Transformer 一定更好”或“latent diffusion 只是把图片缩小”。
        """),
        kind="warn",
    )
    return


@app.cell
def _(mo):
    design_knob = mo.ui.dropdown(
        options={
            "预测什么｜Progressive Distillation / EDM": "prediction",
            "如何控制｜Classifier Guidance / CFG": "guidance",
            "在哪里扩散｜Latent Diffusion": "space",
            "用什么网络｜U-Net / DiT": "backbone",
            "怎样快速走｜DPM-Solver / EDM": "solver",
            "怎样连续化｜Score-SDE": "continuous",
        },
        value="预测什么｜Progressive Distillation / EDM",
        label="选择一个设计旋钮",
    )
    return (design_knob,)


@app.cell
def _(COLORS, Patch, Rectangle, design_knob, mo, plt):
    _layers = [
        ("prediction", "预测参数化", COLORS["data"]),
        ("guidance", "条件引导", COLORS["model"]),
        ("space", "表示空间", COLORS["prior"]),
        ("backbone", "网络骨干", COLORS["warning"]),
        ("solver", "数值求解器", COLORS["success"]),
        ("continuous", "时间形式", COLORS["danger"]),
    ]
    _fig, _ax = plt.subplots(figsize=(10, 3.2))
    for _index, (_key, _label, _color) in enumerate(_layers):
        _alpha = 1.0 if _key == design_knob.value else .2
        _ax.add_patch(Rectangle((_index, 0), .88, 1, color=_color, alpha=_alpha))
        # 高亮色块上单独选择文字色；橙色/琥珀色用深字，其余深色用白字。
        # 未选中层透明度很低，统一使用深字即可。
        _text_color = (
            "#ffffff"
            if _alpha == 1.0 and _key in {"prediction", "space", "solver", "continuous"}
            else "#1f2933"
        )
        _ax.text(
            _index + .44,
            .5,
            _label,
            ha="center",
            va="center",
            rotation=90,
            fontsize=10,
            color=_text_color,
            weight="bold",
        )
        if _index < len(_layers) - 1:
            _ax.annotate("", xy=(_index + 1, .5), xytext=(_index + .88, .5),
                         arrowprops={"arrowstyle": "->", "color": "#64748b"})
    _ax.set(xlim=(-.15, 6), ylim=(-.08, 1.08), title="扩散系统六层图：高亮层是当前论文主要修改对象")
    _ax.axis("off")
    _ax.legend(
        handles=[
            Patch(facecolor=COLORS["data"], label="高亮：当前选择层"),
            Patch(facecolor="#94a3b8", alpha=.2, label="淡色：仍存在但本次不作为主变量"),
        ],
        loc="lower center", bbox_to_anchor=(.5, -0.3), ncol=2, fontsize=8,
    )
    _fig.tight_layout()

    _explanations = {
        "prediction": "同一个 x_t 可以换算出不同预测目标；改变参数化会改变数值尺度、损失权重和蒸馏稳定性。",
        "guidance": "CFG 不改 forward schedule，而是在采样时外推有条件与无条件预测的差方向。",
        "space": "LDM 把扩散主干移动到自编码器 latent；压缩损失与扩散误差成为两类独立瓶颈。",
        "backbone": "概率公式只要求输出 shape 正确；U-Net 与 DiT 提供不同的归纳偏置和计算缩放。",
        "solver": "模型给局部方向，solver 决定如何积分；比较效率时应固定 NFE，而不是只数时间步。",
        "continuous": "SDE/ODE 改写时间动力学；它不自动替换网络，仍需要 time-dependent score。",
    }
    mo.vstack(
        [
            design_knob,
            _fig,
            mo.callout(mo.md(_explanations[design_knob.value]), kind="info"),
            mo.md(
                "**图例与任务：**颜色只标识系统层，不编码论文优劣；本图要求你先判断论文修改哪一层，"
                "再讨论指标。箭头表示实现依赖，不表示左边在历史上必然早于右边。"
            ),
        ],
        gap=0.7,
    )
    return


@app.cell
def _(derivation_map, mo):
    mo.vstack(
        [
            mo.md("## 数学坐标｜四种预测不是四个互不相关的网络"),
            derivation_map(
                [
                    "写出 x_t = alpha_t x_0 + sigma_t epsilon",
                    "定义 v = alpha_t epsilon - sigma_t x_0",
                    "把二维线性变换写成矩阵",
                    "检查 alpha_t² + sigma_t² = 1",
                    "求逆得到 x_0 与 epsilon",
                    "比较不同 t 的目标尺度",
                ]
            ),
            mo.md(r"""
            在 variance-preserving 记号下，
            \[
            x_t=\alpha_t x_0+\sigma_t\epsilon,\qquad
            v_t=\alpha_t\epsilon-\sigma_t x_0,
            \quad \alpha_t^2+\sigma_t^2=1.
            \]
            因为变换矩阵
            \[
            \begin{bmatrix}\alpha_t&\sigma_t\\-\sigma_t&\alpha_t\end{bmatrix}
            \]
            是正交矩阵，所以
            \[
            x_0=\alpha_t x_t-\sigma_t v_t,\qquad
            \epsilon=\sigma_t x_t+\alpha_t v_t.
            \]

            这些是给定 schedule 时的代数等价参数化；有限网络、不同损失权重和优化过程并不
            因此完全等价。Progressive Distillation 使用 velocity parameterization 改善蒸馏，
            EDM 则把预条件、噪声分布、损失权重与采样器拆开审查。
            """),
        ],
        gap=0.8,
    )
    return


@app.cell
def _(PAPERS, paper_evidence_block):
    paper_evidence_block(
        PAPERS["classifier_free_guidance"],
        title="CFG 为什么会提高条件集中度，也可能损失覆盖？",
        original_quote="We jointly train a conditional and an unconditional diffusion model.",
        chinese_translation="同一训练体系同时学习有条件和无条件预测，采样时再混合二者。",
        figure_path="assets/paper_figures/diffusion_variants/cfg_figure2_gaussian_guidance.webp",
        figure_number="原文 Figure 2",
        figure_caption="三高斯混合上的引导强度实验；从左到右逐渐增加 guidance。",
        legend=(
            "最左：无引导的边缘混合密度，三个模式都较宽。",
            "从左到右：guidance strength 增大，每个条件模式更尖、更集中。",
            "深色表示密度更高，不表示图片质量分数。",
            "该图展示的是归一化 toy density，不是 ImageNet FID 曲线。",
        ),
        learning_goal="把 guidance 理解为改变分布集中度，而不是凭空添加新的类别信息。",
        evidence_boundary="三高斯图解释机制趋势；真实图像的质量—多样性权衡还需论文 Figure 4–5 等指标实验。",
    )
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### CFG 公式检查

    采用“scale=1 对应普通条件预测”的课程约定：
    \[
    \epsilon_{\rm cfg}
    =\epsilon_{\rm uncond}
    +w(\epsilon_{\rm cond}-\epsilon_{\rm uncond}).
    \]

    - \(w=0\)：无条件预测；
    - \(w=1\)：普通条件预测；
    - \(w>1\)：沿条件—无条件差向量外推。

    不同论文和代码库可能把 guidance strength 写成 \(w+1\) 或采用 score/x0 记号。
    比较数值前必须先检查公式，不能只比较参数名。
    """)
    return


@app.cell
def _(PAPERS, paper_evidence_block):
    paper_evidence_block(
        PAPERS["latent_diffusion"],
        title="LDM 到底把扩散搬到了哪里？",
        original_quote="We apply them in the latent space of powerful pretrained autoencoders.",
        chinese_translation="作者希望在感知压缩与语义压缩之间选择适合生成的潜空间，以降低扩散计算。",
        figure_path="assets/paper_figures/diffusion_variants/ldm_figure3_architecture.webp",
        figure_number="原文 Figure 3",
        figure_caption="左侧自编码器连接 pixel space 与 latent space；中间 U-Net 在 latent 中去噪；右侧条件经 cross-attention 注入。",
        legend=(
            "粉色 Pixel Space：编码器 E 把 x 变成 z，解码器 D 把 z 还原为图像。",
            "绿色 Latent Space：扩散与去噪 U-Net 都作用于 z_t，不直接作用于 RGB 像素。",
            "黄色 Q/K/V：cross-attention；Q 来自 U-Net 特征，K/V 来自条件表示。",
            "黑色 switch/concat 与灰色 skip：分别表示条件接入方式和 U-Net 跨尺度连接。",
        ),
        learning_goal="沿箭头完整复述训练与生成路径，并指出计算节省来自空间位置减少。",
        evidence_boundary="架构图说明模块连接，不证明压缩后没有信息损失；最终质量同时受自编码器和扩散器限制。",
    )
    return


@app.cell
def _(PAPERS, paper_evidence_block):
    paper_evidence_block(
        PAPERS["dit"],
        title="DiT 改的是概率模型还是 denoiser 骨干？",
        original_quote="We replace the commonly-used U-Net backbone with a transformer that operates on latent patches.",
        chinese_translation="作者保留 latent diffusion 的概率主干，只把处理带噪 latent 的网络换成 Transformer。",
        figure_path="assets/paper_figures/diffusion_variants/dit_figure3_architecture.webp",
        figure_number="原文 Figure 3",
        figure_caption="左：latent patchify 后进入 DiT blocks；中右：三种条件注入 block，论文重点使用 adaLN-Zero。",
        legend=(
            "左侧 Noise 与 Σ：网络预测的均值相关输出和协方差相关输出，不是两个输入数据集。",
            "Patchify：把 32×32×4 latent 切成 token；patch 越小，token 越多。",
            "黄色 Multi-Head Self-Attention：token 间交互；绿色 Pointwise Feedforward：逐 token MLP。",
            "蓝色 Scale/Shift：时间和类别条件通过 adaptive LayerNorm 调制。",
        ),
        learning_goal="把每个图块映射到第 24 章代码 shape，确认输出最终仍需还原到 latent 网格。",
        evidence_boundary="图 3 是架构定义；它不能单独证明 Transformer 比 U-Net 更好。",
    )
    return


@app.cell
def _(PAPERS, paper_evidence_block):
    paper_evidence_block(
        PAPERS["dit"],
        title="怎样阅读 DiT 的 scaling 图而不夸大结论？",
        original_quote="DiTs with higher Gflops consistently have lower FID.",
        chinese_translation="在论文测试的 DiT 模型族、训练预算和 ImageNet 设置中，更高 forward-pass Gflops 与更低 FID 相关。",
        figure_path="assets/paper_figures/diffusion_variants/dit_figure2_scaling.webp",
        figure_number="原文 Figure 2",
        figure_caption="左：不同规模 DiT 的 FID；右：带 guidance 的 DiT 与当时 U-Net/LDM 基线。",
        legend=(
            "纵轴 FID-50K：越低越好；不是百分比，也不是单张图片的分数。",
            "气泡面积：模型一次 forward 的 Gflops；面积而非直径表示计算量。",
            "左图颜色对应 DiT-S/B/L/XL，纵向多个点来自不同 patch size。",
            "右图比较带 guidance 的论文结果；训练数据、预算与实现差异会影响横向公平性。",
        ),
        learning_goal="同时读取纵轴、气泡面积和模型标签，复述“相关于计算量”的限定结论。",
        evidence_boundary="这是论文设计空间内的经验缩放关系，不是“任何数据和预算下 DiT 都优于 U-Net”的定理。",
    )
    return


@app.cell
def _(mo):
    scaling_check = mo.ui.radio(
        options={
            "气泡越大表示 FID 越差": "bubble_fid",
            "纵轴越低越好，气泡面积表示 Gflops": "axes",
            "横轴是训练样本数量": "samples",
        },
        label="不回看图注：DiT Figure 2 的轴和气泡分别表示什么？",
    )
    return (scaling_check,)


@app.cell
def _(mo, scaling_check):
    if scaling_check.value is None:
        _message, _kind = "先回答，再回到原图逐项核对纵轴、气泡面积和模型标签。", "warn"
    elif scaling_check.value == "axes":
        _message, _kind = "正确。FID-50K 越低越好；气泡面积表示 denoiser forward Gflops。", "success"
    else:
        _message, _kind = "不准确。不能把气泡面积、纵轴和横向模型排列解释成同一个量。", "danger"
    mo.callout(mo.md(_message), kind=_kind)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Solver 与连续时间｜不要把网络改进和积分改进混为一谈

    DPM-Solver 利用 diffusion ODE 的半线性结构构造专用高阶更新；EDM 也系统比较
    时间参数化、solver 与 stochastic churn。它们主要修改“怎样使用已训练网络走轨迹”。

    公平比较至少报告：

    - NFE（number of function evaluations，网络函数调用次数）；
    - solver 阶数与实际步数；
    - 时间网格和容差；
    - 同一个模型、同一个 guidance 与同一个评价协议。

    Score-SDE 则把有限噪声层提升为连续时间过程。它修改时间动力学的表达，但仍需
    time-dependent score network。下一篇导读会严格区分 reverse SDE 与
    probability flow ODE 的轨迹和边缘分布。
    """)
    return


@app.cell
def _(mo):
    layer_check = mo.ui.radio(
        options={
            "LDM 主要更换数值求解器": "solver",
            "DiT 主要更换去噪网络骨干": "backbone",
            "CFG 主要更换 forward schedule": "schedule",
        },
        label="哪条论文归因准确？",
    )
    return (layer_check,)


@app.cell
def _(layer_check, mo):
    if layer_check.value is None:
        _message, _kind = "先选择，再回到六层图定位论文真正修改的对象。", "warn"
    elif layer_check.value == "backbone":
        _message, _kind = "正确。DiT 保留扩散概率主干，把 latent denoiser 从 U-Net 换成 Transformer。", "success"
    else:
        _message, _kind = "不准确。LDM 主要改表示空间；CFG 主要改采样时的条件方向组合。", "danger"
    mo.callout(mo.md(_message), kind=_kind)
    return


@app.cell
def _(exercise_block, mo):
    mo.vstack(
        [
            mo.md("## 学习检测｜先归因，再评价"),
            exercise_block(
                understanding=(
                    "为什么不能说“Stable Diffusion 的质量来自 latent diffusion”就结束？",
                    "完整系统还包含自编码器、条件编码与 cross-attention、U-Net/Transformer、数据和训练、CFG、scheduler 与 solver。LDM 解释表示空间设计，但不能独占所有质量归因。",
                ),
                calculation=(
                    r"若 \(\alpha=0.8,\sigma=0.6,x_t=1,v=0.5\)，求 \(x_0\) 与 epsilon。",
                    r"因 \(\alpha^2+\sigma^2=1\)，\(x_0=0.8(1)-0.6(0.5)=0.5\)，\(\epsilon=0.6(1)+0.8(0.5)=1.0\)。",
                ),
                coding=(
                    "比较 Euler 20 步与二阶 solver 10 步时，至少记录哪个成本量？",
                    "记录 NFE。若二阶 solver 每步调用网络两次，两者都可能是 NFE=20；还应记录墙钟时间、设备和 solver 额外张量运算。",
                ),
                exploration=(
                    "用四张原图各写一句“能支持”和“不能支持”的结论。",
                    "参考维度：CFG 图支持 toy 分布更集中但不等于真实 FID；LDM 图支持模块连接但不证明无损；DiT 架构图定义网络但不证明优越；scaling 图支持论文设置内的相关趋势但不是普遍定理。",
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
            "扩散变体可以拆成预测、引导、空间、骨干、求解器和时间形式六层。",
            "原图必须先读图例、轴和模块，再判断它是定义图、机制图还是经验结果图。",
            "代数可换算的预测参数化，在有限网络和不同加权下不保证优化行为相同。",
            "论文贡献归因必须保留实验设置和后续组件，避免把整套系统功劳归给一个名字。",
        ),
        bridge=(
            "下一篇只深读 Score-SDE：我们将从 Fokker–Planck 的密度演化出发，核对 "
            "reverse-time SDE 和 probability flow ODE 为什么共享 marginals，却不共享单条轨迹。"
        ),
    )
    return


if __name__ == "__main__":
    app.run()
