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
    import numpy as np
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
    import matplotlib.pyplot as plt

    return (
        COLORS,
        PAPERS,
        PAPER_GUIDES,
        derivation_map,
        exercise_block,
        intuition_and_rigor,
        mo,
        np,
        paper_evidence_block,
        paper_guide_footer,
        paper_guide_header,
        plt,
    )


@app.cell
def _(PAPER_GUIDES, paper_guide_header):
    GUIDE = PAPER_GUIDES["vae_autoencoder_lineage"]
    paper_guide_header(GUIDE, duration="75–110 分钟")
    return (GUIDE,)


@app.cell
def _(PAPERS, paper_evidence_block):
    paper_evidence_block(
        PAPERS["hinton_autoencoder"],
        title="深层自编码器的历史起点：先压缩，再重构",
        original_quote="High-dimensional data can be converted to low-dimensional codes by training a multilayer neural network with a small central layer.",
        chinese_translation="作者用一个中间层很窄的多层网络，把高维数据变成低维代码，再从代码重构输入。",
        figure_path="assets/paper_figures/vae/hinton_figure1_autoencoder.webp",
        figure_number="原文 Figure 1",
        figure_caption="从逐层预训练到展开为编码器—解码器，再用反向传播微调整个深层自编码器。",
        legend=(
            "左列 Pretraining：每个 RBM 逐层学习特征；这是 2006 年深层网络的训练背景，不是现代 AE 的必需步骤。",
            "中列 Unrolling：下半部是 encoder，上半部是 decoder；中央 30 维 Code layer 是瓶颈。",
            "右列 Fine-tuning：在预训练权重附近，用重构误差对整网反向传播。",
            "方框内数字是各层单元数；W 与 W 的转置表示论文初始化阶段使用对称权重。",
        ),
        learning_goal="沿图从输入走到 30 维代码再回到重构，并指出图中哪一步只保证重构、没有定义可采样的 p(z)。",
        evidence_boundary="该图直接展示论文采用的深层瓶颈与两阶段训练流程；它不说明任意低重构误差的 latent 都连续、可解释或适合从高斯先验随机采样。",
    )
    return


@app.cell
def _(mo):
    mo.callout(
        mo.md(r"""
        ## 先做一个判断，不要急着读摘要

        四个模型都可能画成 \(x\to z\to\hat x\)，但这张箭头图隐藏了关键差别：

        - 普通自编码器（Autoencoder）约束**压缩**；
        - 去噪自编码器（Denoising Autoencoder, DAE）约束**抗扰动恢复**；
        - 收缩自编码器（Contractive Autoencoder, CAE）约束**编码器局部敏感度**；
        - 变分自编码器（Variational Autoencoder, VAE）定义**概率模型与近似推断**。

        **先预测：** 如果普通自编码器把训练数据重构到零误差，为什么从
        \(z\sim\mathcal N(0,I)\) 随机抽一个点仍可能生成垃圾？请先写下一句话答案。
        """),
        kind="warn",
    )
    return


@app.cell
def _(mo):
    paper_focus = mo.ui.dropdown(
        options={
            "普通自编码器｜Hinton & Salakhutdinov (2006)": "ae",
            "去噪自编码器｜Vincent et al. (2008)": "dae",
            "收缩自编码器｜Rifai et al. (2011)": "cae",
            "变分自编码器｜Kingma & Welling (2014)": "vae",
        },
        value="普通自编码器｜Hinton & Salakhutdinov (2006)",
        label="选择一篇论文，查看它真正改变了什么",
    )
    return (paper_focus,)


@app.cell
def _(PAPERS, mo, paper_focus):
    _profiles = {
        "ae": ("hinton_autoencoder", "输入 x", "确定性代码 h(x)", "重构 x_hat", "最小化重构误差", "没有指定 p(z)，随机生成无依据"),
        "dae": ("denoising_autoencoder", "损坏输入 x_tilde", "确定性代码 h(x_tilde)", "恢复干净 x", "对损坏分布平均的重构误差", "训练噪声不是潜变量先验"),
        "cae": ("contractive_autoencoder", "输入 x", "局部稳定代码 h(x)", "重构 x_hat", "重构误差 + Jacobian 范数", "局部不变性不是联合概率模型"),
        "vae": ("aevb", "输入 x", "分布 q_phi(z|x)", "似然 p_theta(x|z)", "最大化 ELBO", "通过 p(z) 定义生成路径"),
    }
    _paper_key, _input, _latent, _output, _objective, _boundary = _profiles[paper_focus.value]
    _paper = PAPERS[_paper_key]
    mo.vstack(
        [
            paper_focus,
            mo.md(
                f"""
                ### {_paper.chinese_title}

                **原题：** [{_paper.title}]({_paper.url})  
                **作者与版本：** {_paper.authors} · {_paper.venue_year}

                ```text
                {_input} → {_latent} → {_output}
                ```

                - **训练约束：** {_objective}
                - **不可越界的结论：** {_boundary}
                - **定点阅读：** {'；'.join(_paper.reading_targets)}。
                """
            ),
        ],
        gap=0.8,
    )
    return


@app.cell
def _(intuition_and_rigor):
    intuition_and_rigor(
        r"""
        四篇论文像四种“整理仓库”的办法。普通 AE 只保证每件旧货都能找到货架；
        DAE 要求标签被弄脏后仍能找回；CAE 要求轻推货物时货架编号不要乱跳；
        VAE 进一步规定整座仓库的地址系统，让我们可以按统一地图随机挑一个地址。
        """,
        r"""
        确定性 AE 学习映射 \(h_\phi:x\mapsto z\) 与
        \(g_\theta:z\mapsto\hat x\)，
        目标通常只有经验重构风险。该目标既不要求经验代码分布
        \(q_{\rm emp}(z)\) 接近某个已知先验，也不要求 decoder 在训练代码之间表现良好。

        VAE 则声明联合分布
        \(p_\theta(x,z)=p(z)p_\theta(x\mid z)\)，并用
        \(q_\phi(z\mid x)\) 近似难算的后验。于是“先从 \(p(z)\) 抽样，再从
        \(p_\theta(x\mid z)\) 生成”是模型定义的一部分，而不是训练后临时添加的操作。
        """,
    )
    return


@app.cell
def _(derivation_map, mo):
    mo.vstack(
        [
            mo.md("## 目标函数对照｜相似的网络外形，不同的数学承诺"),
            derivation_map(
                ["确定重构目标", "加入输入损坏或局部导数约束", "检查仍缺少什么", "声明 p(x,z)", "用 ELBO 学习概率模型"]
            ),
            mo.md(r"""
            | 模型 | 代表目标 | 随机量在哪里 | 能否直接从已知 prior 生成 |
            |---|---|---|---|
            | AE | \(\ell(x,g(h(x)))\) | 通常没有 | 否 |
            | DAE | \(\mathbb E_{\tilde x\sim c(\tilde x\mid x)}\ell(x,g(h(\tilde x)))\) | 人为损坏输入 | 否 |
            | CAE | \(\ell(x,g(h(x)))+\lambda\|J_h(x)\|_F^2\) | 通常没有 | 否 |
            | VAE | \(-\mathbb E_q\log p_\theta(x\mid z)+KL(q_\phi(z\mid x)\|p(z))\) | \(q_\phi(z\mid x)\) 与生成模型 | 是 |

            这里的“否”不是说 AE 永远不能被改造成生成模型，而是说**这些目标本身**没有
            提供 VAE 那样的先验采样保证。若额外拟合代码分布或加入生成机制，那已经增加了新模型假设。
            """),
        ],
        gap=0.8,
    )
    return


@app.cell
def _(mo):
    latent_gap = mo.ui.slider(
        0.5,
        4.0,
        value=2.6,
        step=0.1,
        show_value=True,
        label="普通 AE 两团训练代码之间的间隔",
    )
    sample_seed = mo.ui.slider(0, 20, value=7, step=1, show_value=True, label="重新采样 seed")
    return latent_gap, sample_seed


@app.cell
def _(COLORS, latent_gap, mo, np, plt, sample_seed):
    _rng = np.random.default_rng(sample_seed.value)
    _centers = np.array([-latent_gap.value, latent_gap.value])
    _codes = np.concatenate([
        _rng.normal(_centers[0], 0.23, 100),
        _rng.normal(_centers[1], 0.23, 100),
    ])
    _prior_samples = _rng.normal(0.0, 1.0, 160)
    _distance_to_code = np.min(np.abs(_prior_samples[:, None] - _codes[None, :]), axis=1)
    _unsupported = _distance_to_code > 0.55

    _fig, _axes = plt.subplots(2, 1, figsize=(9, 5.2), sharex=True)
    _axes[0].scatter(_codes, np.zeros_like(_codes), s=16, alpha=.55, color=COLORS["data"])
    _axes[0].set_ylabel("训练代码")
    _axes[0].set_title("示意实验：重构良好并不约束代码之间的空洞")
    _axes[1].scatter(
        _prior_samples[~_unsupported],
        np.zeros((~_unsupported).sum()),
        s=18,
        alpha=.65,
        color=COLORS["success"],
        label="靠近训练代码",
    )
    _axes[1].scatter(
        _prior_samples[_unsupported],
        np.zeros(_unsupported.sum()),
        s=22,
        alpha=.75,
        color=COLORS["danger"],
        label="落入代码空洞",
    )
    _axes[1].set_ylabel("N(0,1) 抽样")
    _axes[1].set_xlabel("一维 latent")
    _axes[1].legend(fontsize=8)
    _fig.tight_layout()

    mo.vstack(
        [
            mo.md(fr"""
            ## 交互反例｜给普通 AE 强行配一个标准高斯 prior

            当前有 **{100 * _unsupported.mean():.1f}%** 的 prior 样本距离任何训练代码超过
            0.55。Decoder 从未在这些位置被重构目标监督，输出可能没有意义。

            这是一维示意实验，不是“所有普通 AE 都会失败”的证明；它构造了一个反例，
            足以否定“重构好就必然能从任意指定 prior 生成”的推论。
            """),
            mo.hstack([latent_gap, sample_seed], widths="equal"),
            _fig,
        ],
        gap=0.8,
    )
    return


@app.cell
def _(mo):
    sensitivity = mo.ui.slider(0.2, 3.0, value=1.4, step=0.1, show_value=True, label="编码器斜率 w")
    return (sensitivity,)


@app.cell
def _(COLORS, mo, np, plt, sensitivity):
    _x = np.linspace(-3, 3, 500)
    _h = np.tanh(sensitivity.value * _x)
    _dh = sensitivity.value * (1 - _h**2)
    _penalty = float(np.mean(_dh**2))
    _fig, _axes = plt.subplots(1, 2, figsize=(9, 3.5))
    _axes[0].plot(_x, _h, color=COLORS["model"])
    _axes[0].set_title("编码 h(x)=tanh(wx)")
    _axes[1].plot(_x, _dh**2, color=COLORS["prior"])
    _axes[1].fill_between(_x, 0, _dh**2, alpha=.2, color=COLORS["prior"])
    _axes[1].set_title("局部敏感度 |dh/dx|²")
    _fig.tight_layout()
    mo.vstack(
        [
            mo.md(fr"""
            ## 收缩自编码器公式拆解｜Jacobian 惩罚到底惩罚什么

            一维时 Jacobian 就是导数。若 \(h(x)=\tanh(wx)\)，则
            \(h'(x)=w(1-\tanh^2(wx))\)。当前网格平均平方导数为 **{_penalty:.3f}**。
            CAE 惩罚它，是要求输入发生小变化 \(\Delta x\) 时，
            \(\Delta h\approx h'(x)\Delta x\) 不要过大。

            它约束局部几何，但没有定义 \(p(z)\)、\(p(x\mid z)\) 或 ELBO；因此不能把
            “更稳健的表示”直接翻译成“已经得到概率生成模型”。
            """),
            sensitivity,
            _fig,
        ],
        gap=0.8,
    )
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 代码对照｜同一个 encoder–decoder 外壳，训练信号来自不同位置

    ```python
    # 普通 AE：只要求当前样本能被重构。
    z = encoder(x)                         # [B, D]
    x_hat = decoder(z)                     # [B, input_dim]
    loss_ae = mse(x_hat, x)

    # DAE：随机性用于损坏输入；目标仍是干净 x。
    x_tilde = corrupt(x)                   # [B, input_dim]
    loss_dae = mse(decoder(encoder(x_tilde)), x)

    # CAE：J 是每个样本的 encoder Jacobian [D, input_dim]。
    # 下面是概念写法；真实逐样本 Jacobian 应使用 torch.func 等工具高效计算。
    loss_cae = loss_ae + lam * J.square().sum(dim=(-2, -1)).mean()

    # VAE：encoder 输出的是分布参数，不是单个代码；prior 进入目标。
    mu, logvar = encoder(x)                # 两者 [B, D]
    std = torch.exp(0.5 * logvar)
    z = mu + std * torch.randn_like(std)
    loss_vae = recon_loss + kl_to_standard_normal(mu, logvar)
    ```

    **删除/误改会怎样：** DAE 若把目标也换成损坏的 `x_tilde`，就失去去噪任务；
    CAE 若对 decoder Jacobian 罚款，含义已经改变；VAE 若删除 KL，便不再约束
    \(q_\phi(z\mid x)\) 与指定 prior 的关系。
    """)
    return


@app.cell
def _(mo):
    mo.callout(
        mo.md(r"""
        ## 阅读博客时需要补上的两个严谨限定

        1. “网络里有噪声”不足以定义 VAE。DAE 的损坏变量服务于去噪目标；VAE 的潜变量
           属于联合概率模型，并伴随后验推断。
        2. “VAE 是 regularized autoencoder”只是一种外形直觉。更准确的研究定位是：
           VAE 是用神经网络参数化生成模型与变分近似的潜变量模型；AE 式结构来自摊销推断。

        Lilian Weng 的文章适合建立谱系，但历史与公式主张应回到上述四篇原文核对。
        """),
        kind="info",
    )
    return


@app.cell
def _(exercise_block, mo):
    mo.vstack(
        [
            mo.md("## 合上资料后的学习检测"),
            exercise_block(
                (
                    "为什么普通 AE、DAE、CAE 即使都有 encoder–decoder，也不能仅凭各自目标就宣称可以从标准高斯 prior 采样？",
                    "它们的目标没有声明并约束一个已知的 `p(z)`，也没有保证 decoder 在经验训练代码之外的 latent 点上有意义。VAE 则把 `p(z)` 与 `p_theta(x|z)` 写入联合概率模型，并用 ELBO 连接数据、后验近似与 prior。",
                ),
                (
                    r"一维 CAE 取 (h(x)=2x)，单个样本的 Jacobian 平方范数是多少？若正则系数为 0.1，它给 loss 增加多少？",
                    r"一维 Jacobian 是导数 (h'(x)=2)，平方 Frobenius 范数为 (2^2=4)；乘 0.1 后增加 (0.4)。这项只衡量编码器局部敏感度。",
                ),
                (
                    "代码 `loss = mse(decoder(encoder(x_tilde)), x_tilde)` 原本想实现 DAE。指出错误并改正。",
                    "目标错误地使用了损坏输入。应写成 `loss = mse(decoder(encoder(x_tilde)), x)`：模型读取 `x_tilde`，但学习恢复干净 `x`。",
                ),
                (
                    "把代码间隔滑到最大，再改变 seed。先预测 prior 样本落入空洞的比例怎样变化，并说明该实验能证明与不能证明什么。",
                    "间隔变大通常使标准高斯样本离两团训练代码更远，空洞比例上升。它以反例说明重构目标不保证任意 prior 可用；它不能证明所有 AE 的实际代码都长这样，也不能比较真实模型生成质量。",
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
        [
            "AE、DAE、CAE 与 VAE 可以共享网络外形，却对数据、扰动、局部几何和概率分布作出不同承诺。",
            "好的重构只约束训练代码附近；没有已知 prior 时，随机生成路径并未自动成立。",
            "AEVB 的关键变化不是再加一种正则项，而是引入联合概率模型、近似后验和可优化的 ELBO。",
        ],
        bridge="现在已经知道 VAE 与早期自编码器谱系的联系和断点。下一篇导读回到 AEVB 原文，逐式核对这个概率模型如何变成可训练代码。",
    )
    return


if __name__ == "__main__":
    app.run()
