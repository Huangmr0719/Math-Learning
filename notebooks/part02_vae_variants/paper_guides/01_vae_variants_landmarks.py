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
        mo,
        np,
        paper_evidence_block,
        paper_guide_footer,
        paper_guide_header,
        plt,
    )


@app.cell
def _(PAPER_GUIDES, paper_guide_header):
    GUIDE = PAPER_GUIDES["vae_variants_landmarks"]
    paper_guide_header(GUIDE, duration="150–210 分钟，建议分三次完成")
    return (GUIDE,)


@app.cell
def _(mo):
    mo.callout(
        mo.md(r"""
        ## 研讨任务｜不要按模型名字背诵

        第 8–13 章并不是六个互不相干的技巧。标准 VAE 暴露出六类不同缺口：

        ```text
        表示容量怎样控制？ → β‑VAE
        latent 为什么会被忽略？ → posterior-collapse 研究
        单样本下界太松怎么办？ → IWAE
        生成需要外部条件怎么办？ → CVAE
        需要离散表示怎么办？ → VQ‑VAE
        简单高斯后验不够灵活怎么办？ → normalizing flow
        ```

        本导读的目标不是把九篇论文压成摘要，而是学会判断：一篇“VAE 变体”究竟修改了
        **训练目标、优化动态、条件概率模型、潜变量类型，还是近似后验族**。
        """),
        kind="warn",
    )
    return


@app.cell
def _(mo):
    paper_station = mo.ui.dropdown(
        options={
            "容量与解耦｜β‑VAE + Burgess": "beta",
            "失败诊断｜Bowman + Lagging Inference": "collapse",
            "多样本下界｜IWAE": "iwae",
            "条件生成｜CVAE": "cvae",
            "离散表示｜VQ‑VAE + VQ‑VAE‑2": "vqvae",
            "灵活后验｜Normalizing Flows": "flow",
        },
        value="容量与解耦｜β‑VAE + Burgess",
        label="选择一个研究问题",
    )
    return (paper_station,)


@app.cell
def _(PAPERS, mo, paper_station):
    _stations = {
        "beta": {
            "papers": ("beta_vae", "beta_capacity"),
            "change": "修改目标函数中 KL 的权重或目标容量",
            "formula": r"L=R+\beta K\quad\text{或}\quad L=R+\gamma|K-C|",
            "warning": "较大的 β 或合适的 C 调度可能促进某些数据上的解耦，但不构成无监督识别真实生成因素的普遍保证。",
        },
        "collapse": {
            "papers": ("bowman_text_vae", "lagging_inference"),
            "change": "诊断优化动态与 decoder/latent 的竞争，不一定改模型定义",
            "formula": r"q_\phi(z|x)\approx p(z),\quad I_q(X;Z)\approx0",
            "warning": "强 decoder、inference lag、目标几何等是互补解释；不能把所有坍塌归因于单一机制。",
        },
        "iwae": {
            "papers": ("iwae",),
            "change": "把单样本 ELBO 换成 K 样本重要性加权下界",
            "formula": r"\mathcal L_K=E\log\frac1K\sum_{k=1}^K\frac{p(x,z_k)}{q(z_k|x)}",
            "warning": "下界随 K 趋紧是目标值结论；K 增大时 inference-network 梯度信噪比并不必然改善。",
        },
        "cvae": {
            "papers": ("cvae", "vae_tutorial"),
            "change": "把边缘似然改成条件似然，并明确条件先验、生成网络与识别网络",
            "formula": r"\log p_\theta(y|x)\ge E_q\log p_\theta(y|x,z)-KL(q_\phi(z|x,y)\|p_\theta(z|x))",
            "warning": "原论文用 x 表示条件、y 表示输出；本课程第 11 章采用 y 为条件、x 为生成数据，读式子时必须先做符号翻译。",
        },
        "vqvae": {
            "papers": ("vq_vae", "vq_vae_2"),
            "change": "把连续随机 latent 改成有限码本索引，并另学离散 prior",
            "formula": r"k^*=\arg\min_k\|z_e-e_k\|^2",
            "warning": "straight-through 是有偏替代梯度；VQ‑VAE‑2 的质量提升还依赖层次码本和强自回归 prior。",
        },
        "flow": {
            "papers": ("normalizing_flows",),
            "change": "保持 ELBO 框架，扩大 q(z|x) 的分布族",
            "formula": r"\log q_K(z_K|x)=\log q_0(z_0|x)-\sum_k\log|\det J_{f_k}(z_{k-1})|",
            "warning": "表达力提升伴随可逆性、log-determinant 计算与数值稳定成本。",
        },
    }
    _station = _stations[paper_station.value]
    _cards = []
    for _key in _station["papers"]:
        _paper = PAPERS[_key]
        _cards.append(
            mo.callout(
                mo.md(
                    f"""
                    **{_paper.chinese_title}**  
                    [{_paper.title}]({_paper.url})  
                    {_paper.authors} · {_paper.venue_year}

                    {_paper.role}

                    **定点阅读：** {'；'.join(_paper.reading_targets)}。
                    """
                ),
                kind="neutral",
            )
        )
    mo.vstack(
        [
            paper_station,
            mo.md(
                fr"""
                ### 这一站改了哪里？

                **修改对象：** {_station['change']}

                \[
                {_station['formula']}
                \]

                **结论边界：** {_station['warning']}
                """
            ),
            *_cards,
        ],
        gap=0.8,
    )
    return


@app.cell
def _(COLORS, mo, plt):
    _labels = ["β-VAE", "坍塌诊断", "IWAE", "CVAE", "VQ-VAE", "Flows"]
    _x = [0, 1, 0, 2, 3, 4]
    _y = [3, 2, 1, 3, 2, 1]
    _descriptions = ["目标权衡", "优化动态", "估计/下界", "条件模型", "latent 类型", "后验族"]
    _fig, _ax = plt.subplots(figsize=(9, 4.6))
    _ax.scatter(_x, _y, s=600, color=[COLORS["prior"], COLORS["danger"], COLORS["model"], COLORS["data"], COLORS["success"], COLORS["muted"]], alpha=.82)
    for _index, (_xi, _yi, _label, _description) in enumerate(zip(_x, _y, _labels, _descriptions)):
        # 第三个点使用橙色；小号白字在橙色上的对比度不足，因此改用深色。
        _text_color = "#1f2933" if _index == 2 else "#ffffff"
        _ax.text(_xi, _yi, f"{_label}\n{_description}", ha="center", va="center", fontsize=9, color=_text_color, weight="bold")
    _ax.set_xticks(range(5), ["目标/优化", "训练失败", "条件", "离散化", "分布变换"])
    _ax.set_yticks([])
    _ax.set_ylim(.3, 3.7)
    _ax.set_title("VAE 变体不是一条单轴排行榜：它们修改模型的不同部位")
    _ax.spines[["left", "right", "top"]].set_visible(False)
    _fig.tight_layout()
    mo.vstack(
        [
            mo.md("## 论文地图｜先定位修改对象，再比较实验结果"),
            _fig,
            mo.md(
                "同一数据集上的数值不能脱离目标比较：IWAE 追求更紧的 likelihood bound，"
                "β‑VAE 关注表示结构，CVAE 关注条件多样性，VQ‑VAE 关注离散代码。"
            ),
        ],
        gap=0.8,
    )
    return


@app.cell
def _(derivation_map, mo):
    mo.vstack(
        [
            mo.md("## 推导站一｜β 目标与容量目标不是一句口号"),
            derivation_map(["写约束问题", "构造 Lagrangian", "固定乘子后省略常数", "承认非凸优化限制", "再看容量 C 调度"]),
            mo.md(r"""
            设 \(R\) 为负期望对数似然，
            \(K=KL(q_\phi(z\mid x)\|p(z))\)。若希望 \(K\le C\)，Lagrangian 为

            \[
            \mathcal J(\theta,\phi,\lambda)=R+\lambda(K-C),
            \qquad \lambda\ge 0.
            \]

            对固定 \(\lambda,C\) 优化模型参数时，\(-\lambda C\) 是常数，留下
            \(R+\lambda K\)，形式上对应 β‑VAE。严格地说，“某个固定 β 目标”与“指定
            C 的约束问题”要在约束资格、强对偶、最优乘子等条件下才可谈最优解对应；
            非凸神经网络训练中不能直接宣布二者全局等价。

            Burgess 等转而使用
            \(R+\gamma|K-C|\)，并在训练中逐渐增加 C。这不是从 Higgins 目标做一次
            纯代数变形，而是新的训练设计：让 latent 先在低容量下学习主要因素，再逐步放宽。
            """),
        ],
        gap=0.8,
    )
    return


@app.cell
def _(mo):
    target_capacity = mo.ui.slider(0.0, 5.0, value=1.5, step=0.1, show_value=True, label="目标容量 C")
    capacity_penalty = mo.ui.slider(0.5, 8.0, value=3.0, step=0.5, show_value=True, label="容量惩罚 gamma")
    return capacity_penalty, target_capacity


@app.cell
def _(COLORS, capacity_penalty, mo, np, plt, target_capacity):
    _k = np.linspace(0, 5, 500)
    _reconstruction = (4.2 - _k) ** 2 / 4
    _objective = _reconstruction + capacity_penalty.value * np.abs(_k - target_capacity.value)
    _best = int(np.argmin(_objective))
    _fig, _ax = plt.subplots(figsize=(8.5, 3.8))
    _ax.plot(_k, _reconstruction, label="R(K): toy reconstruction", color=COLORS["data"])
    _ax.plot(_k, _objective, label="R + gamma|K-C|", color=COLORS["model"])
    _ax.axvline(target_capacity.value, linestyle="--", color=COLORS["prior"], label="target C")
    _ax.axvline(_k[_best], linestyle=":", color=COLORS["danger"], label="toy optimum")
    _ax.set_xlabel("KL / rate K")
    _ax.legend(fontsize=8)
    _fig.tight_layout()
    mo.vstack(
        [
            mo.md(fr"""
            ### 容量调度的 toy 实验

            当前 toy 最优 \(K\approx{_k[_best]:.2f}\)，目标容量为 {target_capacity.value:.2f}。
            当 gamma 足够大时最优点会贴近 C；gamma 有限时 reconstruction 收益仍可能把它拉开。
            这说明 \(|K-C|\) 是软惩罚，不是每一步都严格满足 \(K=C\)。
            """),
            mo.hstack([target_capacity, capacity_penalty], widths="equal"),
            _fig,
        ],
        gap=0.8,
    )
    return


@app.cell
def _(PAPERS, paper_evidence_block):
    paper_evidence_block(
        PAPERS["iwae"],
        title="IWAE 改的是训练下界，不是 encoder—decoder 拓扑",
        original_quote="the same architecture as the VAE, but which uses a strictly tighter log-likelihood lower bound",
        chinese_translation="IWAE 保留 VAE 的网络结构，改用从重要性加权得到的、更紧的对数似然下界。",
        learning_goal="在进入公式前先回答：K 个样本出现在哪个期望里？K=1 时为什么必须退化为普通 ELBO？",
        evidence_boundary="原文的严格变紧与收敛结论有其定理条件；它不意味着有限训练下更大的 K 必然带来更好的 encoder 梯度或更高的最终生成质量。",
    )
    return


@app.cell
def _(derivation_map, mo):
    mo.vstack(
        [
            mo.md("## 推导站二｜IWAE 为什么是下界，为什么 K=1 回到 ELBO"),
            derivation_map(["定义 importance weight", "对 K 个权重取平均", "平均值无偏估计 p(x)", "对 log 用 Jensen", "K=1 化为 ELBO"]),
            mo.md(r"""
            对 \(z_k\overset{\mathrm{iid}}{\sim}q_\phi(z\mid x)\)，定义
            \(w_k=p_\theta(x,z_k)/q_\phi(z_k\mid x)\)。只要支持集条件成立且期望存在，

            \[
            \mathbb E_q[w_k]
            =\int q(z\mid x)\frac{p(x,z)}{q(z\mid x)}\,dz
            =p(x).
            \]

            因而 \(\hat p_K(x)=K^{-1}\sum_k w_k\) 是 evidence 的无偏估计。由 Jensen：

            \[
            \mathcal L_K
            =\mathbb E\log\hat p_K(x)
            \le\log\mathbb E\hat p_K(x)
            =\log p(x).
            \]

            当 K=1，展开 \(\log w_1=\log p(x,z)-\log q(z\mid x)\)，正是标准 ELBO。
            原论文定理给出 \(\mathcal L_{K+1}\ge\mathcal L_K\)；其收敛结论需要额外条件，
            不能从“样本更多”四个字跳过可积性、支持集与权重尾部检查。
            """),
        ],
        gap=0.8,
    )
    return


@app.cell
def _(mo):
    iwae_samples = mo.ui.slider(1, 128, value=8, step=1, show_value=True, label="IWAE 样本数 K")
    iwae_seed = mo.ui.slider(0, 30, value=11, step=1, show_value=True, label="Monte Carlo seed")
    return iwae_samples, iwae_seed


@app.cell
def _(COLORS, iwae_samples, iwae_seed, mo, np, plt):
    _rng = np.random.default_rng(iwae_seed.value)
    _repetitions = 5000
    # Toy importance weights: log-normal with E[w]=1, so log p(x)=0 exactly.
    _log_w = _rng.normal(-0.5, 1.0, size=(_repetitions, iwae_samples.value))
    _max_log_w = np.max(_log_w, axis=1, keepdims=True)
    _log_mean_w = (
        _max_log_w[:, 0]
        + np.log(np.mean(np.exp(_log_w - _max_log_w), axis=1))
    )
    _estimate = float(np.mean(_log_mean_w))
    _standard_error = float(np.std(_log_mean_w, ddof=1) / np.sqrt(_repetitions))
    _fig, _ax = plt.subplots(figsize=(8.5, 3.6))
    _ax.hist(_log_mean_w, bins=60, density=True, color=COLORS["model"], alpha=.72)
    _ax.axvline(0, color=COLORS["danger"], linestyle="--", label="log p(x)=0")
    _ax.axvline(_estimate, color=COLORS["prior"], label="Monte Carlo L_K")
    _ax.set_xlabel("log(mean importance weight)")
    _ax.legend(fontsize=8)
    _fig.tight_layout()
    mo.vstack(
        [
            mo.md(fr"""
            ### IWAE 数值核验｜实验支持，不代替证明

            这里构造 \(w\sim\operatorname{{LogNormal}}(-1/2,1)\)，所以
            \(\mathbb E[w]=1\)、\(\log p(x)=0\)。当前 \(K={iwae_samples.value}\) 时，
            5000 次实验估计
            \(\mathcal L_K\approx{_estimate:.4f}\pm{2*_standard_error:.4f}\)
            （约两倍标准误）。
            拖大 K，估计通常向 0 靠近；单个 seed 下出现轻微非单调不反驳期望层面的定理。
            """),
            mo.hstack([iwae_samples, iwae_seed], widths="equal"),
            _fig,
        ],
        gap=0.8,
    )
    return


@app.cell
def _(derivation_map, mo):
    mo.vstack(
        [
            mo.md("## 推导站三｜CVAE 先翻译符号，再推条件 ELBO"),
            derivation_map(["固定条件 x", "写 log p(y|x)", "乘除 q(z|x,y)", "Jensen", "按条件图分解联合分布"]),
            mo.md(r"""
            Sohn 等原论文用 \(x\) 表示条件输入、\(y\) 表示结构化输出：

            \[
            p_\theta(y,z\mid x)
            =p_\theta(y\mid x,z)p_\theta(z\mid x).
            \]

            插入识别网络 \(q_\phi(z\mid x,y)\) 并用 Jensen：

            \[
            \log p_\theta(y\mid x)\ge
            \mathbb E_q\log p_\theta(y\mid x,z)
            -KL(q_\phi(z\mid x,y)\|p_\theta(z\mid x)).
            \]

            第 11 章为符合全课程习惯，用 \(x\) 表示要生成的数据、\(y\) 表示条件。
            对应关系是“论文 \(x\leftrightarrow\) 课程 \(y\)，论文
            \(y\leftrightarrow\) 课程 \(x\)”；公式结构没有变。训练可用
            \(q(z\mid x,y)\)，生成时真实输出未知，只能从条件 prior
            \(p(z\mid y)\) 抽样。这是读图 1(b)(c) 时最重要的两条路径。
            """),
        ],
        gap=0.8,
    )
    return


@app.cell
def _(PAPERS, paper_evidence_block):
    paper_evidence_block(
        PAPERS["vq_vae"],
        title="离散代码、码本与替代梯度在一张图中怎样分工",
        original_quote="the encoder network outputs discrete, rather than continuous, codes; and the prior is learnt rather than static",
        chinese_translation="VQ‑VAE 与连续 VAE 的两项核心差异，是 encoder 输出离散代码，而且 prior 需要另行学习。",
        figure_path="assets/paper_figures/vae/vq_vae_figure1_architecture.webp",
        figure_number="原文 Figure 1",
        figure_caption="左侧是 encoder—最近邻量化—decoder；右侧放大 embedding space 中的最近邻选择和红色替代梯度。",
        legend=(
            "绿色张量 z_e(x) 是 encoder 的连续输出；它尚不是送入 decoder 的离散表示。",
            "紫色向量 e_k 构成 codebook；最近邻规则把每个 z_e 位置替换为选中的 e_k。",
            "z_q(x) 是量化后的张量，作为 decoder 输入；索引序列还需要另学 prior 才能从零生成。",
            "右图红箭头是 straight-through 的替代梯度路径，不是 argmin 本身的真实导数。",
            "蓝色连线表示前向的数据与码本关系；图中的颜色是论文示意，不代表概率大小。",
        ),
        learning_goal="沿 forward 蓝色路径走一遍，再沿 backward 红色路径走一遍；指出 codebook、encoder 和 decoder 分别由哪项损失更新。",
        evidence_boundary="该图解释算法分工，却不能证明离散代码一定避免 posterior collapse，也不能把替代梯度当作离散 argmin 的无偏梯度。",
    )
    return


@app.cell
def _(derivation_map, mo):
    mo.vstack(
        [
            mo.md("## 推导站四｜VQ‑VAE 的三项损失与两阶段生成"),
            derivation_map(["encoder 输出 z_e", "最近邻选择 e_k", "ST 传 reconstruction 梯度", "两项 stop-gradient 分工", "另学离散 prior"]),
            mo.md(r"""
            为避免优化方向含混，把三项统一写成最小化形式：

            \[
            J=-\log p_\theta(x\mid z_q)
            +\|\operatorname{sg}[z_e]-e\|_2^2
            +\beta\|z_e-\operatorname{sg}[e]\|_2^2.
            \]

            原论文式 (3) 的排版写成 **log p 加两个正平方项**，正文又把第一项称作
            reconstruction loss，因此若直接把整式按统一最小化目标读取，会出现符号歧义。
            教学代码应明确采用 negative log-likelihood 加正惩罚；等价的最大化写法则是
            log-likelihood 减惩罚。第二项只把 codebook 拉向 encoder
            输出；第三项只把 encoder 拉向所选 code。原论文还给出 EMA 作为**替代的**码本
            更新方式，不能写成“所有 VQ‑VAE 都必须 EMA”。

            训练重构模型后，还要拟合离散索引的先验 \(p(k)\) 才能从零生成。VQ‑VAE‑2 又加入
            上下两层 codebook 与更强的自回归 prior；因此其高保真结果不能只归功于“把 latent
            离散化”。
            """),
        ],
        gap=0.8,
    )
    return


@app.cell
def _(mo):
    quantized_value = mo.ui.slider(-2.0, 2.0, value=0.4, step=0.05, show_value=True, label="encoder 输出 z_e")
    return (quantized_value,)


@app.cell
def _(COLORS, mo, np, plt, quantized_value):
    _codebook = np.array([-1.5, -0.4, 0.7, 1.6])
    _distances = (_codebook - quantized_value.value) ** 2
    _index = int(np.argmin(_distances))
    _selected = float(_codebook[_index])
    _fig, _ax = plt.subplots(figsize=(8.5, 2.6))
    _ax.scatter(_codebook, np.zeros_like(_codebook), s=180, color=COLORS["prior"], label="codebook")
    _ax.scatter([quantized_value.value], [.18], s=120, marker="x", color=COLORS["danger"], label="z_e")
    _ax.plot([quantized_value.value, _selected], [.18, 0], linestyle="--", color=COLORS["model"])
    _ax.set_yticks([])
    _ax.set_xlim(-2.2, 2.2)
    _ax.legend(fontsize=8)
    _fig.tight_layout()
    mo.vstack(
        [
            mo.md(fr"""
            ### VQ 边界实验

            当前选择 code **{_index}**，\(z_q={_selected:.2f}\)。穿过相邻 code 的中点时，
            索引会跳变；这正是 `argmin` 不能用普通路径导数训练的原因。Straight-through
            让 backward 近似走一条恒等路径，但没有改变 forward 的离散跳变。
            """),
            quantized_value,
            _fig,
        ],
        gap=0.8,
    )
    return


@app.cell
def _(PAPERS, paper_evidence_block):
    paper_evidence_block(
        PAPERS["normalizing_flows"],
        title="可逆变换怎样逐步雕刻近似后验",
        original_quote="a simple initial density is transformed into a more complex one by applying a sequence of invertible transformations",
        chinese_translation="正规化流从简单初始密度出发，连续应用一系列可逆变换，把它变成更复杂的近似后验。",
        figure_path="assets/paper_figures/vae/normalizing_flows_figure1.webp",
        figure_number="原文 Figure 1",
        figure_caption="planar flow 与 radial flow 在 K=1、2、10 步时，对单位高斯和均匀初始密度的形变效果。",
        legend=(
            "每一行固定同一个初始分布：上行为单位高斯，下行为二维均匀分布。",
            "第一列 q0 是变换前密度；之后分别展示 planar 与 radial 两类可逆变换。",
            "K 是串联的变换步数，不是 Monte Carlo 样本数，也不是 latent 维度。",
            "暖色表示密度较高、冷色表示较低；颜色只用于同一幅密度图内比较。",
            "形状更复杂来自逐步体积伸缩；每步仍需计算相应 Jacobian log-determinant。",
        ),
        learning_goal="比较 K=1 与 K=10，描述多步变换增加了什么形状表达力，并指出仅看密度图无法判断计算成本。",
        evidence_boundary="图 1 展示特定二维 planar/radial flow 的表达效果；它不证明任意可逆网络都能高效表示任意后验，也不免除可逆性与 log-det 的条件。",
    )
    return


@app.cell
def _(derivation_map, mo):
    mo.vstack(
        [
            mo.md("## 推导站五｜Normalizing flow 改 q，不改 ELBO 的基本身份"),
            derivation_map(["从 q_0 抽 z_0", "依次做可逆变换", "每步修正局部体积", "log-det 累加", "把 q_K 放回 ELBO"]),
            mo.md(r"""
            若 \(z_k=f_k(z_{k-1})\)，且每步在所考虑区域内是可微双射、Jacobian 非奇异，

            \[
            \log q_K(z_K\mid x)=\log q_0(z_0\mid x)
            -\sum_{k=1}^K\log\left|
              \det \frac{\partial f_k}{\partial z_{k-1}}
            \right|.
            \]

            这仍然进入标准 ELBO：

            \[
            \mathbb E_{q_K}
            [\log p_\theta(x,z_K)-\log q_K(z_K\mid x)].
            \]

            因此 flow 的核心不是“给 loss 再乘一个系数”，而是让近似后验能弯曲、相关甚至形成
            更复杂形状。代价是每步必须可逆且能高效计算 log-determinant。第 13 章的一维积分
            检查验证概率质量守恒；严格证明来自变量替换定理，两者不能混称。
            """),
        ],
        gap=0.8,
    )
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 公式到代码｜先识别“哪一行代表论文创新”

    ```python
    # β-VAE / capacity control：模型结构可不变，主要改 objective。
    beta_loss = recon + beta * kl
    capacity_loss = recon + gamma * torch.abs(kl - target_capacity)

    # IWAE：log_w shape [K, B]；先沿 K 做稳定 log-mean-exp，再沿 batch 平均。
    iwae_bound = (torch.logsumexp(log_w, dim=0) - math.log(K)).mean()

    # CVAE（课程记号）：condition y 同时进入 prior / encoder / decoder 的具体位置
    # 取决于声明的条件概率图，不能只凭字符串拼接猜公式。
    mu_q, logvar_q = encoder(x, y)
    mu_p, logvar_p = conditional_prior(y)
    x_hat = decoder(z, y)

    # VQ-VAE：forward 用量化值，backward 给 encoder 一条替代梯度。
    z_st = z_e + (z_q - z_e).detach()

    # Flow：每个样本各有一个 log_det；不能把 batch 维也求进 determinant。
    z, log_abs_det = transform(z)
    log_q = log_q - log_abs_det
    ```

    一段代码同时出现这些行，并不代表自动获得所有论文优点。每种改法都有自己的假设、
    评估指标和失败模式；组合后还要重新验证目标是否一致、梯度是否稳定。
    """)
    return


@app.cell
def _(mo):
    mo.callout(
        mo.md(r"""
        ## 论文阅读中的事实校正与开放问题

        - β‑VAE 原论文展示了特定数据和指标上的结果；后来研究指出，无监督解耦依赖归纳偏置，
          因此“大 β 必然恢复真实因素”不是定理。
        - Posterior collapse 是现象名称，不是单一病因名称。Bowman 的强 decoder 场景和 He 等的
          inference lag 解释可以同时成立，也可能不是某个新数据集上的主因。
        - IWAE 的 bound 值随 K 趋紧，不等于 encoder 梯度质量随 K 单调变好。
        - VQ‑VAE 中 codebook loss 更新与 EMA 更新是可选路线；VQ‑VAE‑2 不是只把 K 调大。
        - TD‑VAE 已收入拓展论文表，但不塞入本部分主线。它需要状态空间模型、Markov chain、
          belief state 与时序平滑；建议完成第 14 章后再回读。
        """),
        kind="info",
    )
    return


@app.cell
def _(exercise_block, mo):
    mo.vstack(
        [
            mo.md("## 论文研讨测试｜从新名字反推出修改位置"),
            exercise_block(
                (
                    "不看上文，把 β‑VAE、IWAE、CVAE、VQ‑VAE、normalizing flow 分别归类到：目标权衡、多样本估计、条件模型、latent 类型、后验族。",
                    "β‑VAE→目标权衡；IWAE→多样本估计/更紧下界；CVAE→条件概率模型；VQ‑VAE→离散 latent 类型与替代梯度；normalizing flow→更灵活的近似后验族。Posterior-collapse 论文主要属于失败诊断与优化动态。",
                ),
                (
                    r"某 toy 训练有 \(R=12,K=3,C=2,\gamma=4\)。计算 \(R+\gamma|K-C|\)。它是否严格保证下一步 \(K=2\)？",
                    r"数值为 \(12+4|3-2|=16\)。它只是软惩罚，有限 gamma 和非凸优化都不能保证下一步或最终严格满足 \(K=2\)。",
                ),
                (
                    "`log_w` shape 为 `[K,B]`。代码写成 `torch.logsumexp(log_w, dim=1)` 有什么问题？给出正确的 IWAE batch bound。",
                    "`dim=1` 错把 batch 当重要性样本轴混合。应写 `bound = (torch.logsumexp(log_w, dim=0) - math.log(K)).mean()`：先沿 K 得到每个数据样本的 bound `[B]`，再沿 batch 平均。",
                ),
                (
                    "任选一个现实任务，先判断它真正缺的是条件控制、离散代码、灵活后验还是更紧下界，再选论文；给出一个能证伪你选择的实验。",
                    "示例：语音单位发现需要离散代码，可选 VQ‑VAE。证伪实验应比较连续 VAE 与 VQ‑VAE，在相同 encoder/decoder 预算下检查 code usage、重构、下游音素可分性和 prior 生成；若离散模型 codebook collapse 或下游指标不升，就不能仅凭离散直觉宣称更合适。",
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
            "六类 VAE 变体修改的是不同对象，不能放进一条单轴性能排行榜。",
            "论文中的式子必须连同支持集、可积性、非凸优化、符号约定和评估任务一起阅读。",
            "现在可以从一个新模型的改动位置，反推它应比较的 baseline、记录的指标和可能失败的环节。",
        ],
        bridge="VAE 路线到这里完成了从概率潜变量、训练失败到离散表示和灵活后验的研究地图。下一部分转向扩散：不再一次从 z 解码，而是设计一条逐步破坏与恢复的概率路径。",
    )
    return


if __name__ == "__main__":
    app.run()
