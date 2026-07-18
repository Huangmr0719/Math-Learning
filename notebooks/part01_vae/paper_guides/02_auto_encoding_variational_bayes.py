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
    import torch
    from matplotlib.patches import Circle, FancyArrowPatch, Rectangle
    from scipy.stats import norm

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
        Circle,
        FancyArrowPatch,
        PAPERS,
        PAPER_GUIDES,
        Rectangle,
        derivation_map,
        exercise_block,
        intuition_and_rigor,
        mo,
        norm,
        np,
        paper_evidence_block,
        paper_guide_footer,
        paper_guide_header,
        plt,
        torch,
    )


@app.cell
def _(PAPER_GUIDES, paper_guide_header):
    GUIDE = PAPER_GUIDES["vae_aevb"]
    paper_guide_header(GUIDE, duration="120–180 分钟，可分两次完成")
    return (GUIDE,)


@app.cell
def _(PAPERS, paper_evidence_block):
    paper_evidence_block(
        PAPERS["aevb"],
        title="先在原图中分清生成模型与近似推断",
        original_quote="a reparameterization of the variational lower bound yields a lower bound estimator",
        chinese_translation="作者的第一项关键贡献，是把变分下界重新参数化，得到可用随机梯度优化的下界估计器。",
        figure_path="assets/paper_figures/vae/aevb_figure1_graphical_model.webp",
        figure_number="原文 Figure 1",
        figure_caption="AEVB 的有向图模型：实线表示生成模型，虚线表示对不可解后验的变分近似。",
        legend=(
            "白色 z 是未观测潜变量；灰色 x 是已经观测到的数据。",
            "实线 z→x 属于生成模型 pθ(z)pθ(x|z)，从 prior 产生数据。",
            "虚线 x⇢z 属于近似后验 qφ(z|x)，只在推断与训练中使用。",
            "外框 N 是 plate notation（板式记号），表示同一结构对 N 个独立同分布样本重复。",
            "θ 与 φ 是两套共享参数；箭头指向随机变量不等于参数本身也是随机变量。",
        ),
        learning_goal="先遮住说明文字，只凭线型说出训练时的 x→z 路径与生成时的 z→x 路径，并解释两者为何不能合并成普通确定性 AE。",
        evidence_boundary="图 1 给出概率依赖和参数分工，但不推导 ELBO，也不保证 qφ 等于真实后验；近似误差需要由 KL 分解另行分析。",
    )
    return


@app.cell
def _(mo):
    mo.vstack(
        [
            mo.callout(
                mo.md(
                    r"""
                    **这不是第 1–7 章的重复总结。**

                    前七章是老师把知识按学习顺序重新铺开；这次要反过来，学习作者如何在
                    一篇研究论文里压缩问题、假设、公式、算法与实验。最终胜利不是“看完”，
                    而是能从原文的式 (1)、(4)、(7)、(10) 重建最小 VAE。
                    """
                ),
                kind="info",
            ),
            mo.md(
                r"""
                ## 一手资料与使用方式

                - 原论文：[Auto-Encoding Variational Bayes — arXiv:1312.6114](https://arxiv.org/abs/1312.6114)
                - 本地核对版本：`references/papers/vae/auto-encoding-variational-bayes.pdf`
                - 作者：Diederik P. Kingma、Max Welling；课程采用 ICLR 2014 对应的 arXiv v10。

                本导读的中文内容是**教学性意译与解释**，不是逐字全文翻译。每个关键结论都会
                标出论文节号或式号；英语原句只保留定位概念所需的短语。请始终把原论文视为
                最终来源，把本 notebook 视为阅读脚手架。

                ### 完成后你应当能做到

                1. 用自己的话说出论文面对的两个计算瓶颈和两项核心贡献；
                2. 从式 (1) 推导证据下界，从式 (4) 解释重参数化为何允许路径导数；
                3. 把式 (10) 逐项翻译成 PyTorch 的 `recon_nll + kl`；
                4. 读懂图 1 的实线与虚线，以及图 4 为什么使用高斯分位点；
                5. 区分“论文在特定实验中观察到”与“数学上普遍成立”。
                """
            ),
        ],
        gap=0.8,
    )
    return


@app.cell
def _(mo):
    reading_route = mo.ui.radio(
        options={
            "第一次读：问题 → 图 → 公式 → 代码": "first",
            "复习推导：式 (1) → 式 (10)": "math",
            "研究复盘：贡献 → 证据 → 局限": "research",
        },
        value="第一次读：问题 → 图 → 公式 → 代码",
        label="选择本次阅读路线",
    )
    return (reading_route,)


@app.cell
def _(mo, reading_route):
    _routes = {
        "first": (
            "建议完整顺序",
            "先完成摘要侦察与图 1，再操作 ELBO 缺口和重参数化，最后进入代码与实验批判。",
        ),
        "math": (
            "建议聚焦公式",
            "从式 (1) 的恒等分解开始，依次检查式 (3)、(4)–(5)、(7) 与 (10) 的符号和定义域。",
        ),
        "research": (
            "建议带着审稿问题阅读",
            "追问每项贡献靠哪条推导、哪个实验支持，以及论文没有证明什么。",
        ),
    }
    _title, _message = _routes[reading_route.value]
    mo.callout(mo.md(f"**{_title}**\n\n{_message}"), kind="neutral")
    return


@app.cell
def _(mo):
    abstract_claim = mo.ui.radio(
        options={
            "更深的解码网络": "decoder",
            "可微的下界估计": "sgvb",
            "更大的训练数据": "dataset",
            "离散的潜变量": "discrete",
        },
        label="不回看上文：哪一项是摘要明确强调的核心贡献？",
    )
    abstract_claim
    return (abstract_claim,)


@app.cell
def _(abstract_claim, mo):
    if abstract_claim.value is None:
        _feedback = mo.md("先凭记忆选择，再看反馈。四个选项长度相同，不提供形式线索。")
        _kind = "warn"
    elif abstract_claim.value == "sgvb":
        _feedback = mo.md(
            "正确。摘要首先强调：重参数化后可以得到能用随机梯度方法优化的下界估计器。"
        )
        _kind = "success"
    else:
        _feedback = mo.md(
            "这不是摘要的核心贡献。回到论文首页，只找“问题是什么”和“作者说贡献有几项”。"
        )
        _kind = "danger"
    mo.callout(_feedback, kind=_kind)
    return


@app.cell
def _(intuition_and_rigor, mo):
    mo.vstack(
        [
            mo.md(
                r"""
                ## 摘要侦察｜先翻译问题，不急着翻译每个单词

                论文摘要可以拆成四句话的功能，而不是一长段英语：

                | 功能 | 英语定位短语 | 教学性中文意译 |
                |---|---|---|
                | 问题 | *intractable posterior distributions* | 后验难以直接计算时，怎样高效推断和学习？ |
                | 方法一 | *reparameterization of the variational lower bound* | 把随机变量改写，使下界估计可以直接求梯度。 |
                | 方法二 | *fitting an approximate inference model* | 用一个共享的识别模型快速给出每个样本的近似后验。 |
                | 证据 | *experimental results* | 作者用当时的图像实验展示方法的计算优势。 |

                注意标题 **Auto-Encoding Variational Bayes** 更合适的教学性翻译是
                **“自编码式变分贝叶斯”**。论文提出的是 AEVB 学习算法；当识别模型使用
                神经网络时，才得到后来广泛称为 VAE 的模型形式。
                """
            ),
            intuition_and_rigor(
                r"""
                旧方法像是每来一张图片，都要临时做一轮昂贵侦查，猜测它的潜变量在哪里。
                AEVB 学一个共享的“侦查员” \(q_\phi(z\mid x)\)：新图片进来，一次前向计算
                就给出可能的潜变量分布。这就是**摊销推断**（amortized inference）的核心直觉。
                """,
                r"""
                对独立同分布数据 \(\{x^{(i)}\}_{i=1}^N\)，论文联合优化生成参数
                \(\theta\) 与变分参数 \(\phi\)。\(q_\phi(z\mid x)\) 在所有数据点之间共享
                \(\phi\)，因而不再为每个 \(x^{(i)}\) 单独运行迭代变分优化。论文正文称它为
                recognition model；本课程统一译作**识别模型 / 概率编码器**。
                """,
            ),
        ],
        gap=0.8,
    )
    return


@app.cell
def _(derivation_map, mo):
    mo.vstack(
        [
            mo.md("## 论文路线图｜十四页论文真正需要抓住的骨架"),
            derivation_map(
                [
                    "§2.1 难算的边缘似然与后验",
                    "式 (1)–(3) 证据下界",
                    "式 (4)–(5) 重参数化",
                    "式 (7) 低方差 SGVB 估计器",
                    "算法 1 小批量 AEVB",
                    "式 (9)–(10) 高斯 VAE",
                    "图 2–5 实验与可视化",
                ]
            ),
            mo.callout(
                mo.md(
                    "第一次阅读可以暂时跳过附录 D–F。附录 B 的高斯 KL 与附录 C 的概率编码器/解码器值得回看。"
                ),
                kind="neutral",
            ),
        ],
        gap=0.8,
    )
    return


@app.cell
def _(mo):
    plate_view = mo.ui.radio(
        options={
            "只看生成路径": "generative",
            "只看推断路径": "inference",
            "同时对照两条路径": "both",
        },
        value="同时对照两条路径",
        label="图 1 阅读视角",
    )
    return (plate_view,)


@app.cell
def _(COLORS, Circle, FancyArrowPatch, Rectangle, mo, plate_view, plt):
    _show_generative = plate_view.value in {"generative", "both"}
    _show_inference = plate_view.value in {"inference", "both"}
    _fig, _axis = plt.subplots(figsize=(7.2, 4.2))
    _axis.set_xlim(0, 10)
    _axis.set_ylim(0, 7)
    _axis.axis("off")

    _axis.add_patch(Rectangle((3.1, 0.7), 3.8, 5.6, fill=False, lw=1.7, ec="#64748b"))
    _z = Circle((5, 4.7), 0.58, fc="#dbeafe", ec=COLORS["model"], lw=2)
    _x = Circle((5, 2.1), 0.58, fc="#dcfce7", ec=COLORS["data"], lw=2)
    _axis.add_patch(_z)
    _axis.add_patch(_x)
    _axis.text(5, 4.7, r"$z$", ha="center", va="center", fontsize=18)
    _axis.text(5, 2.1, r"$x$", ha="center", va="center", fontsize=18)
    _axis.text(6.45, 0.95, r"$N$ 个独立样本", ha="right", color="#475569")

    if _show_generative:
        _axis.add_patch(
            FancyArrowPatch(
                (5, 4.08),
                (5, 2.75),
                arrowstyle="-|>",
                mutation_scale=18,
                lw=2.5,
                color=COLORS["model"],
            )
        )
        _axis.text(5.25, 3.42, r"$p_\theta(x\mid z)$", color=COLORS["model"], va="center")
        _axis.text(7.35, 4.95, r"$p_\theta(z)$ 先验", color=COLORS["model"])
        _axis.add_patch(
            FancyArrowPatch(
                (7.2, 4.82),
                (5.62, 4.72),
                arrowstyle="-|>",
                mutation_scale=15,
                lw=1.7,
                color=COLORS["model"],
            )
        )
    if _show_inference:
        _axis.add_patch(
            FancyArrowPatch(
                (4.55, 2.55),
                (4.55, 4.15),
                arrowstyle="-|>",
                mutation_scale=18,
                lw=2.5,
                linestyle="--",
                color=COLORS["prior"],
            )
        )
        _axis.text(2.05, 3.35, r"$q_\phi(z\mid x)$", color=COLORS["prior"], va="center")

    _axis.set_title("根据原论文图 1 重绘：实线是生成模型，虚线是变分近似")
    _fig.tight_layout()
    _explanation = (
        "读图时先问方向：生成时从 z 到 x；推断时已知 x，近似反推 z。"
        " 两条箭头不是同一个函数的正向和严格逆向。"
    )
    mo.vstack(
        [
            mo.md("## 图 1 解剖｜一张图区分模型与推断"),
            plate_view,
            _fig,
            mo.callout(mo.md(_explanation), kind="info"),
        ],
        gap=0.7,
    )
    return


@app.cell
def _(mo):
    posterior_mu = mo.ui.slider(
        -2.5, 2.5, step=0.1, value=1.2, show_value=True, label=r"近似后验均值 $\mu_q$"
    )
    posterior_logvar = mo.ui.slider(
        -2.0, 1.5, step=0.1, value=-0.6, show_value=True, label=r"近似后验 $\log\sigma_q^2$"
    )
    return posterior_logvar, posterior_mu


@app.cell
def _(COLORS, mo, np, plt, posterior_logvar, posterior_mu):
    _mu = posterior_mu.value
    _variance = np.exp(posterior_logvar.value)
    _sigma = np.sqrt(_variance)
    _z = np.linspace(-4.5, 4.5, 500)
    _true_posterior = np.exp(-0.5 * _z**2) / np.sqrt(2 * np.pi)
    _approximate = np.exp(-0.5 * ((_z - _mu) / _sigma) ** 2) / (
        _sigma * np.sqrt(2 * np.pi)
    )
    _kl = 0.5 * (_mu**2 + _variance - 1.0 - posterior_logvar.value)
    _log_evidence = -1.0
    _elbo = _log_evidence - _kl

    _fig, _axes = plt.subplots(1, 2, figsize=(10, 3.8))
    _axes[0].plot(_z, _true_posterior, color=COLORS["data"], lw=2.5, label=r"toy $p_\theta(z\mid x)$")
    _axes[0].plot(_z, _approximate, color=COLORS["prior"], lw=2.5, label=r"$q_\phi(z\mid x)$")
    _axes[0].fill_between(_z, _approximate, alpha=0.12, color=COLORS["prior"])
    _axes[0].set_title("近似后验靠近真实后验时，KL 缺口缩小")
    _axes[0].legend(fontsize=8)
    _axes[0].set_xlabel("z")

    _axes[1].bar(["log evidence", "ELBO"], [_log_evidence, _elbo], color=[COLORS["data"], COLORS["model"]])
    _axes[1].annotate(
        f"KL gap = {_kl:.3f}",
        xy=(1, (_log_evidence + _elbo) / 2),
        xytext=(0.35, (_log_evidence + _elbo) / 2),
        arrowprops={"arrowstyle": "<->", "color": COLORS["prior"]},
        color=COLORS["prior"],
    )
    _axes[1].set_title(r"式 (1)：$\log p_\theta(x)=KL+\mathcal{L}$")
    _axes[1].set_ylabel("数值越高越好")
    _fig.tight_layout()

    mo.vstack(
        [
            mo.md("## 式 (1) 交互实验｜ELBO 到底‘下’在哪里？"),
            mo.hstack([posterior_mu, posterior_logvar], widths="equal"),
            _fig,
            mo.callout(
                mo.md(
                    rf"""
                    当前 toy 设定中：\(KL={_kl:.3f}\)，所以
                    \(\mathcal L={_log_evidence:.1f}-{_kl:.3f}={_elbo:.3f}\)。

                    **重要限定：** 为了把式 (1) 画出来，这里特意令“真实后验”是标准正态。
                    它不是 VAE 通常选择的先验 \(p(z)\)。真实任务中
                    \(p_\theta(z\mid x)\) 正是难以计算的对象。
                    """
                ),
                kind="warn",
            ),
        ],
        gap=0.7,
    )
    return


@app.cell
def _(intuition_and_rigor, mo):
    mo.vstack(
        [
            mo.md(
                r"""
                ## 式 (1)–(3) 严格推导｜论文压缩掉的代数台阶

                **目标：** 证明论文式 (1)

                \[
                \log p_\theta(x)
                =D_{KL}\!\left(q_\phi(z\mid x)\|p_\theta(z\mid x)\right)
                +\mathcal L(\theta,\phi;x).
                \]

                **假设与定义域：** \(q_\phi(z\mid x)\) 已归一化；在 \(q>0\) 的地方
                \(p_\theta(z\mid x)>0\)；相关期望有限；\(p_\theta(x)>0\)。

                1. **[KL 定义]**

                   \[
                   D_{KL}(q\|p_\theta(z\mid x))
                   =\mathbb E_q\!\left[\log q_\phi(z\mid x)-\log p_\theta(z\mid x)\right].
                   \]

                2. **[贝叶斯公式]** 使用
                   \(\log p_\theta(z\mid x)=\log p_\theta(x,z)-\log p_\theta(x)\)：

                   \[
                   D_{KL}=\mathbb E_q[\log q_\phi(z\mid x)-\log p_\theta(x,z)]
                   +\log p_\theta(x).
                   \]

                   最后一项可以移出期望，因为它不依赖 \(z\)，并且
                   \(\mathbb E_q[1]=1\)。

                3. **[移项并定义 ELBO]**

                   \[
                   \log p_\theta(x)
                   =D_{KL}+\underbrace{\mathbb E_q[
                   \log p_\theta(x,z)-\log q_\phi(z\mid x)]}_{\mathcal L(\theta,\phi;x)}.
                   \]

                4. **[联合分布分解]** 把
                   \(p_\theta(x,z)=p_\theta(x\mid z)p_\theta(z)\) 代入：

                   \[
                   \mathcal L
                   =\mathbb E_q[\log p_\theta(x\mid z)]
                   -D_{KL}(q_\phi(z\mid x)\|p_\theta(z)),
                   \]

                   这就是论文式 (3)。第一项奖励解释 / 重构数据，第二项约束近似后验。

                **哪里才用到“下界”？** 上面首先是恒等式；再使用
                \(D_{KL}\ge0\)，才得到 \(\mathcal L\le\log p_\theta(x)\)。
                """
            ),
            intuition_and_rigor(
                "ELBO 像一张保守成绩单。真实成绩 `log p(x)` 看不到，但我们知道它等于成绩单加上一段非负缺口。",
                r"单次 Monte Carlo 的 \(\widetilde{\mathcal L}\) 可能因采样波动高于 \(\log p(x)\)；真正构成下界的是期望形式 \(\mathcal L\)，不是每一个随机估计值。",
            ),
            mo.md(
                r"""
                ### 维度与方向检查

                | 对象 | 类型 / shape | 必须检查 |
                |---|---|---|
                | \(x^{(i)}\) | 单个观测向量，`[data_dim]` | 数据定义域与似然匹配 |
                | \(z\) | 潜变量向量，`[latent_dim]` | 连续，论文重参数化假设适用 |
                | \(\log p_\theta(x\mid z)\) | 每样本标量 | 数据维通常在内部求和 |
                | \(D_{KL}(q_\phi\|p)\) | 每样本非负标量 | 方向是近似后验到目标分布 |
                | \(\mathcal L\) | 每样本标量 | 论文最大化；代码常最小化其负值 |
                """
            ),
        ],
        gap=0.8,
    )
    return


@app.cell
def _(mo):
    rp_mu = mo.ui.slider(-2.0, 2.0, step=0.1, value=0.7, show_value=True, label=r"$\mu$")
    rp_logvar = mo.ui.slider(-2.0, 1.5, step=0.1, value=-0.4, show_value=True, label=r"$\log\sigma^2$")
    rp_epsilon = mo.ui.slider(-2.5, 2.5, step=0.1, value=1.0, show_value=True, label=r"固定噪声 $\epsilon$")
    return rp_epsilon, rp_logvar, rp_mu


@app.cell
def _(COLORS, mo, np, plt, rp_epsilon, rp_logvar, rp_mu):
    _sigma = np.exp(0.5 * rp_logvar.value)
    _z_value = rp_mu.value + _sigma * rp_epsilon.value
    _grid = np.linspace(-4, 4, 500)
    _standard = np.exp(-0.5 * _grid**2) / np.sqrt(2 * np.pi)
    _transformed = np.exp(-0.5 * ((_grid - rp_mu.value) / _sigma) ** 2) / (
        _sigma * np.sqrt(2 * np.pi)
    )

    _fig, _axes = plt.subplots(1, 2, figsize=(10, 3.6), sharey=True)
    _axes[0].plot(_grid, _standard, color=COLORS["prior"], lw=2.4)
    _axes[0].scatter([rp_epsilon.value], [0], s=70, color=COLORS["prior"], zorder=3)
    _axes[0].set_title(r"固定来源：$\epsilon\sim\mathcal{N}(0,1)$")
    _axes[0].set_xlabel(r"$\epsilon$")
    _axes[1].plot(_grid, _transformed, color=COLORS["model"], lw=2.4)
    _axes[1].scatter([_z_value], [0], s=70, color=COLORS["model"], zorder=3)
    _axes[1].set_title(r"平移并缩放：$z=\mu+\sigma\epsilon$")
    _axes[1].set_xlabel(r"$z$")
    _fig.tight_layout()

    mo.vstack(
        [
            mo.md("## 式 (4)–(5) 动起来｜随机性搬家，而不是消失"),
            mo.hstack([rp_mu, rp_logvar, rp_epsilon], widths="equal"),
            _fig,
            mo.md(
                rf"""
                当前 \(\sigma=\exp(0.5\times {rp_logvar.value:.1f})={_sigma:.3f}\)，所以
                \(z={rp_mu.value:.1f}+{_sigma:.3f}\times {rp_epsilon.value:.1f}={_z_value:.3f}\)。

                论文式 (4) 写成一般形式
                \(z=g_\phi(\epsilon,x),\ \epsilon\sim p(\epsilon)\)。
                关键不是让采样消失，而是让噪声分布不再依赖 \(\phi\)，参数依赖进入可微函数
                \(g_\phi\)。在满足可微和可交换求导 / 积分等条件时，梯度可以沿这条路径传播。
                """
            ),
        ],
        gap=0.7,
    )
    return


@app.cell
def _(mo):
    mc_samples = mo.ui.dropdown(
        options={"100 个样本": 100, "1,000 个样本": 1000, "10,000 个样本": 10000},
        value="1,000 个样本",
        label="Monte Carlo 样本数",
    )
    mc_seed = mo.ui.slider(0, 20, step=1, value=7, show_value=True, label="随机种子（改变即重新采样）")
    return mc_samples, mc_seed


@app.cell
def _(mc_samples, mc_seed, mo, np, rp_logvar, rp_mu, torch):
    _rng = np.random.default_rng(mc_seed.value)
    _epsilon_np = _rng.normal(size=mc_samples.value)
    _mu = float(rp_mu.value)
    _logvar = float(rp_logvar.value)
    _sigma = float(np.exp(0.5 * _logvar))
    _z_np = _mu + _sigma * _epsilon_np

    _analytic_expectation = _mu**2 + _sigma**2
    _mc_expectation = float(np.mean(_z_np**2))
    _exact_mu_gradient = 2 * _mu

    _mu_t = torch.tensor(_mu, dtype=torch.float64, requires_grad=True)
    _eps_t = torch.tensor(_epsilon_np, dtype=torch.float64)
    _z_t = _mu_t + _sigma * _eps_t
    _estimate_t = (_z_t**2).mean()
    _estimate_t.backward()
    _autograd_mu = float(_mu_t.grad)

    _h = 1e-5
    _plus = np.mean(((_mu + _h) + _sigma * _epsilon_np) ** 2)
    _minus = np.mean(((_mu - _h) + _sigma * _epsilon_np) ** 2)
    _finite_difference = float((_plus - _minus) / (2 * _h))

    _log_q = -0.5 * (
        np.log(2 * np.pi) + _logvar + ((_z_np - _mu) ** 2) / (_sigma**2)
    )
    _log_p = -0.5 * (np.log(2 * np.pi) + _z_np**2)
    _mc_kl = float(np.mean(_log_q - _log_p))
    _analytic_kl = 0.5 * (_mu**2 + _sigma**2 - 1 - _logvar)

    mo.vstack(
        [
            mo.md("## 可执行核验｜式 (5) 的期望、梯度与高斯 KL"),
            mo.hstack([mc_samples, mc_seed], widths="equal"),
            mo.md(
                fr"""
                | 检查对象 | 解析值 | 数值值 | 绝对误差 |
                |---|---:|---:|---:|
                | $E[z^2]$ | {_analytic_expectation:.6f} | {_mc_expectation:.6f} | {abs(_analytic_expectation - _mc_expectation):.2e} |
                | $\partial E[z^2]/\partial\mu$ | {_exact_mu_gradient:.6f} | {_autograd_mu:.6f}（autograd） | {abs(_exact_mu_gradient - _autograd_mu):.2e} |
                | 同一随机数有限差分 | {_autograd_mu:.6f} | {_finite_difference:.6f} | {abs(_autograd_mu - _finite_difference):.2e} |
                | $KL(q\|\mathcal N(0,1))$ | {_analytic_kl:.6f} | {_mc_kl:.6f} | {abs(_analytic_kl - _mc_kl):.2e} |
                """
            ),
            mo.callout(
                mo.md(
                    "数值相符支持代码实现与解析公式一致，但不构成一般性数学证明。改变种子和样本数，观察 Monte Carlo 误差如何波动。"
                ),
                kind="warn",
            ),
        ],
        gap=0.7,
    )
    return


@app.cell
def _(mo):
    objective_view = mo.ui.radio(
        options={
            "论文视角：最大化 ELBO": "paper",
            "代码视角：最小化 loss": "code",
        },
        value="论文视角：最大化 ELBO",
        label="切换目标函数视角",
    )
    return (objective_view,)


@app.cell
def _(mo, objective_view):
    if objective_view.value == "paper":
        _formula = r"""
        \[
        \widetilde{\mathcal L}^{B}
        =-D_{KL}(q_\phi(z\mid x)\|p(z))
        +\frac1L\sum_{l=1}^{L}\log p_\theta(x\mid z^{(l)}).
        \]

        两项都写成“越大越好”：负 KL 越接近 0 越好，log likelihood 越大越好。
        """
    else:
        _formula = r"""
        \[
        \text{loss}=-\widetilde{\mathcal L}^{B}
        =\underbrace{-\log p_\theta(x\mid z)}_{\text{reconstruction NLL}}
        +\underbrace{D_{KL}(q_\phi\|p)}_{\text{KL loss}}.
        \]

        两项都写成“越小越好”。这只是整体乘以 \(-1\)，不是另一个训练目标。
        """
    mo.vstack(
        [
            mo.md("## 式 (7) 与式 (10)｜从论文最大化到代码最小化"),
            objective_view,
            mo.md(_formula),
        ],
        gap=0.7,
    )
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### 高斯 VAE 的式 (10)

    论文令 \(q_\phi(z\mid x)=\mathcal N(\mu,\operatorname{diag}(\sigma^2))\)、
    \(p(z)=\mathcal N(0,I)\)，得到

    \[
    \widetilde{\mathcal L}
    =\frac12\sum_{j=1}^{J}
    (1+\log\sigma_j^2-\mu_j^2-\sigma_j^2)
    +\frac1L\sum_{l=1}^{L}\log p_\theta(x\mid z^{(l)}).
    \]

    第一项正是 \(-KL\)，不是 \(KL\)。把整式取负，才得到常见代码：

    ```python
    # 输入 mu、logvar 的 shape 都是 [batch_size, latent_dim]。
    # logvar = log(sigma**2)，所以 0.5 * logvar = log(sigma)。
    std = torch.exp(0.5 * logvar)

    # epsilon 的分布不依赖 encoder 参数；shape 与 std 完全相同。
    epsilon = torch.randn_like(std)
    z = mu + std * epsilon                 # 论文式 (4) / 式 (10)

    # decoder 给出 p_theta(x|z) 的参数。这里假设二值数据采用 Bernoulli likelihood。
    x_probability = decoder(z)

    # -log p_theta(x|z)：先沿 data_dim 求和，保留 batch 维。
    recon_nll = F.binary_cross_entropy(
        x_probability, x, reduction="none"
    ).sum(dim=1)                            # shape: [batch_size]

    # KL：先沿 latent_dim 求和，保留 batch 维。
    kl = -0.5 * (
        1 + logvar - mu.pow(2) - logvar.exp()
    ).sum(dim=1)                            # shape: [batch_size]

    # 论文最大化每样本 ELBO；代码最小化 batch 平均负 ELBO。
    loss = (recon_nll + kl).mean()          # scalar
    ```

    **删改诊断：** 把 `exp(0.5 * logvar)` 写成 `exp(logvar)` 会把方差当成标准差；
    把 `sum(dim=1)` 换成无条件 `mean()` 会改变数据维和潜维的相对尺度；漏掉最外层负号
    则会把 KL 往更大方向优化。

    ### 论文式 (8) 与常见 batch mean 为什么长得不同？

    论文用 \(\frac{N}{M}\sum_{i=1}^{M}\widetilde{\mathcal L}^{(i)}\) 无偏估计整个数据集的
    ELBO 总和。训练代码通常最小化 `negative_elbo.mean()`，相当于再除以固定常数 \(N\)。
    在没有额外正则且 \(N\) 固定时，正比例缩放不改变最优点；但会改变梯度大小、学习率
    含义以及与其他正则项的相对权重，因此实现中仍要明确 reduction。
    """)
    return


@app.cell
def _(PAPERS, paper_evidence_block):
    paper_evidence_block(
        PAPERS["aevb"],
        title="二维 latent 不是一张随意排列的样本拼图",
        original_quote="Visualisations of learned data manifold for generative models with two-dimensional latent space",
        chinese_translation="作者在二维潜空间上有序取点，并把每个点送入概率解码器，从而可视化学到的数据流形。",
        figure_path="assets/paper_figures/vae/aevb_figure4_manifold.webp",
        figure_number="原文 Figure 4",
        figure_caption="二维 AEVB 在 Frey Face 与 MNIST 上学到的生成流形；网格坐标先由单位方形的等距分位点映射到高斯 latent。",
        legend=(
            "左图每个小格是 Frey Face decoder 在一个二维 z 坐标处的输出。",
            "右图每个小格是 MNIST decoder 的输出；相邻格只改变较小的 latent 位移。",
            "横、纵方向分别扫描 z 的两个维度，不代表两个轴天然对应固定语义。",
            "网格先在概率分位点上等距，再经标准高斯 inverse CDF 映到 z；因此边缘处 z 间距会变大。",
        ),
        learning_goal="比较图中央与边缘的变化，并解释为什么按高斯分位点排网格，比直接在 z 上等距更能覆盖 prior 的相近概率质量。",
        evidence_boundary="连续过渡是该二维模型与这些数据上的实验结果；它不证明任意 VAE 都会学到解耦坐标，也不能由图像平滑性推出精确似然更高。",
    )
    return


@app.cell
def _(mo):
    manifold_scheme = mo.ui.radio(
        options={
            "论文思路：概率分位点等距": "quantile",
            "常见误读：潜坐标直接等距": "linear_z",
        },
        value="论文思路：概率分位点等距",
        label="图 4 的 latent 网格方式",
    )
    manifold_size = mo.ui.dropdown(
        options={"3×3（先看结构）": 3, "5×5（推荐）": 5, "7×7（更多细节）": 7},
        value="5×5（推荐）",
        label="网格密度",
    )
    return manifold_scheme, manifold_size


@app.cell
def _(manifold_scheme, manifold_size, mo, norm, np, plt):
    _n = manifold_size.value
    if manifold_scheme.value == "quantile":
        # 原论文图 4 先在单位正方形中等距取概率，再用高斯逆 CDF 映射到 z。
        # 教学实现避开概率 0 和 1，因为 norm.ppf(0/1) 会得到无穷大。
        _probabilities = np.linspace(0.08, 0.92, _n)
        _coordinates = norm.ppf(_probabilities)
        _coordinate_note = "概率质量较均匀；z 坐标本身并不等距"
    else:
        _coordinates = np.linspace(-2.0, 2.0, _n)
        _coordinate_note = "z 坐标等距；标准正态中心区域会被相对低估"

    _pixel = np.linspace(-1, 1, 20)
    _xx, _yy = np.meshgrid(_pixel, _pixel)

    def _toy_decoder(_z1, _z2):
        # 这不是训练后的 MNIST decoder，而是一个确定、连续的 toy decoder。
        # 它只负责展示“二维 latent 网格 -> 平滑变化的观测图像”这一读图原则。
        _cx = 0.38 * np.tanh(_z1)
        _cy = 0.38 * np.tanh(_z2)
        _width_x = 0.13 + 0.09 / (1 + np.exp(-_z2))
        _width_y = 0.16 + 0.08 / (1 + np.exp(_z1))
        _blob = np.exp(
            -0.5 * (((_xx - _cx) / _width_x) ** 2 + ((_yy - _cy) / _width_y) ** 2)
        )
        _echo = 0.55 * np.exp(
            -0.5
            * (
                ((_xx + 0.45 * _cx) / (_width_y + 0.05)) ** 2
                + ((_yy + 0.45 * _cy) / (_width_x + 0.05)) ** 2
            )
        )
        return np.clip(_blob + _echo, 0, 1)

    _fig, _axes = plt.subplots(_n, _n, figsize=(7.3, 7.3))
    _axes = np.atleast_2d(_axes)
    for _row, _z2 in enumerate(_coordinates[::-1]):
        for _column, _z1 in enumerate(_coordinates):
            _axes[_row, _column].imshow(_toy_decoder(_z1, _z2), cmap="gray", vmin=0, vmax=1)
            _axes[_row, _column].axis("off")
    _fig.suptitle("教学重绘：二维 latent manifold 的连续变化", fontsize=13)
    _fig.tight_layout()

    mo.vstack(
        [
            mo.md("## 图 4 复现思路｜为什么要先取分位点，再做高斯逆 CDF？"),
            mo.hstack([manifold_scheme, manifold_size], widths="equal"),
            _fig,
            mo.md(
                f"""
                当前 z 轴坐标：`{np.round(_coordinates, 3).tolist()}`  
                解释：**{_coordinate_note}**。

                原论文图 4 用真实训练好的二维 VAE decoder 生成 Frey Face 和 MNIST manifold；
                上图没有复制论文图片，也没有冒充复现实验结果，而是用可检查的 toy decoder
                忠实演示其坐标构造方法。真正复现论文图还需要训练模型并固定数据预处理、网络、
                优化器和随机种子。
                """
            ),
        ],
        gap=0.7,
    )
    return


@app.cell
def _(mo):
    algorithm_step = mo.ui.slider(
        1, 5, step=1, value=1, show_value=True, label="算法 1：逐步翻译成训练代码"
    )
    return (algorithm_step,)


@app.cell
def _(algorithm_step, mo):
    _steps = {
        1: (
            "抽取小批量",
            "论文从 N 个数据点中随机抽 M 个，避免每次遍历全数据集。",
            "x = next(iter(dataloader))  # shape: [M, data_dim]",
            "检查 M 是 batch 大小，不是 Monte Carlo 样本数 L。",
        ),
        2: (
            "识别模型给出分布参数",
            "共享的 q_phi(z|x) 为每个样本输出近似后验参数。",
            "mu, logvar = encoder(x)     # [M, latent_dim]",
            "这一步体现摊销推断，而不是为每个样本单独优化一组参数。",
        ),
        3: (
            "抽噪声并重参数化",
            "论文从固定 p(epsilon) 抽样，再通过 g_phi 生成 posterior 样本。",
            "eps = torch.randn_like(mu)\nz = mu + torch.exp(0.5 * logvar) * eps",
            "eps、mu、std 必须形状一致；随机性留在 eps 中。",
        ),
        4: (
            "估计负 ELBO 并求梯度",
            "式 (7) 把解析 KL 与采样重构项组合，代码取负后反向传播。",
            "loss = (recon_nll + kl).mean()\nloss.backward()",
            "不要把数值 loss 与论文要最大化的 ELBO 符号混为一谈。",
        ),
        5: (
            "更新 theta 与 phi",
            "同一目标同时训练生成模型和识别模型。",
            "optimizer.step()\noptimizer.zero_grad()",
            "decoder 参数 theta 与 encoder 参数 phi 都必须在 optimizer 中。",
        ),
    }
    _title, _paper, _code, _check = _steps[algorithm_step.value]
    mo.vstack(
        [
            mo.md("## 算法 1 翻译器｜从 AEVB 伪代码到 PyTorch"),
            algorithm_step,
            mo.callout(
                mo.md(
                    f"""
                    ### 第 {algorithm_step.value} 步：{_title}

                    **论文动作：** {_paper}

                    ```python
                    {_code}
                    ```

                    **本步自检：** {_check}
                    """
                ),
                kind="info",
            ),
        ],
        gap=0.7,
    )
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 实验结果怎么读｜先判断证据强度，再记结论

    ### 论文实际支持了什么

    - 图 2 在作者给定的 MNIST / Frey Face、MLP、优化器与比较方法下，AEVB 的下界
      收敛比 wake-sleep 更快，并达到更好的数值。它支持“该方法在这些设置中有效”，
      不是“AEVB 在所有任务上必然优于所有推断方法”的定理。
    - 作者在实验中使用小批量 \(M=100\)、每个数据点 \(L=1\) 个 posterior 样本，报告
      这一设置已经可用。这是经验选择，不是 \(L=1\) 永远最优的证明。
    - 图 4 与图 5 展示二维流形和不同 latent 维度的样本，说明概率 decoder 可从 prior
      样本生成观测；图片好看不等于已经精确计算 marginal likelihood。

    ### 三个容易被二手材料抹掉的限定

    1. **连续潜变量限定。** 论文的路径重参数化需要合适的可微变换；普通离散采样不能
       直接照搬，后续才会学习 VQ-VAE 等不同策略。
    2. **单次估计不是下界保证。** \(\mathcal L\) 是下界；带噪的
       \(\widetilde{\mathcal L}\) 单次取值可能越过 \(\log p(x)\)。
    3. **同期独立工作。** 论文相关工作明确提到 Rezende、Mohamed 与 Wierstra 的
       stochastic backpropagation 是独立发展的相关视角，因此不宜把整个思想史压成
       “一篇论文凭空发明一切”。

    ### 发现原文笔误也是阅读能力

    论文第 5 节开头有一句把 generative model 写成 encoder、variational approximation
    写成 decoder，与第 2–3 节定义及同节后文相反。应以模型公式为准：
    \(q_\phi(z\mid x)\) 是概率编码器 / 识别模型，\(p_\theta(x\mid z)\) 是概率解码器。
    读论文不是默认每个词都绝对无误，而是让符号、公式、图和代码相互校验。
    """)
    return


@app.cell
def _(mo):
    claim_one = mo.ui.checkbox(label="不看答案，复述论文的两个计算瓶颈")
    claim_two = mo.ui.checkbox(label="从空白纸推到式 (1) 的 ELBO 恒等分解")
    claim_three = mo.ui.checkbox(label="逐行解释式 (10) 对应的 PyTorch loss")
    claim_four = mo.ui.checkbox(label="说明图 4 的概率分位点与 z 等距有何不同")
    return claim_four, claim_one, claim_three, claim_two


@app.cell
def _(claim_four, claim_one, claim_three, claim_two, mo):
    _checks = [claim_one, claim_two, claim_three, claim_four]
    _completed = sum(int(_item.value) for _item in _checks)
    _kind = "success" if _completed == 4 else "warn"
    _message = (
        "四项检索均已完成。现在进入章末测试，答案仍然先保持折叠。"
        if _completed == 4
        else "勾选只代表你已经在不看上文的情况下真正做过；不会自动判定内容正确。"
    )
    mo.vstack(
        [
            mo.md("## 合上论文后的检索练习｜训练长期记忆，而不是眼熟"),
            mo.vstack(_checks, gap=0.3),
            mo.callout(mo.md(f"已完成 **{_completed}/4** 项。{_message}"), kind=_kind),
        ],
        gap=0.7,
    )
    return


@app.cell
def _(exercise_block, mo):
    mo.vstack(
        [
            mo.md("## 论文导读测试｜能否从原文重建模型？"),
            exercise_block(
                (
                    "AEVB 同时缓解了哪两个瓶颈？为什么 recognition model 使推断具有“摊销”性质？",
                    "一是边缘似然和真实后验难以直接计算，二是大数据下为每个样本运行迭代推断代价过高。共享的 `q_phi(z|x)` 用同一组参数处理所有样本；训练成本被分摊，测试时一次前向计算即可给出近似后验。",
                ),
                (
                    "一维时设 `mu=1`、`logvar=0`，并观测到一次 `log p_theta(x|z)=-3`。算 KL、该次式 (10) ELBO 估计和代码最小化的 loss。",
                    "`logvar=0` 表示 `sigma^2=1`。因此 `KL=0.5*(mu^2+sigma^2-1-logvar)=0.5`；负 KL 为 `-0.5`，ELBO 估计是 `-0.5+(-3)=-3.5`；代码最小化其相反数，所以 `loss=3.5`。",
                ),
                (
                    "修正代码 `std = torch.exp(logvar); z = mu + std * epsilon`，并说明为何修正后对应论文的高斯重参数化。",
                    "应改为 `std = torch.exp(0.5 * logvar)`。因为 `logvar=log(sigma^2)=2*log(sigma)`，乘 `0.5` 后再指数化才得到标准差 `sigma`；于是 `z=mu+sigma*epsilon` 与论文式 (10) 一致。",
                ),
                (
                    "在“式 (1) 交互实验”中寻找 KL 接近 0 的参数，再解释为什么这项 toy 结果不能证明真实 VAE 的 posterior 已被精确恢复。",
                    "把 `mu_q` 调到 0、`log sigma_q^2` 调到 0 时，两条标准正态重合，toy KL 为 0。真实 VAE 的 `p_theta(z|x)` 通常不可计算，训练约束的解析 KL 是 `q_phi(z|x)` 到 prior `p(z)`；toy 图只是验证式 (1) 的缺口关系，不能证明真实后验匹配。",
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
            "论文的核心不是‘把 Autoencoder 加一点噪声’，而是用 SGVB 估计器联合学习生成模型与摊销近似后验。",
            "式 (1) 是恒等分解，KL 非负才把 ELBO 变成下界；式 (3) 再分成重构期望与 prior KL。",
            "式 (4) 把参数依赖移入可微变换，式 (7) 解析 KL、采样重构，式 (10) 落到高斯 VAE。",
            "图和实验必须连同坐标构造、数据、基线与限定一起阅读，不能把经验观察改写成普遍定理。",
            "现在已经能把论文符号、中文概念、PyTorch shape 和数值验证放进同一张地图。",
        ],
        bridge=(
            "原论文的标准 ELBO 给重构与 KL 固定了 1:1 权重。下一部分从 β-VAE 开始追问："
            "如果主动改变这项约束的强度，表示、重构与失败现象会怎样变化？"
        ),
    )
    return


if __name__ == "__main__":
    app.run()
