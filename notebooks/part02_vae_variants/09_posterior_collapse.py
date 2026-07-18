import marimo

__generated_with = "0.23.13"
app = marimo.App(width="medium")


@app.cell
def _():
    import sys as _sys
    from pathlib import Path as _Path
    _root = _Path(__file__).resolve().parents[2]
    if str(_root) not in _sys.path:
        _sys.path.insert(0, str(_root))
    import marimo as mo
    import numpy as np
    from src.teaching import CHAPTERS, chapter_footer, chapter_header, derivation_map, exercise_block, intuition_and_rigor
    from src.visualization import COLORS, configure_matplotlib
    configure_matplotlib()
    import matplotlib.pyplot as plt
    return CHAPTERS, COLORS, chapter_footer, chapter_header, derivation_map, exercise_block, intuition_and_rigor, mo, np, plt


@app.cell
def _(CHAPTERS, chapter_header):
    chapter_header(CHAPTERS[9], duration="80–110 分钟")
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 本章为什么存在

    VAE 可能得到很低的 KL，看起来“正则得很好”，实际却完全不使用 latent：

    \[
    q_\phi(z|x)\approx p(z),\qquad
    D_{KL}(q_\phi(z|x)\|p(z))\approx0.
    \]

    不同输入得到几乎相同 posterior，decoder 只能依赖自身能力生成。这叫 posterior collapse。

    ## 你已经知道什么

    - 第 8 章：更强 KL 压力会减少 latent 信息。
    - 第 4 章：在满足支持集等常规条件时，KL=0 表示两个分布几乎处处相同。
    - **回忆：** KL 很小为什么不一定是好消息？
    """)
    return


@app.cell
def _(intuition_and_rigor):
    intuition_and_rigor(
        "Encoder 像给 decoder 递纸条；如果 decoder 自己就能猜答案，而递纸条还要交 KL 费用，最省事的策略就是永远递空白纸条。",
        r"""
        Collapse 时 \(q_\phi(z|x)\) 对 x 近似不变，常见诊断包括 KL 接近 0、
        active latent units 很少、打乱 z 后重构几乎不变。互信息
        \(I_q(X;Z)=\mathbb E_x KL(q(z|x)\|q(z))\) 描述 z 对 x 平均保存的信息；
        它与逐样本 prior KL 相关但并不完全相同。
        """,
    )
    return


@app.cell
def _(derivation_map, mo):
    mo.vstack(
        [
            mo.md("## 为什么空白 latent 是局部最优"),
            derivation_map(["decoder 很强", "早期 z 信息噪声大", "忽略 z 可先改善重构", "KL 同时下降", "encoder 梯度变弱", "形成 collapse"]),
            mo.md(r"""
            常用干预：

            - KL annealing：训练早期令 \(\beta\) 小，逐渐升到 1；
            - free bits：每个 latent group 的少量 KL 不惩罚；
            - weakening decoder / skip connections：迫使 decoder 读取 z；
            - 改善优化与 posterior 表达力。

            Free bits 的一种写法：
            \[
            L=L_{\rm recon}+\sum_j\max(\lambda,KL_j).
            \]
            不同论文实现记号不同，必须确认是“下限截断 loss”还是“超出容量才收费”。

            ### 数学暂停站：逐样本 KL 到底在惩罚什么？

            定义 aggregated posterior

            \[
            q(z)=\mathbb E_{x\sim p_{\rm data}}q_\phi(z|x).
            \]

            它表示先随机抽一条数据 \(x\)，再从 encoder 抽取 \(z\) 后，忽略 x 得到的
            整体 latent 分布。平均 prior KL 可以严格分解为

            \[
            \mathbb E_x KL(q(z|x)\|p(z))
            =
            I_q(X;Z)+KL(q(z)\|p(z)).
            \]

            推导只需在对数比中乘除 \(q(z)\)：

            \[
            \log\frac{q(z|x)}{p(z)}
            =
            \log\frac{q(z|x)}{q(z)}
            +
            \log\frac{q(z)}{p(z)}.
            \]

            对联合分布 \(q(x,z)=p_{\rm data}(x)q(z|x)\) 取期望，第一项就是互信息
            \(I_q(X;Z)\)，第二项化成 aggregated posterior 到 prior 的 KL。

            因此逐样本 KL 同时压低两件事：z 保存多少关于 x 的信息，以及整体 latent
            分布离 prior 多远。Posterior collapse 对应第一项也接近 0，而不只是
            “整体分布看起来像 prior”。
            """),
        ],
        gap=0.8,
    )
    return


@app.cell
def _(mo):
    decoder_strength = mo.ui.slider(0.0, 3.0, value=1.5, step=0.1, show_value=True, label="decoder 自己解决任务的能力")
    beta = mo.ui.slider(0.0, 3.0, value=1.0, step=0.1, show_value=True, label="KL 权重")
    return beta, decoder_strength


@app.cell
def _(COLORS, beta, decoder_strength, mo, np, plt):
    _info = np.linspace(0, 2.5, 400)
    _recon = np.maximum(0, 2.5 - decoder_strength.value - _info) ** 2
    _kl = _info ** 2
    _loss = _recon + beta.value * _kl
    _best = int(np.argmin(_loss))
    _fig, _axes = plt.subplots(1, 2, figsize=(10, 3.7))
    _axes[0].plot(_info, _recon, label="recon", color=COLORS["data"])
    _axes[0].plot(_info, beta.value * _kl, label="beta×KL", color=COLORS["prior"])
    _axes[0].plot(_info, _loss, label="total", color=COLORS["model"])
    _axes[0].axvline(_info[_best], linestyle="--", color=COLORS["danger"])
    _axes[0].set_xlabel("toy latent 信息量")
    _axes[0].set_ylabel("loss / 加权项")
    _axes[0].legend(fontsize=8)
    _anneal = np.linspace(0, 1, 100)
    _axes[1].plot(_anneal, np.minimum(1, 2 * _anneal), label="linear warmup")
    _axes[1].plot(_anneal, np.ones_like(_anneal), label="beta=1 from start")
    _axes[1].set_title("KL annealing")
    _axes[1].set_xlabel("训练进度（归一化）")
    _axes[1].set_ylabel("KL 权重 beta")
    _axes[1].legend(fontsize=8)
    _fig.tight_layout()
    mo.vstack([mo.md(fr"""## 交互诊断

    toy 最优 latent 信息量约为 **{_info[_best]:.3f}**。
    decoder 越强或 beta 越大，最优点越容易靠近 0。该实验解释机制，不证明所有 collapse 都由同一原因造成。"""),
    mo.hstack([decoder_strength, beta], widths="equal"), _fig])
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 诊断代码

    ```python
    # 每个 latent dimension 先对 batch 求平均 KL。
    kl_per_dim = -0.5 * (
        1 + logvar - mu.pow(2) - logvar.exp()
    ).mean(dim=0)

    # 只看 total KL 会掩盖“少数维度工作、其余维度死亡”。
    active_units = (kl_per_dim > 0.01).sum()

    # 打乱 z 后重构若几乎不变，decoder 很可能没使用 latent。
    shuffled_reconstruction = decoder(z[torch.randperm(z.shape[0])])
    ```

    ## 错误与反例

    - KL 下降就宣布训练更好：必须结合重构、active units 与 z 扰动实验。
    - 把 \(q(z|x)\)、aggregated posterior \(q(z)\) 和 prior \(p(z)\) 当成同一个对象。
    - Annealing 永远不升到目标 beta：只是暂时绕开约束。
    - 把 collapse 与过拟合混为一谈：一个是 latent 未使用，一个是泛化问题。

    原始资料：[Bowman et al., Generating Sentences from a Continuous Space](https://arxiv.org/abs/1511.06349)
    """)
    return


@app.cell
def _(exercise_block, mo):
    mo.vstack(
        [
            mo.md("## 分层练习"),
            exercise_block(
                ("用纸条类比解释 collapse。", "decoder 自己能完成任务且读取纸条要付 KL 成本，于是 encoder 发送与输入无关的信息。"),
                ("按本章的 KL-threshold 代理，8 个维度中只有 2 个满足 KL>0.01，代理 active count 是多少？", "代理计数是 2。它只回答“有几维超过本章阈值”，不能冒充文献中基于 posterior mean 跨数据方差定义的 active units。"),
                ("已知 `kl_elementwise` shape 为 `[B,D]`，写出本章 KL-threshold 代理指标的 `kl_per_dim` 与 `active_units` 代码。", "`kl_per_dim = kl_elementwise.mean(dim=0)`，`active_units = (kl_per_dim > 0.01).sum()`。这是本章用于诊断的 KL-threshold 代理；文献也常用 posterior mean 跨数据的方差定义 active units，二者不可混称为同一指标。"),
                ("调高 decoder strength 与 beta，预测最优信息量。", "两者通常都会把 toy 最优点推向 0。"),
            ),
        ],
        gap=0.8,
    )
    return


@app.cell
def _(CHAPTERS, chapter_footer):
    chapter_footer(CHAPTERS[9], ["Collapse 是 posterior 对输入失去依赖。", "低 KL 既可能是规整，也可能是 latent 未使用。", "诊断需要分项日志和干预实验。", "下一步从更多 posterior 样本改善变分下界。"])
    return


if __name__ == "__main__":
    app.run()
