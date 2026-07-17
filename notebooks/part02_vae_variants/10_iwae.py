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
    chapter_header(CHAPTERS[10], duration="90–120 分钟")
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 本章为什么存在

    ELBO 用一个 posterior 样本估计 evidence。若 \(q(z|x)\) 漏掉了重要区域，
    单样本估计会很松。IWAE 同时抽 K 个样本，用 importance weights 平均：

    \[
    \mathcal L_K=
    \mathbb E\log\left[\frac1K\sum_{k=1}^K
    \frac{p(x,z_k)}{q(z_k|x)}\right].
    \]

    ## 你已经知道什么

    - 第 5 章：ELBO 是 \(\log p(x)\) 的下界。
    - 第 4 章：Monte Carlo 用样本平均近似期望。
    - **回忆：** 为什么先平均 importance weights，再取 log，而不是平均 log weight？
    """)
    return


@app.cell
def _(intuition_and_rigor):
    intuition_and_rigor(
        "一次投篮可能严重误判水平；多投几次更可能覆盖真正重要的 latent。importance weight 会让更能解释 x 的样本发言更大声。",
        r"""
        \(w_k=p(x,z_k)/q(z_k|x)>0\)，且 \(z_k\) 独立来自 q。
        \(\frac1K\sum w_k\) 是 \(p(x)\) 的无偏估计，但 log 后通常有负偏差。
        在适当条件下 \(\mathcal L_1\le\mathcal L_K\le\log p(x)\)，这里的单调性是对采样期望而言；
        某一次随机抽样的 K+1 估计不保证高于 K。
        """,
    )
    return


@app.cell
def _(derivation_map, mo):
    mo.vstack(
        [
            mo.md("## Importance sampling 与 log-sum-exp"),
            derivation_map(["从 q 采样 z", "计算 p(x,z)/q(z|x)", "K 个 weight 求平均", "对平均值取 log", "Jensen 得到下界", "用 log-sum-exp 稳定实现"]),
            mo.md(r"""
            数值稳定公式：
            \[
            \log\frac1K\sum_ke^{a_k}
            =m+\log\sum_ke^{a_k-m}-\log K,\quad m=\max_k a_k.
            \]
            减去 m 避免指数溢出；最后必须加回 m。`log_w` shape `[K,B]` 时，
            `logsumexp(dim=0)` 沿样本轴聚合，留下 `[B]`，再对 batch 平均。

            ### 为什么 importance weight 的平均值无偏？

            需要支持集条件：只要 \(p(x,z)>0\)，proposal \(q(z|x)\) 就不能为 0。
            对单个样本 \(Z\sim q(z|x)\)：

            \[
            \begin{aligned}
            E_q\left[\frac{p(x,Z)}{q(Z|x)}\right]
            &=\int q(z|x)\frac{p(x,z)}{q(z|x)}dz\\
            &=\int p(x,z)dz\\
            &=p(x).
            \end{aligned}
            \]

            K 个独立 weight 的平均仍然无偏。再由 log 的凹性和 Jensen：

            \[
            E\log\left(\frac1K\sum_kw_k\right)
            \le
            \log E\left(\frac1K\sum_kw_k\right)
            =\log p(x).
            \]

            所以 \(\mathcal L_K\) 是下界。\(K=1\) 时，
            \(E_q[\log p(x,z)-\log q(z|x)]\) 正好退化为普通 ELBO。
            “\(\mathcal L_K\) 随 K 单调不减”的完整证明还需要对不同 K 的随机子集取期望，
            不能只凭“样本更多应该更好”代替。
            """),
        ],
        gap=0.8,
    )
    return


@app.cell
def _(mo):
    sample_count = mo.ui.slider(1, 100, value=5, step=1, show_value=True, label="importance samples K")
    resample_seed = mo.ui.number(value=7, start=0, stop=9999, step=1, label="随机种子")
    return resample_seed, sample_count


@app.cell
def _(COLORS, mo, np, plt, resample_seed, sample_count):
    _rng = np.random.default_rng(int(resample_seed.value))
    # toy latent: q Bernoulli(0.35); joint p(x,z) 两个正权重
    _q = np.array([0.65, 0.35])
    _joint = np.array([0.08, 0.42])
    _evidence = float(_joint.sum())
    _z = _rng.choice(2, size=sample_count.value, p=_q)
    _weights = _joint[_z] / _q[_z]
    _estimate = float(np.log(np.mean(_weights)))
    _trials = []
    for _k in range(1, 101):
        _values = []
        for _ in range(1000):
            _samples = _rng.choice(2, size=_k, p=_q)
            _values.append(np.log(np.mean(_joint[_samples] / _q[_samples])))
        _trials.append(np.mean(_values))
    _fig, _axes = plt.subplots(1, 2, figsize=(10, 3.7))
    _axes[0].bar(np.arange(len(_weights)), _weights, color=COLORS["data"])
    _axes[0].axhline(_evidence, linestyle="--", color=COLORS["success"], label="p(x)")
    _axes[0].set_title("本次 importance weights")
    _axes[0].legend(fontsize=8)
    _axes[1].plot(range(1, 101), _trials, color=COLORS["model"], label="E[L_K] Monte Carlo")
    _axes[1].axhline(np.log(_evidence), linestyle="--", color=COLORS["success"], label="log p(x)")
    _axes[1].legend(fontsize=8)
    _fig.tight_layout()
    mo.vstack([mo.md(fr"""## 交互实验

    本次 \(\mathcal L_K={_estimate:.4f}\)，真实 \(\log p(x)={np.log(_evidence):.4f}\)。
    单次估计会波动；右图用 1000 次重复近似其期望，支持下界随 K 变紧。"""),
    mo.hstack([sample_count, resample_seed], widths="equal"), _fig])
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 稳定代码

    ```python
    # log_w shape: [K, batch]，每行对应一个 posterior sample。
    log_w = log_p_x_given_z + log_p_z - log_q_z_given_x

    # 沿 K 轴做 log(mean(exp(log_w)))。
    log_mean_weight = torch.logsumexp(log_w, dim=0) - math.log(K)

    # 对 batch 求平均，最大化该值；训练 loss 取负号。
    loss = -log_mean_weight.mean()
    ```

    ## 错误与反例

    - 直接 `exp(log_w).mean().log()`：大权重会溢出。
    - q 在重要区域为 0：importance ratio 无法修复 proposal 完全漏掉的支持集。
    - 宣称每次增加 K 都必然提高当前 batch：单调性是期望性质。
    - 认为 K 无限大总是免费：计算、内存增加，encoder 梯度信噪比也可能恶化。

    原始资料：[Burda et al., Importance Weighted Autoencoders](https://arxiv.org/abs/1509.00519)
    """)
    return


@app.cell
def _(exercise_block, mo):
    mo.vstack(
        [
            mo.md("## 分层练习"),
            exercise_block(
                ("importance weight 大表示什么？", "该 z 在联合模型下能很好解释 x，但 q 给它的采样概率相对较低。"),
                ("log weights 为 2、3，写出 K=2 的稳定计算。", "取 `m=3`，则 `logmeanexp = 3 + log(exp(-1)+1) - log(2) ≈ 2.620`。先减最大值可避免直接计算大指数时溢出。"),
                ("若 `log_w` shape 为 `[K,B]`，用 `torch.logsumexp` 写出稳定的 batch 平均 IWAE bound。", "`bound = (torch.logsumexp(log_w, dim=0) - math.log(K)).mean(dim=0)`。先沿 posterior sample 轴 `K` 得到每个样本的 log-mean-weight `[B]`，再沿 batch 平均。"),
                ("改变 seed，观察单次 L_K 是否单调。", "通常不严格单调，这正好区分单次随机值与期望定理。"),
            ),
        ],
        gap=0.8,
    )
    return


@app.cell
def _(CHAPTERS, chapter_footer):
    chapter_footer(CHAPTERS[10], ["IWAE 用多个 importance samples 构造更紧的期望下界。", "无偏 evidence 估计经过 log 后不再无偏。", "log-sum-exp 是必要的数值稳定技巧。", "下一章加入条件 y 控制生成内容。"])
    return


if __name__ == "__main__":
    app.run()
