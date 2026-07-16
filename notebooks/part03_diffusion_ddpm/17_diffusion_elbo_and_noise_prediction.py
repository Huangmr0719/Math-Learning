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
    from src.math_checks import (
        ddpm_mean_from_noise,
        ddpm_posterior_mean_variance,
        diffusion_coefficients,
        isotropic_gaussian_kl_same_variance,
        q_sample_from_x0,
    )
    from src.teaching import (
        CHAPTERS,
        chapter_footer,
        chapter_header,
        derivation_map,
        exercise_block,
        intuition_and_rigor,
    )
    from src.visualization import COLORS, configure_matplotlib

    configure_matplotlib()
    import matplotlib.pyplot as plt

    return (
        CHAPTERS,
        COLORS,
        chapter_footer,
        chapter_header,
        ddpm_mean_from_noise,
        ddpm_posterior_mean_variance,
        derivation_map,
        diffusion_coefficients,
        exercise_block,
        intuition_and_rigor,
        isotropic_gaussian_kl_same_variance,
        mo,
        np,
        plt,
        q_sample_from_x0,
    )


@app.cell
def _(CHAPTERS, chapter_header):
    chapter_header(CHAPTERS[17], duration="110–150 分钟")
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 1. 本章为什么存在

    第 16 章已经知道理想 reverse posterior 的形式，但网络究竟应该输出什么？
    如果直接让网络预测一整个概率分布，训练目标又怎样得到？

    DDPM 从类似 VAE 的 variational lower bound 出发，最终得到非常简单的训练任务：

    > 随机选择时间 t，给数据加入一份已知噪声，让网络把这份噪声猜出来。

    \[
    L_{\mathrm{simple}}
    =\mathbb E\left[
    \|\epsilon-\epsilon_\theta(x_t,t)\|^2
    \right].
    \]

    本章要严谨说明：这个 MSE 从哪里来，以及它与完整 ELBO 哪里相同、哪里不同。

    ## 2. 你已经知道什么

    - 第 5 章：ELBO 把难算的 log likelihood 变成可优化下界。
    - 第 4、6 章：Gaussian KL 在协方差固定时可解析计算。
    - 第 15 章：可以由 \(x_0,\epsilon,t\) 一步构造 \(x_t\)。
    - 第 16 章：真实 reverse posterior 的 mean 可解析得到。

    **回忆问题：** 数值相同的两个 Gaussian，若方差固定，它们的差异主要由什么决定？
    """)
    return


@app.cell
def _(intuition_and_rigor):
    intuition_and_rigor(
        r"""
        加噪过程像老师自己把一段干净语音混入已知杂音。因为老师保留了杂音原稿，
        可以直接要求学生指出“刚才加入了哪段杂音”，无需事先准备人工标签。
        这是一种 self-supervised training target。
        """,
        r"""
        Negative ELBO 可分解为终点 prior matching、各时间步 reverse KL 和最终
        reconstruction 三类项。对 \(t\ge2\)，若 model variance 固定，则

        \[
        D_{KL}\bigl(q(x_{t-1}|x_t,x_0)\|p_\theta(x_{t-1}|x_t)\bigr)
        =C_t\|\epsilon-\epsilon_\theta(x_t,t)\|^2,
        \]

        其中 \(C_t>0\) 只由 schedule 与 model variance 决定。
        原始 DDPM 的 simple loss 去掉了这些时间权重，因此是重加权 surrogate，
        不能不加条件地称为“与完整 ELBO 完全相等”。
        """,
    )
    return


@app.cell
def _(derivation_map, mo):
    mo.md("## 3–5. 从 Diffusion ELBO 到 epsilon MSE")
    derivation_map(
        [
            "展开 reverse joint probability",
            "把 negative ELBO 分成逐步 KL",
            "固定 reverse variance",
            "Gaussian KL 化成 mean squared error",
            "用 epsilon 参数化 reverse mean",
            "得到加权 noise MSE",
            "定义简化的 unweighted loss",
        ]
    )
    return


@app.cell
def _(mo):
    mo.md(r"""
    定义 reverse model：

    \[
    p_\theta(x_{0:T})
    =p(x_T)\prod_{t=1}^T p_\theta(x_{t-1}\mid x_t).
    \]

    经过与 VAE ELBO 相同的“加上并减去 forward posterior、使用 Bayes rule”
    的整理，negative ELBO 可写成：

    \[
    \begin{aligned}
    L_{\mathrm{VLB}}
    =&\ \mathbb E_q\Big[
    D_{KL}(q(x_T|x_0)\|p(x_T))\\
    &+\sum_{t=2}^{T}
    D_{KL}(q(x_{t-1}|x_t,x_0)\|
    p_\theta(x_{t-1}|x_t))\\
    &-\log p_\theta(x_0|x_1)
    \Big].
    \end{aligned}
    \]

    这里写的是要**最小化的 negative ELBO**。符号若写成要最大化的 ELBO，
    整体正负号会相反。

    ### epsilon 参数化

    由第 15 章：

    \[
    x_t=\sqrt{\bar\alpha_t}x_0+
    \sqrt{1-\bar\alpha_t}\epsilon,
    \]

    所以

    \[
    x_0=
    \frac{x_t-\sqrt{1-\bar\alpha_t}\epsilon}
         {\sqrt{\bar\alpha_t}}.
    \]

    将它代入第 16 章 posterior mean，可化简为

    \[
    \tilde\mu_t=
    \frac{1}{\sqrt{\alpha_t}}
    \left(
    x_t-\frac{\beta_t}{\sqrt{1-\bar\alpha_t}}\epsilon
    \right).
    \]

    用网络预测 \(\epsilon_\theta(x_t,t)\) 替换真实 \(\epsilon\)，便得到
    \(\mu_\theta(x_t,t)\)。

    若 model variance 为 \(\sigma_t^2I\)，对应 KL 的参数相关部分为

    \[
    \frac{\beta_t^2}
    {2\sigma_t^2\alpha_t(1-\bar\alpha_t)}
    \|\epsilon-\epsilon_\theta(x_t,t)\|^2.
    \]

    ### shape 检查

    - `epsilon`, `predicted_noise`, `xt`, `x0` shape 相同；
    - MSE 先对 feature/pixel 维求和或平均，再对 batch 与随机 t 取期望；
    - t 必须输入网络，否则同一个 xt 在不同噪声强度下含义不明确；
    - simple loss 删除的是正的时间权重，不是所有实验下都无影响的常数。
    """)
    return


@app.cell
def _(mo):
    elbo_t = mo.ui.slider(
        2, 100, value=55, step=1, show_value=True, label="时间 t"
    )
    predicted_epsilon = mo.ui.slider(
        -2.5,
        2.5,
        value=0.0,
        step=0.1,
        show_value=True,
        label="网络预测 epsilon_theta",
    )
    return elbo_t, predicted_epsilon


@app.cell
def _(
    COLORS,
    ddpm_mean_from_noise,
    ddpm_posterior_mean_variance,
    diffusion_coefficients,
    elbo_t,
    isotropic_gaussian_kl_same_variance,
    mo,
    np,
    plt,
    predicted_epsilon,
    q_sample_from_x0,
):
    _betas = np.linspace(1e-4, 0.02, 100)
    _alphas, _alpha_bars = diffusion_coefficients(_betas)
    _index = elbo_t.value - 1
    _x0 = np.array([1.2])
    _true_epsilon = np.array([-0.8])
    _xt = q_sample_from_x0(_x0, float(_alpha_bars[_index]), _true_epsilon)

    _true_mean, _posterior_variance = ddpm_posterior_mean_variance(
        _x0,
        _xt,
        alpha_t=float(_alphas[_index]),
        alpha_bar_t=float(_alpha_bars[_index]),
        alpha_bar_previous=float(_alpha_bars[_index - 1]),
        beta_t=float(_betas[_index]),
    )
    _model_mean = ddpm_mean_from_noise(
        _xt,
        np.array([predicted_epsilon.value]),
        alpha_t=float(_alphas[_index]),
        alpha_bar_t=float(_alpha_bars[_index]),
        beta_t=float(_betas[_index]),
    )
    _kl = isotropic_gaussian_kl_same_variance(
        _true_mean, _model_mean, _posterior_variance
    )
    _weight = (
        _betas[_index] ** 2
        / (
            2
            * _posterior_variance
            * _alphas[_index]
            * (1 - _alpha_bars[_index])
        )
    )
    _weighted_mse = _weight * float(
        (_true_epsilon[0] - predicted_epsilon.value) ** 2
    )

    _epsilon_grid = np.linspace(-2.5, 2.5, 300)
    _simple_curve = (_epsilon_grid - _true_epsilon[0]) ** 2
    _weighted_curve = _weight * _simple_curve
    _fig, _axes = plt.subplots(1, 2, figsize=(10, 3.8))
    _axes[0].plot(
        _epsilon_grid, _simple_curve, label="simple MSE", color=COLORS["data"]
    )
    _axes[0].plot(
        _epsilon_grid,
        _weighted_curve,
        label="该 t 的 VLB 权重 × MSE",
        color=COLORS["model"],
    )
    _axes[0].axvline(_true_epsilon[0], color=COLORS["success"], linestyle="--")
    _axes[0].scatter(
        [predicted_epsilon.value],
        [(predicted_epsilon.value - _true_epsilon[0]) ** 2],
        color=COLORS["danger"],
    )
    _axes[0].legend(fontsize=8)
    _axes[0].set_xlabel("epsilon_theta")
    _axes[0].set_title("同一时间步的最小点相同")

    _mean = np.sqrt(_alpha_bars[_index]) * _x0[0]
    _variance = 1 - _alpha_bars[_index]
    _x_grid = np.linspace(_mean - 4 * np.sqrt(_variance), _mean + 4 * np.sqrt(_variance), 400)
    _density = np.exp(-0.5 * (_x_grid - _mean) ** 2 / _variance) / np.sqrt(
        2 * np.pi * _variance
    )
    _score = -(_x_grid - _mean) / _variance
    _axes[1].plot(_x_grid, _density, color=COLORS["prior"], label="q(xt|x0)")
    _axes[1].quiver(
        _x_grid[::20],
        np.zeros_like(_x_grid[::20]),
        _score[::20],
        np.zeros_like(_score[::20]),
        angles="xy",
        scale_units="xy",
        scale=8,
        color=COLORS["model"],
        width=0.004,
    )
    _axes[1].set_title("score 指向高密度区域")
    _axes[1].legend(fontsize=8)
    _fig.tight_layout()

    mo.vstack(
        [
            mo.md(
                fr"""
                ## 7、9. KL 与加权噪声 MSE 的数值等价

                固定真实 \(\epsilon={_true_epsilon[0]}\)：

                - Gaussian KL（相同 variance）：`{_kl:.8f}`
                - 推导得到的时间权重 × noise MSE：`{_weighted_mse:.8f}`
                - absolute error：`{abs(_kl-_weighted_mse):.2e}`

                这里验证的是单个时间步、固定相同 covariance 的参数相关项。
                """
            ),
            mo.hstack([elbo_t, predicted_epsilon], widths="equal"),
            _fig,
        ]
    )
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 新数学：score

    Score 不是“分数”，而是 log density 对输入的梯度：

    \[
    s(x)=\nabla_x\log p(x).
    \]

    在一维中它就是 log density 曲线的斜率；它通常指向 density 增大的方向。
    对 conditional Gaussian：

    \[
    \nabla_{x_t}\log q(x_t|x_0)
    =-\frac{x_t-\sqrt{\bar\alpha_t}x_0}
           {1-\bar\alpha_t}
    =-\frac{\epsilon}{\sqrt{1-\bar\alpha_t}}.
    \]

    网络预测噪声与预测 conditional score 只差一个已知时间缩放。
    对真实 noisy marginal \(q_t(x_t)\)，最优 MSE 预测器给出
    \(\mathbb E[\epsilon|x_t]\)，它对应 marginal score，而不是某个未知
    单一样本 \(x_0\) 的 conditional score。

    ## 8. 教学版训练代码

    ```python
    # x0 shape: [batch, feature...]。
    # 每个样本独立抽一个整数时间，shape: [batch]。
    t = torch.randint(0, T, (batch_size,), device=x0.device)

    # epsilon 是监督标签，由我们自己生成，因此无需人工标注。
    epsilon = torch.randn_like(x0)

    # alpha_bar_t reshape 后沿 feature 维广播。
    x_t = (
        torch.sqrt(alpha_bar_t) * x0
        + torch.sqrt(1.0 - alpha_bar_t) * epsilon
    )

    # 网络必须同时知道 noisy sample 与时间。
    predicted_noise = model(x_t, t)

    # 默认 mean 会对 batch 与全部 feature 共同平均，输出标量。
    loss = F.mse_loss(predicted_noise, epsilon)
    ```
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 10. 错误与反例

    1. **宣称 simple MSE 与完整 ELBO 完全相等。**
       它删除了时间相关权重。理想无限容量下各 t 的条件最优预测相同，但有限模型的权衡会改变。

    2. **网络只输入 xt，不输入 t。**
       同一个数值 xt 在不同噪声强度下需要不同解释，模型无法判断当前应去噪多少。

    3. **把预测 epsilon 与预测 score 当成数值完全相同。**
       两者相差 \(-1/\sqrt{1-\bar\alpha_t}\) 的时间缩放。

    4. **把采样损失相符称为证明 ELBO。**
       数值实验只检查特定实现；ELBO 分解仍依赖概率恒等式。

    ## 原始资料

    - [Ho et al., DDPM，公式 (5)、(8)–(14)](https://arxiv.org/abs/2006.11239)
    """)
    return


@app.cell
def _(exercise_block, mo):
    mo.md("## 11. 分层练习")
    exercise_block(
        (
            "为什么噪声预测可以 self-supervised？",
            "训练者自己采样 epsilon 并构造 xt，因此真实噪声标签天然已知，不需要人工标注。",
        ),
        (
            r"若真实 epsilon=1.2、预测为 0.7，单元素 simple MSE 是多少？",
            r"\((1.2-0.7)^2=0.25\)。",
        ),
        (
            "删除 model 的时间输入会造成什么歧义？",
            "模型无法区分轻噪声和重噪声状态，也不知道应采用哪一组 schedule 系数。",
        ),
        (
            "拖动 t，观察 VLB 权重曲线相对 simple MSE 如何变化。",
            "两条曲线在固定 t 下最小点相同，但尺度随 t 改变；跨时间训练时这会改变有限容量模型的关注重点。",
        ),
    )
    return


@app.cell
def _(CHAPTERS, chapter_footer):
    chapter_footer(
        CHAPTERS[17],
        [
            "Diffusion negative ELBO 可分成终点 KL、逐步 reverse KL 和重构项。",
            "固定 variance 后，逐步 Gaussian KL 化成加权 noise MSE。",
            "simple loss 删除时间权重，是实用 surrogate 而非无条件完全等价。",
            "预测 epsilon 与预测 score 通过已知时间缩放相连。",
        ],
    )
    return


if __name__ == "__main__":
    app.run()
