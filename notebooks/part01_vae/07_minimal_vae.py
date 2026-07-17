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
    import torch
    from src.models import TinyVAE, load_digits_data, train_digits_vae, vae_loss
    from src.teaching import CHAPTERS, chapter_footer, chapter_header, course_styles, derivation_map, exercise_block, intuition_and_rigor
    from src.visualization import COLORS, configure_matplotlib

    configure_matplotlib()
    import matplotlib.pyplot as plt

    return (
        CHAPTERS,
        COLORS,
        TinyVAE,
        chapter_footer,
        chapter_header,
        course_styles,
        derivation_map,
        exercise_block,
        intuition_and_rigor,
        load_digits_data,
        mo,
        plt,
        torch,
        train_digits_vae,
        vae_loss,
    )


@app.cell
def _(CHAPTERS, chapter_header):
    chapter_header(CHAPTERS[7], duration="90–150 分钟")
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 本章为什么存在

    前六章已经分别得到：

    - encoder 应输出 approximate posterior 的参数；
    - ELBO 提供训练目标；
    - Gaussian KL 有闭式解；
    - reparameterization 让采样可训练。

    本章把它们组装成一个最小但完整的 VAE。使用 scikit-learn 自带的
    8×8 handwritten digits 数据，不联网、不下载大型模型。

    ## 你已经知道什么

    \[
    x\rightarrow(\mu,\log\sigma^2)
    \rightarrow z
    \rightarrow\hat{x}
    \]

    训练最小化：

    \[
    L=L_{\mathrm{reconstruction}}+L_{\mathrm{KL}}.
    \]

    **回忆问题：** encoder 为什么不能只输出一个确定的 \(z\)？
    """)
    return


@app.cell
def _(intuition_and_rigor):
    intuition_and_rigor(
        r"""
        Encoder 像给每张数字图片画一个“可能位置范围”，decoder 根据从范围中抽到的位置重建图片。
        重构项要求图片信息保留下来，KL 项要求这些范围整体靠近统一的标准正态地图。
        """,
        r"""
        本章采用
        \(q_\phi(z\mid x)=\mathcal N(\mu_\phi(x),
        \operatorname{diag}(\exp(\operatorname{logvar}_\phi(x))))\)，
        \(p(z)=\mathcal N(0,I)\)。

        decoder 输出 Bernoulli 参数 \(\hat{x}\in(0,1)^{64}\)，因此使用 binary cross entropy
        作为负 log likelihood 的实现。digits 像素已缩放到 \([0,1]\)。
        """,
    )
    return


@app.cell
def _(derivation_map, mo):
    mo.vstack(
        [
            mo.md("## 从公式到完整 forward"),
            derivation_map(["x 输入 encoder", "得到 mu 与 logvar", "重参数化采样 z", "decoder 得到 x_hat", "计算 reconstruction + KL", "反向传播更新参数"]),
        ],
        gap=0.8,
    )
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### 网络中的概率含义

    | 数学对象 | 代码模块 | shape |
    |---|---|---|
    | \(q_\phi(z\mid x)\) | encoder + `mu_head` + `logvar_head` | `[B, latent_dim]` |
    | \(z=\mu+\sigma\epsilon\) | `reparameterize` | `[B, latent_dim]` |
    | \(p_\theta(x\mid z)\) | decoder | `[B, 64]` |
    | reconstruction loss | BCE | scalar |
    | \(KL(q_\phi\|p)\) | Gaussian closed form | scalar |

    这里 \(B\) 是 batch size。输入 8×8 图片在进入全连接网络前展开成 64 维向量。

    ### 为什么本章用 BCE？

    Decoder 最后一层是 sigmoid，每个输出位于 \((0,1)\)。把每个归一化像素近似看成
    Bernoulli 参数时，negative log likelihood 对应 binary cross entropy。
    这是一种建模选择，不意味着真实灰度像素严格是 Bernoulli 随机变量。
    """)
    return


@app.cell
def _(mo):
    latent_dim = mo.ui.dropdown(options={"2（可视化）": 2, "4": 4, "8": 8}, value="2（可视化）", label="latent dimension")
    epochs = mo.ui.dropdown(options={"快速：6 epochs": 6, "标准：12 epochs": 12, "更充分：24 epochs": 24}, value="标准：12 epochs", label="训练轮数")
    train_button = mo.ui.run_button(label="显式开始本地训练")
    z1 = mo.ui.slider(-3.0, 3.0, step=0.1, value=0.0, show_value=True, label="生成坐标 z1")
    z2 = mo.ui.slider(-3.0, 3.0, step=0.1, value=0.0, show_value=True, label="生成坐标 z2")
    return epochs, latent_dim, train_button, z1, z2


@app.cell
def _(epochs, latent_dim, train_button, train_digits_vae):
    # 打开 notebook 时 train_button.value 为 False，因此不会自动训练。
    # 用户点击按钮后，此 cell 才调用本地训练器；不发生网络下载。
    training_result = (
        train_digits_vae(epochs=epochs.value, latent_dim=latent_dim.value)
        if train_button.value
        else None
    )
    return (training_result,)


@app.cell
def _(
    COLORS,
    epochs,
    latent_dim,
    load_digits_data,
    mo,
    plt,
    torch,
    train_button,
    training_result,
    z1,
    z2,
):
    _controls = mo.vstack(
        [
            mo.hstack([latent_dim, epochs, train_button], widths="equal"),
            mo.hstack([z1, z2], widths="equal"),
        ],
        gap=0.6,
    )
    if training_result is None:
        _preview_images, _preview_labels = load_digits_data()
        _preview_fig, _preview_axes = plt.subplots(2, 5, figsize=(7, 3.2))
        for _index, _axis in enumerate(_preview_axes.ravel()):
            _axis.imshow(_preview_images[_index].reshape(8, 8), cmap="gray")
            _axis.set_title(f"label={int(_preview_labels[_index])}")
            _axis.axis("off")
        _preview_fig.suptitle("离线 digits 数据：训练按钮按下前只展示数据，不训练模型")
        _preview_fig.tight_layout()
        _view = mo.vstack(
            [
                mo.callout(
                    mo.md(
                        """
                        训练尚未执行。先观察数据并阅读公式与代码，再点击按钮。
                        这样可以保证 notebook 打开时不会意外触发昂贵计算。
                        """
                    ),
                    kind="warn",
                ),
                _preview_fig,
            ],
            gap=0.6,
        )
    else:
        _history = training_result.history
        _model = training_result.model
        _images = training_result.images
        _labels = training_result.labels

        _model.eval()
        with torch.no_grad():
            _reconstruction, _mu, _logvar = _model(_images[:12])
            if latent_dim.value == 2:
                _z = torch.tensor([[z1.value, z2.value]], dtype=torch.float32)
            else:
                _z = torch.zeros((1, latent_dim.value), dtype=torch.float32)
                _z[0, 0] = z1.value
                _z[0, 1] = z2.value
            _generated = _model.decode(_z).reshape(8, 8).numpy()
            _all_mu, _all_logvar = _model.encode(_images)

        _fig, _axes = plt.subplots(2, 3, figsize=(10, 6))
        _axes[0, 0].plot(_history["total"], label="total", color=COLORS["data"])
        _axes[0, 0].plot(_history["reconstruction"], label="reconstruction", color=COLORS["model"])
        _axes[0, 0].plot(_history["kl"], label="KL", color=COLORS["prior"])
        _axes[0, 0].set_title("每个 epoch 的三类 loss")
        _axes[0, 0].legend(fontsize=8)

        _axes[0, 1].imshow(_images[0].reshape(8, 8), cmap="gray")
        _axes[0, 1].set_title("原始数字")
        _axes[0, 2].imshow(_reconstruction[0].reshape(8, 8), cmap="gray")
        _axes[0, 2].set_title("VAE 重构")
        _axes[1, 0].imshow(_generated, cmap="gray")
        _axes[1, 0].set_title(f"从 z=({z1.value:.1f},{z2.value:.1f}) 解码")

        if latent_dim.value == 2:
            _scatter = _axes[1, 1].scatter(
                _all_mu[:, 0],
                _all_mu[:, 1],
                c=_labels,
                s=8,
                cmap="tab10",
                alpha=0.65,
            )
            _axes[1, 1].set_title("所有样本的 posterior mean")
            _fig.colorbar(_scatter, ax=_axes[1, 1], fraction=0.046)
        else:
            _axes[1, 1].hist(_all_mu[:, 0].numpy(), bins=30, color=COLORS["data"])
            _axes[1, 1].set_title("latent 第 1 维的 mu")

        _axes[1, 2].hist(_all_logvar.detach().numpy().ravel(), bins=30, color=COLORS["prior"])
        _axes[1, 2].set_title("encoder 输出的 logvar")
        for _ax in _axes.ravel():
            _ax.tick_params(labelsize=7)
        _fig.tight_layout()
        _view = mo.vstack(
            [
                mo.md(
                    fr"""
                    训练完成：**{epochs.value} epochs**，latent dimension = **{latent_dim.value}**。
                    当前最终 total / reconstruction / KL 分别为
                    **{_history['total'][-1]:.3f} / {_history['reconstruction'][-1]:.3f} / {_history['kl'][-1]:.3f}**。
                    """
                ),
                _fig,
            ]
        )

    mo.vstack(
        [
            mo.md("## 交互实验：显式训练、重构与 latent 采样"),
            _controls,
            _view,
        ]
    )
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 教学版核心代码

    ```python
    def forward(self, x):
        # encoder 先把 64 维图片压到 hidden representation。
        hidden = torch.relu(self.encoder(x))

        # 两个 head 分别给出 q_phi(z|x) 的均值和对数方差。
        # shape: [batch_size, latent_dim]
        mu = self.mu_head(hidden)
        logvar = self.logvar_head(hidden)

        # logvar -> std -> epsilon -> z
        std = torch.exp(0.5 * logvar)
        epsilon = torch.randn_like(std)
        z = mu + std * epsilon

        # decoder 给出 p_theta(x|z) 的参数。
        reconstruction = self.decode(z)
        return reconstruction, mu, logvar
    ```

    ```python
    # reconstruction loss：先对 batch 和 64 个像素求和，
    # 再除以 batch_size，得到“每个样本平均重构损失”。
    reconstruction_loss = BCE(x_hat, x, reduction="sum") / x.shape[0]

    # KL 同样先对 batch 和 latent dimension 求和，再除以 batch_size。
    kl_loss = -0.5 * torch.sum(
        1 + logvar - mu.pow(2) - logvar.exp()
    ) / x.shape[0]

    total_loss = reconstruction_loss + kl_loss
    ```

    选择 `sum / batch_size` 而不是直接 `mean()`，是为了保留每张图片所有像素
    negative log likelihood 的总量。不同 reduction 都可以使用，但比较实验时必须一致。
    """)
    return


@app.cell
def _(TinyVAE, load_digits_data, mo, torch, vae_loss):
    _images, _labels = load_digits_data()
    _model = TinyVAE(latent_dim=2)
    _reconstruction, _mu, _logvar = _model(_images[:16])
    _total, _reconstruction_term, _kl_term = vae_loss(
        _reconstruction, _images[:16], _mu, _logvar
    )
    _finite = bool(
        torch.isfinite(_total)
        and torch.isfinite(_reconstruction_term)
        and torch.isfinite(_kl_term)
    )
    mo.md(
        fr"""
        ## shape 与数值烟测

        - input shape：`{tuple(_images[:16].shape)}`
        - reconstruction shape：`{tuple(_reconstruction.shape)}`
        - mu/logvar shape：`{tuple(_mu.shape)}`
        - total loss 是标量：`{_total.ndim == 0}`
        - 三个 loss 均为有限数：`{_finite}`

        这个测试验证 forward 与 loss 的 shape 和数值定义域；它不证明模型已经学会生成。
        """
    )
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 错误与反例

    - **去掉 KL：** 模型更像普通 Autoencoder，重构可能变好，但随机 prior 采样会失去依据。
    - **令 \(z=\mu\)：** 训练变成确定性编码，无法体验 posterior 的随机性。
    - **只打印 total loss：** 无法判断改进来自重构还是 KL，也难以诊断 posterior collapse。
    - **训练打开即运行：** 破坏教材的本地轻量原则，因此本章必须点击按钮才训练。
    """)
    return


@app.cell
def _(exercise_block, mo):
    mo.vstack(
        [
            mo.md("## 分层练习"),
            exercise_block(
                ("用一条完整链路解释一张图片如何经过 VAE。", "图片进入 encoder 得到 mu/logvar；重参数化采样 z；decoder 根据 z 输出重构；重构项与 KL 共同更新 encoder 和 decoder。"),
                ("若 reconstruction=40，KL=2，total 是多少？若 beta=4 呢？", "标准 VAE total=42；beta=4 时 total=40+4×2=48，这会更强调 posterior 接近 prior。"),
                ("补全日志字典：`metrics = {'total': ____, 'recon': ____, 'kl': ____}`；为什么不能只保存 total？", "分别填 `total_loss.item()`、`reconstruction_loss.item()`、`kl_loss.item()`。total 只显示总体结果；分项日志才能诊断重构—正则权衡及 KL 是否趋近 0。"),
                ("分别选择 latent dim 2、4、8 训练。比较重构、KL 和可视化难度。", "通常容量增加可能改善重构，但二维以上无法直接完整绘图；结果受训练轮数和随机性影响，应使用相同设置公平比较。"),
            ),
        ],
        gap=0.8,
    )
    return


@app.cell
def _(CHAPTERS, chapter_footer):
    chapter_footer(
        CHAPTERS[7],
        [
            "VAE 把 approximate posterior、重参数化和 decoder 连接成端到端模型。",
            "训练 loss 是 reconstruction 与 Gaussian KL 的和，即负 ELBO 的实现。",
            "公式到代码时必须追踪 batch、pixel、latent 三类维度及 reduction。",
            "重构、随机生成、latent map 和分项 loss 必须共同观察。",
        ],
    )
    return


if __name__ == "__main__":
    app.run()
