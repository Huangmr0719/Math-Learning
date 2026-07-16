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
    import torch
    from src.models import (
        TinyNoiseMLP,
        make_torch_schedule,
        q_sample_torch,
        sample_eight_gaussians,
        train_toy_ddpm,
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
        TinyNoiseMLP,
        chapter_footer,
        chapter_header,
        derivation_map,
        exercise_block,
        intuition_and_rigor,
        make_torch_schedule,
        mo,
        plt,
        q_sample_torch,
        sample_eight_gaussians,
        torch,
        train_toy_ddpm,
    )


@app.cell
def _(CHAPTERS, chapter_header):
    chapter_header(CHAPTERS[18], duration="100–160 分钟")
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 1. 本章为什么存在

    前四章已经分别得到：

    - forward Markov chain；
    - 任意时间的一步加噪公式；
    - reverse Gaussian posterior；
    - noise prediction MSE。

    本章把它们组装成一个最小 DDPM。为了让 CPU 也能快速观察完整闭环，
    数据不是大型图片，而是程序生成的二维八团分布：

    \[
    \text{八团数据}
    \rightarrow \text{随机时间加噪}
    \rightarrow \text{预测噪声}
    \rightarrow \text{从 Gaussian 逐步反向采样}.
    \]

    ## 2. 你已经知道什么

    \[
    x_t=\sqrt{\bar\alpha_t}x_0+
    \sqrt{1-\bar\alpha_t}\epsilon,
    \]

    \[
    L_{\mathrm{simple}}
    =\mathbb E\|\epsilon-\epsilon_\theta(x_t,t)\|^2.
    \]

    **回忆问题：** 为什么训练时可以随机抽一个 t，而不必按顺序生成全部中间状态？
    """)
    return


@app.cell
def _(intuition_and_rigor):
    intuition_and_rigor(
        r"""
        网络像一个修复工：它不仅看到受损物品 \(x_t\)，还收到“损坏等级” t。
        训练时随机练习各种损坏等级；生成时从最严重的噪声开始，按等级倒序修复。
        """,
        r"""
        模型 \(\epsilon_\theta:\mathbb R^2\times\{0,\ldots,T-1\}\to\mathbb R^2\)
        接收二维 noisy sample 和离散时间，输出同 shape 的噪声预测。

        Reverse step 使用

        \[
        \mu_\theta(x_t,t)=
        \frac{1}{\sqrt{\alpha_t}}
        \left(
        x_t-\frac{\beta_t}{\sqrt{1-\bar\alpha_t}}
        \epsilon_\theta(x_t,t)
        \right).
        \]

        对 \(t>0\) 再加入 \(\sqrt{\tilde\beta_t}z\)；到最后一步不加噪声。
        """,
    )
    return


@app.cell
def _(derivation_map, mo):
    mo.vstack(
        [
            mo.md("## 3–5. 最小 DDPM 完整数据流"),
            derivation_map(
                [
                    "采样干净 x0",
                    "随机采样时间 t",
                    "采样真实 epsilon",
                    "闭式构造 xt",
                    "网络预测 epsilon",
                    "MSE 更新参数",
                    "从 xT 倒序执行 reverse step",
                ]
            ),
        ],
        gap=0.8,
    )
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### 新数学：sinusoidal time embedding

    t 原本只是一个整数。如果直接把大整数输入网络，数值尺度不理想，也难以表达不同频率的
    时间变化。常见方法把 t 映射成多组正弦和余弦：

    \[
    \operatorname{emb}(t)
    =[\sin(\omega_1t),\cos(\omega_1t),\ldots,
      \sin(\omega_kt),\cos(\omega_kt)].
    \]

    日常直觉：钟表的时针、分针、秒针都用周期位置表示时间；不同频率共同区分时刻。

    严格地说，本章 embedding 是一个确定函数，不含可训练随机性。网络将
    `[x coordinate, y coordinate, time embedding]` 拼接后预测二维噪声。

    ### shape 表

    | 对象 | shape | 含义 |
    |---|---|---|
    | `x0`, `xt`, `epsilon` | `[B, 2]` | 二维样本或噪声 |
    | `t` | `[B]` | 每个样本自己的整数时间 |
    | `time_embedding` | `[B, time_dim]` | 时间特征 |
    | `predicted_noise` | `[B, 2]` | 与 epsilon 对齐 |
    | `loss` | scalar | 对 batch 和二维 feature 平均 |
    """)
    return


@app.cell
def _(mo):
    training_steps = mo.ui.dropdown(
        options={
            "快速演示：300 updates": 300,
            "标准实验：700 updates": 700,
            "更充分：1200 updates": 1200,
        },
        value="快速演示：300 updates",
        label="训练强度",
    )
    diffusion_steps = mo.ui.dropdown(
        options={"40 reverse steps": 40, "80 reverse steps": 80},
        value="80 reverse steps",
        label="扩散步数 T",
    )
    ddpm_seed = mo.ui.number(
        value=7, start=0, stop=9999, step=1, label="训练随机种子"
    )
    train_ddpm_button = mo.ui.run_button(label="显式训练二维 DDPM")
    trajectory_stage = mo.ui.slider(
        0, 4, value=0, step=1, show_value=True, label="反向轨迹阶段（0=最噪）"
    )
    return (
        ddpm_seed,
        diffusion_steps,
        train_ddpm_button,
        training_steps,
        trajectory_stage,
    )


@app.cell
def _(
    ddpm_seed,
    diffusion_steps,
    train_ddpm_button,
    train_toy_ddpm,
    training_steps,
):
    # 打开 notebook 时按钮值为 False，因此不自动训练。
    # 用户点击后，才在本地程序生成的数据上执行轻量实验；不发生网络下载。
    ddpm_result = (
        train_toy_ddpm(
            training_steps=training_steps.value,
            diffusion_steps=diffusion_steps.value,
            seed=int(ddpm_seed.value),
        )
        if train_ddpm_button.value
        else None
    )
    return (ddpm_result,)


@app.cell
def _(
    COLORS,
    ddpm_result,
    ddpm_seed,
    diffusion_steps,
    mo,
    plt,
    sample_eight_gaussians,
    torch,
    train_ddpm_button,
    training_steps,
    trajectory_stage,
):
    _controls = mo.vstack(
        [
            mo.hstack(
                [training_steps, diffusion_steps, ddpm_seed, train_ddpm_button],
                widths="equal",
            ),
            trajectory_stage,
        ],
        gap=0.6,
    )
    if ddpm_result is None:
        _preview_data = sample_eight_gaussians(
            512, seed=int(ddpm_seed.value)
        ).numpy()
        _generator = torch.Generator().manual_seed(int(ddpm_seed.value))
        _preview_noise = torch.randn(
            (512, 2), generator=_generator
        ).numpy()
        _preview_fig, _preview_axes = plt.subplots(1, 2, figsize=(7.5, 3.2))
        _preview_axes[0].scatter(
            _preview_data[:, 0],
            _preview_data[:, 1],
            s=5,
            alpha=0.4,
            color=COLORS["data"],
        )
        _preview_axes[0].set_title(r"目标数据 $x_0$：八个模式")
        _preview_axes[1].scatter(
            _preview_noise[:, 0],
            _preview_noise[:, 1],
            s=5,
            alpha=0.4,
            color=COLORS["prior"],
        )
        _preview_axes[1].set_title(r"起点 $x_T$：Gaussian noise")
        for _axis in _preview_axes:
            _axis.set_xlim(-3.5, 3.5)
            _axis.set_ylim(-3.5, 3.5)
            _axis.set_aspect("equal")
        _preview_fig.tight_layout()
        _view = mo.vstack(
            [
                mo.callout(
                    mo.md(
                        """
                        训练尚未执行。先比较目标分布与 Gaussian 起点，再阅读公式和核心代码。
                        点击按钮后才会训练；打开或导出 notebook 不会自动执行长时间计算。
                        """
                    ),
                    kind="warn",
                ),
                _preview_fig,
            ],
            gap=0.6,
        )
    else:
        _keys = sorted(ddpm_result.trajectory, reverse=True)
        _selected_key = _keys[trajectory_stage.value]
        _trajectory_points = ddpm_result.trajectory[_selected_key].numpy()
        _data = ddpm_result.data.numpy()
        _samples = ddpm_result.samples.numpy()

        _fig, _axes = plt.subplots(1, 4, figsize=(13, 3.4))
        _axes[0].plot(ddpm_result.losses, color=COLORS["data"], alpha=0.8)
        if len(ddpm_result.losses) >= 30:
            _window = 30
            _moving = [
                sum(ddpm_result.losses[max(0, _i - _window + 1) : _i + 1])
                / min(_i + 1, _window)
                for _i in range(len(ddpm_result.losses))
            ]
            _axes[0].plot(_moving, color=COLORS["model"], linewidth=2)
        _axes[0].set_title("noise prediction MSE")
        _axes[0].set_xlabel("update")

        _axes[1].scatter(
            _data[:, 0], _data[:, 1], s=4, alpha=0.35, color=COLORS["data"]
        )
        _axes[1].set_title(r"训练数据 $x_0$")
        _axes[2].scatter(
            _trajectory_points[:, 0],
            _trajectory_points[:, 1],
            s=4,
            alpha=0.35,
            color=COLORS["model"],
        )
        _axes[2].set_title(f"reverse 后 step={_selected_key}")
        _axes[3].scatter(
            _samples[:, 0],
            _samples[:, 1],
            s=4,
            alpha=0.35,
            color=COLORS["prior"],
        )
        _axes[3].set_title("最终生成样本")
        for _ax in _axes[1:]:
            _ax.set_xlim(-3.5, 3.5)
            _ax.set_ylim(-3.5, 3.5)
            _ax.set_aspect("equal")
        _fig.tight_layout()

        _view = mo.vstack(
            [
                mo.md(
                    fr"""
                    训练完成：device=`{ddpm_result.device}`，
                    updates=`{len(ddpm_result.losses)}`，
                    最后一次 MSE=`{ddpm_result.losses[-1]:.4f}`。

                    拖动“反向轨迹阶段”可单步查看粒子怎样从 Gaussian 逐渐组织成八个团。
                    快速模式只要求看见趋势，不保证每个团都达到论文级质量。
                    """
                ),
                _fig,
            ]
        )

    mo.vstack(
        [
            mo.md("## 7. 交互实验：显式训练与 reverse trajectory"),
            _controls,
            _view,
        ]
    )
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 8. 教学版训练代码

    ```python
    # 1. 每个 batch 样本独立选择时间。
    # t shape: [batch_size]，取值为 0,...,T-1。
    t = torch.randint(0, T, (batch_size,), device=device)

    # 2. 从程序生成的数据中取得 x0，shape: [batch_size, 2]。
    x0 = data[random_indices]

    # 3. 生成监督标签 epsilon；与 x0 shape 完全相同。
    epsilon = torch.randn_like(x0)

    # 4. 根据每个样本自己的 alpha_bar_t，一步构造 xt。
    # 系数 reshape 为 [batch_size, 1]，沿二维 feature 广播。
    x_t = (
        torch.sqrt(alpha_bar_t) * x0
        + torch.sqrt(1.0 - alpha_bar_t) * epsilon
    )

    # 5. 网络同时读取 noisy point 与 time embedding。
    predicted_noise = model(x_t, t)

    # 6. 对 batch 和两个坐标共同平均，得到 scalar loss。
    loss = F.mse_loss(predicted_noise, epsilon)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    ```

    ## Reverse sampling 核心

    ```python
    # 从最噪时刻开始。x shape: [sample_count, 2]。
    x = torch.randn(sample_count, 2)

    # 时间方向必须从 T-1 走向 0。
    for t in reversed(range(T)):
        predicted_noise = model(x, time_batch)

        # 由 epsilon_theta 构造 reverse Gaussian mean。
        mean = (
            x
            - beta_t / torch.sqrt(1.0 - alpha_bar_t)
            * predicted_noise
        ) / torch.sqrt(alpha_t)

        if t > 0:
            # posterior_variance = beta_tilde_t。
            # 每一步重新采样 z，保持 DDPM 的随机生成。
            z = torch.randn_like(x)
            x = mean + torch.sqrt(posterior_variance_t) * z
        else:
            # 最后一步不再加噪声。
            x = mean
    ```

    `src/models/diffusion.py` 保存完整训练器；本章保留的代码展示数学主干，
    设备选择、轨迹记录等工程细节放在公共模块中。
    """)
    return


@app.cell
def _(
    TinyNoiseMLP,
    make_torch_schedule,
    mo,
    q_sample_torch,
    sample_eight_gaussians,
    torch,
):
    _schedule = make_torch_schedule(steps=12)
    _x0 = sample_eight_gaussians(16, seed=3)
    _t = torch.arange(16) % _schedule.steps
    _xt, _epsilon = q_sample_torch(_x0, _t, _schedule)
    _model = TinyNoiseMLP()
    _prediction = _model(_xt, _t, _schedule.steps)
    _loss = torch.mean((_prediction - _epsilon) ** 2)
    _all_finite = bool(
        torch.isfinite(_xt).all()
        and torch.isfinite(_prediction).all()
        and torch.isfinite(_loss)
    )

    mo.md(
        fr"""
        ## 6、9. 未训练模型的 shape 与数值烟测

        - x0 shape：`{tuple(_x0.shape)}`
        - t shape：`{tuple(_t.shape)}`
        - xt / epsilon shape：`{tuple(_xt.shape)}`
        - predicted noise shape：`{tuple(_prediction.shape)}`
        - loss 是标量：`{_loss.ndim == 0}`
        - 所有值有限：`{_all_finite}`
        - alpha_bar 单调下降：`{bool(torch.all(_schedule.alpha_bars[1:] < _schedule.alpha_bars[:-1]))}`

        这只验证数据流、shape 和定义域。未训练网络的有限 loss 不表示已经学会生成。
        """
    )
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 10. 错误与反例

    1. **打开 notebook 就自动训练。**
       会破坏可复现和轻量原则；本章只在点击按钮后运行。

    2. **训练时固定同一个 t。**
       网络只学会一种噪声强度，reverse chain 的其他步骤没有可靠预测。

    3. **每一步都使用 \(\sqrt{\beta_t}\) 作为 reverse noise。**
       这混淆了 forward variance 与 posterior variance \(\tilde\beta_t\)。

    4. **只看 loss，不看生成轨迹。**
       MSE 下降不保证采样循环、时间方向和 schedule 索引都正确。

    5. **把二维成功直接等同于图像模型成功。**
       图像还需要更强网络、空间结构、更多训练和更复杂数据；数学主干相同，工程难度不同。

    ## 原始资料

    - [Ho et al., Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2006.11239)
    - 工程参考：[Hugging Face Diffusion Models from Scratch](https://github.com/huggingface/diffusion-models-class/blob/main/unit1/02_diffusion_models_from_scratch.ipynb)
    """)
    return


@app.cell
def _(exercise_block, mo):
    mo.vstack(
        [
            mo.md("## 11. 分层练习"),
            exercise_block(
                (
                    "从 x0 到 loss，用一句话串起训练链路。",
                    "随机抽 t 与 epsilon，由闭式公式构造 xt，网络根据 xt 和 t 预测 epsilon，再用 MSE 更新参数。",
                ),
                (
                    "若 batch=128、每个样本二维，predicted_noise 应是什么 shape？",
                    "`[128, 2]`；它必须与真实 epsilon 完全一致，才能逐元素计算误差。",
                ),
                (
                    "为什么 reverse sampling 的 `if t > 0` 不能删除？",
                    "t=0 时理论 posterior variance 为 0；继续注入噪声会破坏最终生成结果。",
                ),
                (
                    "比较 300 与 1200 updates，并观察 loss 和八团结构。哪些变化才算真正改进？",
                    "不仅看 loss 更低，还要看粒子从噪声形成多个清晰模式、覆盖八个团且不过度集中；随机实验需固定 seed 公平比较。",
                ),
            ),
        ],
        gap=0.8,
    )
    return


@app.cell
def _(CHAPTERS, chapter_footer):
    chapter_footer(
        CHAPTERS[18],
        [
            "最小 DDPM 由随机时间加噪、noise MSE 和倒序 Gaussian sampling 组成。",
            "时间 embedding 让同一个网络处理不同噪声强度。",
            "训练和采样必须逐项检查 shape、schedule 索引与时间方向。",
            "DDPM 已形成闭环，但同一网络并不只允许这一种采样轨迹。",
        ],
    )
    return


if __name__ == "__main__":
    app.run()
