import marimo

__generated_with = "0.23.13"
app = marimo.App(width="medium")


@app.cell
def _():
    import sys as _sys
    from pathlib import Path as _Path
    _root=_Path(__file__).resolve().parents[2]
    if str(_root) not in _sys.path:_sys.path.insert(0,str(_root))
    import marimo as mo
    import numpy as np
    from src.teaching import CHAPTERS,chapter_footer,chapter_header,derivation_map,exercise_block,intuition_and_rigor
    from src.visualization import COLORS,configure_matplotlib
    configure_matplotlib();import matplotlib.pyplot as plt
    return CHAPTERS,COLORS,chapter_footer,chapter_header,derivation_map,exercise_block,intuition_and_rigor,mo,np,plt


@app.cell
def _(CHAPTERS,chapter_header):
    chapter_header(CHAPTERS[23],duration="75–105 分钟")
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 1. 本章为什么存在

    512×512 RGB 图像有 786,432 个数。在像素空间每步运行大型去噪网络代价很高。
    Latent Diffusion 先用 autoencoder 压缩：
    \[
    x\xrightarrow{E}z,\quad \text{diffusion in }z,\quad z_0\xrightarrow{D}\hat x.
    \]

    ## 2. 你已经知道什么

    - VAE/VQ-VAE 可把图像压到 latent。
    - DDPM 可在任意连续张量空间加噪。
    - **回忆：** 压缩为什么不可能无限强而完全无损？
    """)
    return


@app.cell
def _(intuition_and_rigor):
    intuition_and_rigor(
        "先把高清地图压成保留道路与地标的简图，在简图上规划路线，再还原为详细地图。计算变少，但压缩时丢掉的细节无法靠扩散凭空保证恢复。",
        r"""
        若空间下采样因子为 f，二维 token 数约下降 \(f^2\) 倍。训练目标在
        \(z_0=E(x)\) 上定义，而最终感知质量同时受 autoencoder distortion
        与 diffusion error 影响。Latent scaling 必须使训练时 z 的尺度与 noise schedule 匹配。
        """,
    )
    return


@app.cell
def _(derivation_map,mo):
    mo.md("## 3–6. 压缩率与感知失真")
    derivation_map(["图像编码为 z0","在 z 上训练 forward/noise prediction","从 Gaussian 采样 zT","latent reverse sampling","decoder 还原图像"])
    mo.md(r"""
    压缩率不是只有文件大小：
    \[
    r=\frac{C_zH_zW_z}{C_xH_xW_x}.
    \]
    Pixel MSE 小不一定感知相似；感知损失比较预训练特征，但也不是人类评价的严格替代。
    shape 必须区分像素 `[B,3,H,W]` 与 latent `[B,C_z,H/f,W/f]`。

    若 \(H,W\) 都缩小 f 倍，空间位置数从 \(HW\) 变成 \(HW/f^2\)。
    但总体计算并不严格只缩小 \(f^2\)：latent channels、网络宽度、
    attention 的 \(N^2\) 成本和 decoder 计算都要计入。

    Latent Diffusion 的总误差至少有两层：

    \[
    x\xrightarrow{E}z_0
    \xrightarrow{\text{diffusion sampling}}\hat z_0
    \xrightarrow{D}\hat x.
    \]

    即使 \(\hat z_0=z_0\)，仍可能有 autoencoder distortion
    \(D(E(x))\ne x\)；即使 autoencoder 很好，diffusion 也可能产生
    \(\hat z_0\ne z_0\)。诊断时必须分别做“纯重构”和“完整生成”实验。
    """)
    return


@app.cell
def _(mo):
    factor=mo.ui.dropdown(
        options={"1":1,"2":2,"4":4,"8":8,"16":16},
        value="8",
        label="spatial downsample factor f（必须整除 512）",
    )
    latent_channels=mo.ui.slider(2,16,value=4,step=1,show_value=True,label="latent channels")
    return factor,latent_channels


@app.cell
def _(COLORS,factor,latent_channels,mo,np,plt):
    _pixel=3*512*512
    _latent=latent_channels.value*(512//factor.value)**2
    _ratio=_latent/_pixel
    _f=np.array([1,2,4,8,16])
    _cost=latent_channels.value*(512/_f)**2
    _distortion=1-np.exp(-_f/12)
    _fig,_axes=plt.subplots(1,2,figsize=(9,3.8))
    _axes[0].plot(_f,_cost/_pixel,color=COLORS["data"]);_axes[0].axvline(factor.value,linestyle="--",color=COLORS["danger"])
    _axes[0].set_ylabel("latent scalars / pixel scalars")
    _axes[1].plot(_f,_distortion,color=COLORS["model"]);_axes[1].set_title("toy compression distortion")
    _fig.tight_layout()
    mo.vstack([mo.md(fr"""## 7、9. 算力—失真权衡

    512×512 输入标量数={_pixel:,}；当前 latent 标量数={_latent:,}；
    比例={_ratio:.4f}。真实显存/算力还受网络宽度与 attention 影响。"""),mo.hstack([factor,latent_channels],widths="equal"),_fig])
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 8. 数据流代码

    ```python
    with torch.no_grad():
        z0 = autoencoder.encode(images) * latent_scale
    zt, epsilon = q_sample(z0, t)
    epsilon_hat = denoiser(zt, t, condition)
    loss = F.mse_loss(epsilon_hat, epsilon)

    # 采样完成后必须除回同一 scale。
    images = autoencoder.decode(z0_generated / latent_scale)
    ```

    ## 10. 错误与反例

    - 忘记 latent scale：schedule 的信噪比不再符合设计。
    - 把 autoencoder 重构误差归咎于 diffusion。
    - 压缩率越大越好：语义或纹理会在扩散前永久丢失。

    原始资料：[Rombach et al., High-Resolution Image Synthesis with Latent Diffusion Models](https://arxiv.org/abs/2112.10752)
    """)
    return


@app.cell
def _(exercise_block,mo):
    mo.md("## 11. 分层练习")
    exercise_block(
        ("Latent Diffusion 节省什么？","主要减少去噪网络处理的空间位置与张量规模。"),
        ("f=8 时空间位置缩小多少倍？","每边缩小 8，位置数缩小 64 倍。"),
        ("为何 decode 前除 latent_scale？","恢复 autoencoder 训练时预期的 latent 数值尺度。"),
        ("调大 f 与 channels，观察标量比例。","f 二次降低空间量，channels 线性增加容量。"),
    )
    return


@app.cell
def _(CHAPTERS,chapter_footer):
    chapter_footer(CHAPTERS[23],["Latent Diffusion 在压缩表示上去噪。","计算节省主要来自空间位置减少。","质量同时受 autoencoder 与 diffusion 影响。","下一章比较 UNet 与 Transformer 去噪器。"])
    return


if __name__=="__main__":app.run()
