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
    chapter_header(CHAPTERS[24],duration="80–110 分钟")
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 1. 本章为什么存在

    Diffusion 的概率公式没有规定 denoiser 必须是 UNet。DiT 把 latent image 切成 patch
    token，用 Transformer 预测噪声或 velocity。

    ## 2. 你已经知道什么

    - 输入输出必须保持相同空间 shape。
    - 时间与条件信息要注入网络。
    - **回忆：** patch 变大时 token 数如何变化？
    """)
    return


@app.cell
def _(intuition_and_rigor):
    intuition_and_rigor(
        "UNet 像先看局部纹理、逐层缩小后理解全局，再把细节补回；DiT 像把图像切成卡片，让每张卡片通过 attention 与所有卡片交流。",
        r"""
        对 \(H\times W\) latent、patch size P，token 数
        \(N=(H/P)(W/P)\)，前提是 \(P\) 同时整除 \(H,W\)。
        Self-attention 的基础交互矩阵为 `[B,heads,N,N]`，
        计算随 \(N^2\) 增长。UNet 具有卷积局部性与多尺度 inductive bias；
        这里的 inductive bias 指架构预先带入的偏好，例如卷积默认邻近像素
        更相关、同一局部模式可在不同位置复用。DiT 的这种局部先验较弱，
        通常更依赖数据和算力，但具有良好的规模扩展表现。
        """,
    )
    return


@app.cell
def _(derivation_map,mo):
    mo.md("## 3–6. Patch embedding 与 self-attention")
    derivation_map(["latent 切 patch","每个 patch 展平线性投影","加入位置/时间/条件","QK^T 得 attention","加权 V","投影回 patch 并重组"])
    mo.md(r"""
    \[
    A=\operatorname{softmax}(QK^\top/\sqrt d),\qquad \mathrm{Attention}=AV.
    \]
    每个 token 通过三个线性映射得到：

    - query \(Q\)：我正在寻找什么信息；
    - key \(K\)：我能用什么特征被别人匹配；
    - value \(V\)：匹配成功后实际传递什么内容。

    若 `Q,K,V` shape 都是 `[B,heads,N,d]`，则 \(QK^\top\) 的 shape
    是 `[B,heads,N,N]`。倒数第二个 N 选择 query，最后一个 N 枚举 keys。
    因此 softmax 必须沿最后的 key 轴，每一行权重和为 1。

    假设 Q、K 各分量均值约为 0、方差约为 1，d 个乘积相加后的点积方差
    约为 d；除以 \(\sqrt d\) 把标准差拉回常数量级，避免 softmax 过早饱和。
    """)
    return


@app.cell
def _(mo):
    resolution=mo.ui.dropdown(options={"32×32":32,"64×64":64,"128×128":128},value="64×64",label="latent resolution")
    return (resolution,)


@app.cell
def _(mo,resolution):
    _valid=[_p for _p in range(1,17) if resolution.value%_p==0]
    patch=mo.ui.dropdown(
        options={str(_p):_p for _p in _valid},
        value="4" if 4 in _valid else str(_valid[0]),
        label="patch size P（只列出合法因数）",
    )
    return (patch,)


@app.cell
def _(COLORS,mo,np,patch,plt,resolution):
    _n=(resolution.value//patch.value)**2
    _p=np.array([_value for _value in range(1,17) if resolution.value%_value==0])
    _tokens=(resolution.value//_p)**2
    _cost=_tokens**2
    _matrix=np.exp(-np.abs(np.arange(min(_n,64))[:,None]-np.arange(min(_n,64))[None,:])/8)
    _matrix/=_matrix.sum(axis=1,keepdims=True)
    _fig,_axes=plt.subplots(1,2,figsize=(9,3.8))
    _axes[0].plot(_p,_cost/_cost.max(),color=COLORS["data"]);_axes[0].axvline(patch.value,linestyle="--",color=COLORS["danger"])
    _axes[0].set_yscale("log");_axes[0].set_title("relative attention N²")
    _axes[1].imshow(_matrix,cmap="viridis",aspect="auto");_axes[1].set_title("toy attention rows sum to 1")
    _fig.tight_layout()
    mo.vstack([mo.md(fr"""## 7、9. Token 数与 attention 成本

    当前 token 数 \(N={_n}\)，attention score 元素约 \(N^2={_n**2:,}\)。
    当前 \(P={patch.value}\) 能整除 \(H=W={resolution.value}\)，因此 patchify
    与 unpatchify 的空间 shape 可以闭环。toy attention 最大行和误差=
    `{np.max(np.abs(_matrix.sum(axis=1)-1)):.2e}`。"""),mo.hstack([resolution,patch],widths="equal"),_fig])
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 8. Shape 代码

    ```python
    # x: [B,C,H,W] -> tokens: [B,N,P*P*C] -> [B,N,D]
    tokens = patchify(x)
    tokens = patch_projection(tokens)
    tokens = transformer(tokens, time_condition)
    prediction = unpatchify(output_projection(tokens))
    assert prediction.shape == x.shape
    ```

    ## 10. 错误与反例

    - patch size 不整除 H/W，却用整数除法静默截断 token 数：重组 shape 会失败。
    - attention softmax 沿 query 轴：每个 query 不再对 keys 归一化。
    - 宣称 DiT 在所有规模都优于 UNet：小数据、小算力下 inductive bias 很重要。

    原始资料：[Peebles & Xie, Scalable Diffusion Models with Transformers](https://arxiv.org/abs/2212.09748)
    """)
    return


@app.cell
def _(exercise_block,mo):
    mo.md("## 11. 分层练习")
    exercise_block(
        ("UNet 与 DiT 的核心差别？","UNet 主要用多尺度卷积，DiT 把图像表示为 token 并用 self-attention 交互。"),
        ("64×64、P=8，N？","8×8=64 tokens。"),
        ("为什么输出要 unpatchify？","扩散 sampler 需要与输入 xt 相同的空间张量 shape。"),
        ("减小 patch，观察 N²。","token 数按 1/P² 增长，attention 矩阵按约 1/P⁴ 增长。"),
    )
    return


@app.cell
def _(CHAPTERS,chapter_footer):
    chapter_footer(CHAPTERS[24],["Denoiser 架构独立于扩散概率主干。","DiT 用 patch token 与 self-attention。","attention 成本对 token 数二次增长。","下一章固定向量场，研究更好的数值求解。"])
    return


if __name__=="__main__":app.run()
