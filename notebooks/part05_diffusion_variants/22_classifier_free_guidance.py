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
    chapter_header(CHAPTERS[22],duration="80–110 分钟")
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 1. 本章为什么存在

    条件模型知道提示词，但生成可能不够听话。Classifier-Free Guidance 同一网络做两次预测：
    无条件与有条件，然后沿两者差异外推：
    \[
    \epsilon_{\rm cfg}=\epsilon_u+w(\epsilon_c-\epsilon_u).
    \]

    本章采用常见代码库的 scale 约定：\(w=0\) 为 unconditional，
    \(w=1\) 为普通 conditional，\(w>1\) 才是额外 guidance。

    ## 2. 你已经知道什么

    - 第 11 章：条件变量控制生成。
    - 第 17 章：预测噪声对应一个去噪方向。
    - **回忆：** w=0、1 分别得到什么？
    """)
    return


@app.cell
def _(intuition_and_rigor):
    intuition_and_rigor(
        "无条件预测说“什么都可以”，条件预测说“更像猫的方向在这里”。CFG 把两者差向量放大，让模型更坚决地朝条件走。",
        r"""
        训练时随机丢弃条件，使一个网络近似学习 conditional 与 unconditional prediction。
        采样时 \(w=1\) 是普通 conditional prediction；\(w>1\) 是线性外推而非概率凸组合。
        更强 guidance 往往提升条件一致性，却可能降低多样性并导致过饱和。

        **记号约定：** 部分论文写成
        \((1+s)\epsilon_c-s\epsilon_u\)，其中 \(s=0\) 已经是 conditional。
        它与本章写法完全等价，只需令 \(w=1+s\)。比较论文或代码时必须先检查
        “scale=0”究竟表示 unconditional 还是 conditional。
        """,
    )
    return


@app.cell
def _(derivation_map,mo):
    mo.vstack(
        [
            mo.md("## 3–6. Conditional score 差与线性外推"),
            derivation_map(["训练时随机置空条件","得到 unconditional prediction","得到 conditional prediction","计算条件差向量","乘 guidance scale","进入 sampler"]),
            mo.md(r"""
            由 Bayes score 恒等式：
            \[
            \nabla_x\log p(x|c)=\nabla_x\log p(x)+\nabla_x\log p(c|x).
            \]
            推导只需从
            \[
            \log p(x|c)=\log p(x)+\log p(c|x)-\log p(c)
            \]
            开始，对 \(x\) 求梯度。因为 \(p(c)\) 与 \(x\) 无关，
            \(\nabla_x\log p(c)=0\)，所以常数项消失。

            conditional 与 unconditional score 的差近似提供“条件分类”方向，因此无需单独分类器。
            公式可用于 epsilon、v 或 score，但必须在同一参数化空间做线性组合。
            """),
        ],
        gap=0.8,
    )
    return


@app.cell
def _(mo):
    scale=mo.ui.slider(0,12,value=5,step=.5,show_value=True,label="CFG scale w")
    dropout=mo.ui.slider(0,0.5,value=.1,step=.05,show_value=True,label="训练 condition dropout")
    return dropout,scale


@app.cell
def _(COLORS,dropout,mo,np,plt,scale):
    _u=np.array([.4,.2]);_c=np.array([1.0,.8]);_g=_u+scale.value*(_c-_u)
    _fig,_axes=plt.subplots(1,2,figsize=(9,3.8))
    _axes[0].quiver([0,0,0],[0,0,0],[_u[0],_c[0],_g[0]],[_u[1],_c[1],_g[1]],angles="xy",scale_units="xy",scale=1,color=[COLORS["muted"],COLORS["data"],COLORS["model"]])
    _axes[0].set_xlim(-1,8);_axes[0].set_ylim(-1,8);_axes[0].set_aspect("equal")
    _w=np.linspace(0,12,100)
    _alignment=1-np.exp(-_w/3);_diversity=np.exp(-_w/10)
    _axes[1].plot(_w,_alignment,label="toy condition alignment",color=COLORS["data"])
    _axes[1].plot(_w,_diversity,label="toy diversity",color=COLORS["prior"])
    _axes[1].axvline(scale.value,linestyle="--",color=COLORS["danger"]);_axes[1].legend(fontsize=8)
    _fig.tight_layout()
    mo.vstack([mo.md(fr"""## 7、9. Guidance 外推

    unconditional=`{_u}`，conditional=`{_c}`，guided=`{_g}`。
    dropout={dropout.value:.2f} 决定训练中无条件样本比例；它不是采样 scale。"""),mo.hstack([scale,dropout],widths="equal"),_fig])
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 8. 代码

    ```python
    # 训练：以 p_uncond 把 condition 替换为空 token。
    condition = torch.where(drop_mask, null_condition, condition)

    # 采样：同一 xt、t 做两次 forward。
    eps_u = model(xt, t, null_condition)
    eps_c = model(xt, t, condition)
    eps = eps_u + guidance_scale * (eps_c - eps_u)
    ```

    ## 10. 错误与反例

    - 把 scale=0 说成 conditional：它给 unconditional。
    - 从论文复制 scale 数值却不核对约定：本章的 w 与某些论文的 s 相差 1。
    - scale 越大越好：可能过饱和、模式变少、产生伪影。
    - 训练没有 condition dropout 却期待可靠 unconditional branch。

    原始资料：[Ho & Salimans, Classifier-Free Diffusion Guidance](https://arxiv.org/abs/2207.12598)
    """)
    return


@app.cell
def _(exercise_block,mo):
    mo.vstack(
        [
            mo.md("## 11. 分层练习"),
            exercise_block(
                ("CFG 放大哪个方向？","conditional prediction 减 unconditional prediction 的差方向。"),
                ("u=2、c=3、w=4，guided？","2+4×(3-2)=6。"),
                ("为什么训练要丢条件？","让同一网络学到可用于 CFG 的 unconditional prediction。"),
                ("增大 scale，观察 toy alignment/diversity。","一致性提高但多样性下降，体现常见权衡而非严格定律。"),
            ),
        ],
        gap=0.8,
    )
    return


@app.cell
def _(CHAPTERS,chapter_footer):
    chapter_footer(CHAPTERS[22],["CFG 用同一网络的有条件/无条件差进行外推。","scale=1 是普通条件预测。","更强 guidance 有一致性—多样性权衡。","下一章把 diffusion 移到压缩 latent space。"])
    return


if __name__=="__main__":app.run()
