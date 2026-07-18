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
    from src.math_checks import coupling_marginals
    from src.teaching import CHAPTERS,chapter_footer,chapter_header,derivation_map,exercise_block,intuition_and_rigor
    from src.visualization import COLORS,configure_matplotlib
    configure_matplotlib();import matplotlib.pyplot as plt
    return CHAPTERS,COLORS,chapter_footer,chapter_header,coupling_marginals,derivation_map,exercise_block,intuition_and_rigor,mo,np,plt


@app.cell
def _(CHAPTERS,chapter_header):
    chapter_header(CHAPTERS[29],duration="100–140 分钟")
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 本章为什么存在

    Flow Matching 的线性路径仍需要决定哪个 source 样本配哪个 target 样本。
    随机 pairing 可能让轨迹交叉、速度互相抵消。Optimal Transport（OT）
    在所有合法 coupling 中寻找低运输成本方案；Rectified Flow 则可以从任意
    coupling 出发，学习尽量沿配对直线运动的 ODE，并通过 rectification/reflow
    逐步改善 learned coupling。二者有关，但不是同一个算法。

    ## 你已经知道什么

    - 第 28 章：单对样本速度为 x1-x0。
    - 第 2 章：联合概率表的行和、列和给出边缘分布。
    - **回忆：** 同样两组起终点，不同配对为何总路程不同？
    """)
    return


@app.cell
def _(intuition_and_rigor):
    intuition_and_rigor(
        "两排学生交换座位。随便配对会交叉奔跑；把每个人分到附近目标，路线更短、更少相撞。Coupling 就是配对规则。",
        r"""
        对 source \(p_0\)、target \(p_1\)，coupling
        \(\pi(x_0,x_1)\) 是一个联合分布，必须满足
        \[
        \int\pi(x_0,x_1)dx_1=p_0(x_0),\qquad
        \int\pi(x_0,x_1)dx_0=p_1(x_1).
        \]
        所有满足这两个条件的 coupling 构成集合 \(\Pi(p_0,p_1)\)。
        二次 OT 在这个集合中最小化
        \[
        \min_{\pi\in\Pi(p_0,p_1)}E_\pi\|X_1-X_0\|^2.
        \]
        OT displacement interpolation 使用 OT coupling 定义概率路径；
        Rectified Flow 的 rectification 则不要求初始 coupling 已经是 OT。
        更低成本、较少交叉常使路径更容易数值积分，但 minibatch 近似与
        高维距离是否符合语义仍会影响结果。
        """,
    )
    return


@app.cell
def _(derivation_map,mo):
    mo.vstack(
        [
            mo.md("## Coupling、transport cost 与 reflow"),
            derivation_map(["固定 source/target marginals","选择联合 pairing","计算平方运输成本","寻找低成本 coupling","训练 velocity","用模型生成新 pairs","再次 rectification"]),
            mo.md(r"""
            Rectified Flow 的 reflow 思想：先用已有模型把 source 样本映射到生成终点，
            将这组更一致的 source–target pairing 作为新训练数据，再训练直线插值速度。
            “直”通常指轨迹曲率/离散误差更低，不等于所有 velocity 在全空间常数。

            ### 数学暂停站：coupling 不等于 marginal

            对离散变量，coupling 是一张联合概率表。表中第 \(i,j\) 格表示
            “source 取第 \(i\) 个值，同时 target 取第 \(j\) 个值”的概率。
            每一行求和得到 source marginal，每一列求和得到 target marginal。

            两张联合表可以具有完全相同的行和与列和，却把概率质量放在不同格子中。
            因此 coupling 改变的是“谁和谁配对”，不是两端各自有哪些样本。

            ### 四个容易混淆的对象

            1. **Independent coupling**：独立抽取 source 与 target，是一种合法 coupling。
            2. **OT coupling**：在合法 coupling 中最小化所选 transport cost。
            3. **OT probability path**：用 OT coupling 做 displacement interpolation。
            4. **Rectified Flow / reflow**：从给定 coupling 学 ODE，再用模型诱导的新
               coupling 重新训练，使轨迹趋于更直。

            Rectified Flow 论文证明的是 rectification 使一类凸 transport costs
            不增加；不能把它简写成“每次 reflow 都精确求出了 OT”。
            """),
        ],
        gap=0.8,
    )
    return


@app.cell
def _(mo):
    coupling_mass=mo.ui.slider(.1,.5,value=.3,step=.05,show_value=True,label="联合表左上角概率 a")
    return (coupling_mass,)


@app.cell
def _(COLORS,coupling_marginals,coupling_mass,mo,np,plt):
    _a=coupling_mass.value
    _table=np.array([[_a,.5-_a],[.6-_a,_a-.1]])
    _source,_target=coupling_marginals(_table)
    _fig,_ax=plt.subplots(figsize=(5.5,3.8))
    _image=_ax.imshow(_table,vmin=0,vmax=.5,cmap="Blues")
    for _i in range(2):
        for _j in range(2):
            # 深蓝格使用白字，浅蓝格使用深字，避免 coupling 改变后数字消失。
            _text_color="#ffffff" if _table[_i,_j]>=.28 else "#111827"
            _ax.text(_j,_i,f"{_table[_i,_j]:.2f}",ha="center",va="center",color=_text_color,weight="bold")
    _ax.set_xticks([0,1],["target A","target B"])
    _ax.set_yticks([0,1],["source A","source B"])
    _ax.set_title("颜色与数字共同表示 joint probability")
    _fig.colorbar(_image,ax=_ax,label="joint probability")
    _fig.tight_layout()
    mo.vstack([mo.md(fr"""### 联合概率表交互

    当前 coupling 为
    \[
    \begin{{bmatrix}}
    {_table[0,0]:.2f}&{_table[0,1]:.2f}\\
    {_table[1,0]:.2f}&{_table[1,1]:.2f}
    \end{{bmatrix}}.
    \]

    行和（source marginal）始终为 `{np.round(_source,2)}`，
    列和（target marginal）始终为 `{np.round(_target,2)}`。
    拖动 \(a\) 会改变联合配对，却不改变两端边缘分布。

    **图例与任务：**格内数字与蓝色深浅都表示该 source–target 配对的联合概率；
    行是 source，列是 target，右侧色条给出颜色刻度。拖动时分别检查每行和、每列和。
    这张 \(2\times2\) 表只演示 coupling 的定义，不证明某个 coupling 具有最小运输成本。"""),coupling_mass,_fig])
    return


@app.cell
def _(mo):
    coupling=mo.ui.dropdown(options={"随机 coupling":"random","排序/近似 OT":"sorted"},value="随机 coupling",label="pairing")
    pair_seed=mo.ui.number(value=7,start=0,stop=9999,step=1,label="seed")
    return coupling,pair_seed


@app.cell
def _(COLORS,coupling,mo,np,pair_seed,plt):
    _rng=np.random.default_rng(int(pair_seed.value));_n=14
    _x0=np.sort(_rng.normal(-1.5,.8,_n));_x1=np.sort(_rng.normal(1.5,.8,_n))
    if coupling.value=="random":_paired=_x1[_rng.permutation(_n)]
    else:_paired=_x1
    _cost=float(np.mean((_paired-_x0)**2))
    _fig,_axes=plt.subplots(1,2,figsize=(9,3.8))
    for _i in range(_n):_axes[0].plot([0,1],[_x0[_i],_paired[_i]],color=COLORS["data"],alpha=.7,label="配对连线" if _i==0 else None)
    _axes[0].scatter(np.zeros(_n),_x0,color=COLORS["prior"],label="source 样本");_axes[0].scatter(np.ones(_n),_paired,color=COLORS["model"],label="target 样本");_axes[0].set_xticks([0,1],["source","target"]);_axes[0].set_ylabel("一维位置");_axes[0].set_title("coupling 决定谁与谁相连");_axes[0].legend(fontsize=7)
    _t=np.linspace(0,1,100);_paths=(1-_t[:,None])*_x0[None,:]+_t[:,None]*_paired[None,:]
    _axes[1].plot(_t,_paths,alpha=.5);_axes[1].set(xlabel="时间 t",ylabel=r"插值位置 $x_t$",title="每个配对的线性插值轨迹")
    _fig.tight_layout()
    mo.vstack([mo.md(fr"""## Coupling 改变路径

    当前平均平方 transport cost={_cost:.4f}。一维排序 pairing 是二次成本 OT 的解；
    高维情形不能简单逐坐标排序。

    **图例与任务：**左图紫点/绿点是两端经验分布，蓝线是当前配对；右图每条线是
    该配对随时间的插值。切换 coupling 时同时观察交叉数和平方成本。
    本实验只证明一维等质量经验分布在平方成本下的排序匹配结论；不能外推为高维语义 OT。"""),mo.hstack([coupling,pair_seed],widths="equal"),_fig])
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Minibatch pairing 代码

    ```python
    # cost[i,j] 是 source i 到 target j 的平方距离。
    cost = torch.cdist(x0, x1).pow(2)  # [B,B]
    assignment = solve_assignment(cost)
    paired_x1 = x1[assignment]
    target_velocity = paired_x1 - x0
    ```

    ## 错误与反例

    - 把独立采样称为唯一 coupling：它只是众多联合分布之一。
    - 把 Rectified Flow 等同于“先精确求 OT”：rectification 可从任意 coupling 出发。
    - 一维排序法直接推广到高维逐坐标排序。
    - transport cost 小就保证感知语义正确：所选空间和距离决定“近”的含义。

    参考：[Liu et al., Flow Straight and Fast](https://arxiv.org/abs/2209.03003)
    """)
    return


@app.cell
def _(exercise_block,mo):
    mo.vstack(
        [
            mo.md("## 分层练习"),
            exercise_block(
                ("coupling 改变边缘分布吗？","合法 coupling 必须保持指定的 source 和 target 边缘；它改变的是联合配对。若行和或列和改变，那张表就不再属于同一个 Π(p0,p1)。"),
                ("source=[0,2]、target=[1,3]，排序成本均值？","((1-0)^2+(3-2)^2)/2=1。"),
                ("已知 `source,target: [B,D]`，写出两两平方欧氏距离 cost matrix，并说明 shape。","写 `cost = ((source[:, None, :] - target[None, :, :]) ** 2).sum(dim=-1)`，shape 为 `[B,B]`：第 `(i,j)` 项是第 i 个 source 与第 j 个 target 的候选成本。"),
                ("切换 coupling，观察交叉与成本。","在一维、等质量经验分布、平方距离成本且只考虑一一 permutation matching 时，排序配对是最优的，不只是经验上“通常更好”。更高维或不同约束下不能直接套用这个结论。"),
            ),
        ],
        gap=0.8,
    )
    return


@app.cell
def _(CHAPTERS,chapter_footer):
    chapter_footer(CHAPTERS[29],["Coupling 决定 source 与 target 的联合配对。","OT 最小化期望运输成本。","更直路径通常更易用少步求解。","最后一章统一比较三类生成建模范式。"])
    return


if __name__=="__main__":app.run()
