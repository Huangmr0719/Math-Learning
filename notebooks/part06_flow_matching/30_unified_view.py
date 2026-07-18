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
    chapter_header(CHAPTERS[30],duration="100–140 分钟")
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 本章为什么存在

    学完许多模型后，最危险的状态是只记住名称。本章用同一组问题比较：

    1. 简单随机源是什么？
    2. 数据分布怎样与随机源连接？
    3. 训练监督信号是什么？
    4. 生成时需要几次网络调用？
    5. likelihood、重构与可控性如何获得？

    ## 你已经知道什么

    - VAE：一次 latent sampling + decoder。
    - Diffusion：逐步逆转已知 noising path。
    - Flow Matching：学习 ODE velocity field。
    - **回忆：** 三者共同目标是什么？
    """)
    return


@app.cell
def _(intuition_and_rigor):
    intuition_and_rigor(
        "VAE 学一张“压缩地图”；Diffusion 学很多次“擦掉噪点”的动作；Flow Matching 学整片空间里的“运输箭头”。它们都把简单随机数变成数据，但中间桥梁不同。",
        r"""
        三者都定义参数化生成分布 \(p_\theta(x)\)，但 latent-variable marginalization、
        reverse Markov/SDE dynamics 与 deterministic ODE pushforward 是不同概率结构。
        “统一视角”用于比较，不应抹去目标函数、likelihood 处理和采样随机性的真实差异。
        """,
    )
    return


@app.cell
def _(derivation_map,mo):
    mo.vstack(
        [
            mo.md("## 统一坐标系"),
            derivation_map(["选择简单 base noise","定义连接数据的中间结构","构造可训练局部目标","学习 decoder/score/velocity","数值或一次映射生成","用失败模式诊断"]),
            mo.md(r"""
            | 维度 | VAE | DDPM/DDIM | Flow Matching |
            |---|---|---|---|
            | 中间对象 | latent z | noisy state xt | probability path xt |
            | 学习对象 | encoder/decoder | noise/score | velocity |
            | 典型训练 | ELBO | noise MSE | velocity MSE |
            | 典型采样 | 一次 decoder | 多步 reverse | ODE solve |
            | likelihood | ELBO 是 log-likelihood 下界 | DDPM 有 VLB；连续 score 模型可经 probability-flow ODE 估计 | 仅当作为 CNF 积分 divergence 时可估 |
            | 常见失败 | collapse/blur | 慢、guidance artifacts | coupling/solver error |

            同一模型家族内部也有例外，因此表格是导航，不是定义。

            ### “可以估计 likelihood”到底意味着什么？

            - **VAE：** 通常直接计算的是 ELBO，而不是精确的 \(\log p_\theta(x)\)。
            - **DDPM：** 离散概率模型具有变分下界；这与“DDIM 使用确定性采样轨迹”
              不是同一个 likelihood 结论。
            - **Score SDE：** 若使用 probability flow ODE，并沿轨迹积分 divergence，
              可以进行连续变量的 likelihood 计算。
            - **Flow Matching：** velocity MSE 本身不会自动输出 density。只有当学习到的
              向量场满足 CNF 所需正则条件，并额外积分
              \(d\log p_t(x_t)/dt=-\nabla\cdot v_t(x_t)\) 时，才得到 likelihood。

            因此，“训练目标能计算”“能生成样本”“能评估归一化密度”是三个不同问题。
            """),
        ],
        gap=0.8,
    )
    return


@app.cell
def _(mo):
    priority=mo.ui.dropdown(options={"需要重构/表示":"representation","最高样本保真":"fidelity","少步连续运输":"transport"},value="最高样本保真",label="任务优先级")
    compute=mo.ui.slider(1,10,value=5,step=1,show_value=True,label="可用采样计算")
    return compute,priority


@app.cell
def _(COLORS,compute,mo,np,plt,priority):
    _names=["VAE","Diffusion","Flow Matching"]
    _scores={
        "representation":np.array([9,4,4]),
        "fidelity":np.array([5,9,8]),
        "transport":np.array([4,6,9]),
    }[priority.value].astype(float)
    _cost=np.array([1,8,4],float)
    _feasible=_scores-np.maximum(0,_cost-compute.value)
    _best=int(np.argmax(_feasible))
    _fig,_axes=plt.subplots(1,2,figsize=(9,3.8))
    _axes[0].bar(_names,_scores,color=[COLORS["data"],COLORS["model"],COLORS["prior"]]);_axes[0].set(title="手工设定的 toy 任务适配分",ylabel="toy score（越高越匹配当前优先级）")
    _axes[1].scatter(_cost,_scores,s=120,color=[COLORS["data"],COLORS["model"],COLORS["prior"]])
    for _i,_name in enumerate(_names):_axes[1].text(_cost[_i]+.1,_scores[_i],_name)
    _axes[1].axvline(compute.value,linestyle="--",color=COLORS["danger"],label="当前计算预算");_axes[1].set(xlabel="手工设定的 toy 采样成本",ylabel="toy task score",title="适配度—预算坐标");_axes[1].legend(fontsize=8)
    _fig.tight_layout()
    mo.vstack([mo.md(fr"""## 模型选择不是排行榜

    当前 toy 规则推荐 **{_names[_best]}**。这只是迫使我们明确任务与预算；
    真实选择还需要数据规模、条件类型、likelihood、编辑能力和现有生态证据。

    **图例与证据边界：**蓝/绿/紫分别表示 VAE、Diffusion、Flow Matching；右图红虚线是
    当前 toy 预算。所有分数与成本都是课程为了练习决策而手工设定的量，不是论文 benchmark，
    不能用来声称某个模型家族普遍优于另一个。学习任务是先改变需求，再解释推荐为何变化。"""),mo.hstack([priority,compute],widths="equal"),_fig])
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 从研究问题反推模型

    ```python
    def choose_family(requirements):
        # 先写需求，不先写模型名字。
        need_representation = requirements["encode_and_reconstruct"]
        need_likelihood = requirements["density_or_bound"]
        sampling_budget = requirements["network_calls"]
        conditioning = requirements["control_signal"]
        # 随后用小规模基线验证，而不是仅凭流行度选择。
    ```

    ## 错误与反例

    - “新模型统一旧模型”就认为旧模型无价值：结构目标不同。
    - 只比较 FID，不比较采样成本、重构、覆盖率与条件一致性。
    - 把 toy 成功外推为研究结论：必须设计任务相关验证。
    - 把 velocity MSE 当作 likelihood：它没有包含 density 或 divergence 积分。

    ## 原始资料回看

    - [Auto-Encoding Variational Bayes](https://arxiv.org/abs/1312.6114)
    - [DDPM](https://arxiv.org/abs/2006.11239)
    - [DDIM](https://arxiv.org/abs/2010.02502)
    - [Flow Matching](https://arxiv.org/abs/2210.02747)
    """)
    return


@app.cell
def _(exercise_block,mo):
    mo.vstack(
        [
            mo.md("## 分层综合练习"),
            exercise_block(
                ("三类模型共同做什么？","把简单随机源通过可学习概率结构转换为目标数据分布。"),
                ("一次 VAE decoder 与 50 步 sampler，网络调用量大致差多少？","若各步一次网络调用，约 1 次对 50 次；实际模型大小仍不同。"),
                ("给 VAE、Diffusion、Flow 三类模型各写一个最关键的 `assert` shape 断言。","可写 `assert mu.shape == logvar.shape`；`assert x_t.shape == epsilon.shape == pred.shape`；`assert state.shape == velocity.shape`。前两者分别保护逐 latent 维的 Gaussian 参数和逐元素噪声损失，最后一个保护 ODE 状态更新。"),
                ("为自己的研究写需求表再选择模型。","至少比较表示需求、条件控制、采样预算、likelihood、数据规模和失败诊断，不设唯一答案。"),
            ),
        ],
        gap=0.8,
    )
    return


@app.cell
def _(CHAPTERS,chapter_footer):
    chapter_footer(CHAPTERS[30],["VAE、Diffusion、Flow Matching 都连接简单分布与数据分布。","它们的中间结构、训练目标和采样机制不同。","模型选择应从任务约束反推。","课程终点不是记住名称，而是能推导、实现、验证和诊断。"])
    return


if __name__=="__main__":app.run()
