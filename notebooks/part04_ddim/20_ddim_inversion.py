import marimo

__generated_with = "0.23.13"
app = marimo.App(width="medium")


@app.cell
def _():
    import sys as _sys
    from pathlib import Path as _Path
    _root = _Path(__file__).resolve().parents[2]
    if str(_root) not in _sys.path: _sys.path.insert(0, str(_root))
    import marimo as mo
    import numpy as np
    from src.teaching import CHAPTERS, chapter_footer, chapter_header, derivation_map, exercise_block, intuition_and_rigor
    from src.visualization import COLORS, configure_matplotlib
    configure_matplotlib()
    import matplotlib.pyplot as plt
    return CHAPTERS, COLORS, chapter_footer, chapter_header, derivation_map, exercise_block, intuition_and_rigor, mo, np, plt


@app.cell
def _(CHAPTERS, chapter_header):
    chapter_header(CHAPTERS[20], duration="80–110 分钟")
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 1. 本章为什么存在

    文本生成从随机噪声开始，但图像编辑要先把一张真实图映射到模型的噪声轨迹。
    DDIM inversion 近似反向执行确定性采样：
    \[
    x_0\rightarrow x_1\rightarrow\cdots\rightarrow x_T.
    \]

    ## 2. 你已经知道什么

    - 第 19 章：eta=0 给出确定性 DDIM 更新。
    - 数值逆过程会积累局部误差。
    - **回忆：** 为什么有确定性 forward map 仍不保证数值上精确可逆？
    """)
    return


@app.cell
def _(intuition_and_rigor):
    intuition_and_rigor(
        "像沿着山路倒车：知道每个弯道的大致方向不代表能完全重走轮胎印，因为方向预测在前进和倒退时取自略不同位置。",
        r"""
        DDIM inversion 常假设相邻步的网络噪声预测近似不变，用当前状态预测的
        \(\epsilon_\theta(x_t,t)\) 构造下一 noisy state。这是数值近似，不是严格解析逆。
        重构误差来自有限步长、模型误差、条件 guidance 和离散化。
        """,
    )
    return


@app.cell
def _(derivation_map, mo):
    mo.md("## 3–6. Invert → edit → denoise")
    derivation_map(["真实 x0", "按确定性公式向高噪声推进", "保存 latent trajectory", "修改条件", "从某时刻倒序去噪", "比较重构与编辑"])
    mo.md(r"""
    一个简化确定性路径可写成
    \[
    x_t=\sqrt{\bar\alpha_t}\hat x_0+
    \sqrt{1-\bar\alpha_t}\epsilon_\theta.
    \]
    从当前时刻 \(t\) 走向更高噪声时刻 \(s>t\)，教学版 inversion 使用
    \[
    \hat x_0^{(t)}
    =\frac{x_t-\sqrt{1-\bar\alpha_t}\epsilon_\theta(x_t,t)}
           {\sqrt{\bar\alpha_t}},
    \]
    \[
    x_s=\sqrt{\bar\alpha_s}\hat x_0^{(t)}
        +\sqrt{1-\bar\alpha_s}\epsilon_\theta(x_t,t).
    \]
    它假设从 \(t\) 到 \(s\) 的区间内，同一个噪声预测足以描述路径。

    时间方向检查尤其重要：inversion 使用 alpha_bar 逐渐减小的索引；
    generation 使用同一索引列表反转。误差应比较相同数据范围和同一 decoder 输出空间。
    """)
    return


@app.cell
def _(mo):
    inversion_steps = mo.ui.slider(5,100,value=20,step=5,show_value=True,label="inversion steps")
    model_bias = mo.ui.slider(0.0,0.2,value=0.05,step=0.01,show_value=True,label="noise predictor bias")
    return inversion_steps, model_bias


@app.cell
def _(COLORS, inversion_steps, mo, model_bias, np, plt):
    _clean=2.0
    _true_epsilon=-0.8
    _times=np.linspace(0.0,1.0,inversion_steps.value+1)
    _alpha_bar=np.exp(-4.0*_times)
    _alpha=np.sqrt(_alpha_bar)
    _sigma=np.sqrt(1.0-_alpha_bar)
    _ideal=_alpha*_clean+_sigma*_true_epsilon

    def _predict_epsilon(_state):
        # bias=0 时返回同一条理想路径的真实 epsilon。
        # bias>0 时加入依赖当前状态的系统误差，模拟不完美网络。
        return _true_epsilon+model_bias.value*np.tanh(_state)

    _x=_clean
    _inverted=[_x]
    for _index in range(inversion_steps.value):
        _eps_hat=_predict_epsilon(_x)
        _x0_hat=(_x-_sigma[_index]*_eps_hat)/_alpha[_index]
        _x=_alpha[_index+1]*_x0_hat+_sigma[_index+1]*_eps_hat
        _inverted.append(_x)

    _reconstructed=[_x]
    for _index in range(inversion_steps.value,0,-1):
        _eps_hat=_predict_epsilon(_x)
        _x0_hat=(_x-_sigma[_index]*_eps_hat)/_alpha[_index]
        _x=_alpha[_index-1]*_x0_hat+_sigma[_index-1]*_eps_hat
        _reconstructed.append(_x)

    _inverted=np.array(_inverted)
    _reconstructed=np.array(_reconstructed)
    _error=float(abs(_reconstructed[-1]-_clean))
    _reverse_times=_times[::-1]
    _reverse_ideal=_ideal[::-1]
    _fig,_axes=plt.subplots(1,2,figsize=(9,3.8))
    _axes[0].plot(_times,_ideal,"--",label="ideal DDIM path",color=COLORS["success"])
    _axes[0].plot(_times,_inverted,"o-",label="inversion",color=COLORS["data"])
    _axes[0].plot(_reverse_times,_reconstructed,"s-",label="reconstruction",color=COLORS["model"])
    _axes[0].set_xlabel("noise time");_axes[0].set_ylabel("1D state");_axes[0].legend(fontsize=8)
    _axes[1].plot(_reverse_times,np.abs(_reconstructed-_reverse_ideal),color=COLORS["danger"])
    _axes[1].set_xlabel("reverse noise time");_axes[1].set_ylabel("|state - ideal path|")
    _axes[1].set_title("逐步偏离理想 DDIM 路径")
    _fig.tight_layout()
    mo.vstack([mo.md(fr"""## 7、9. 一维 DDIM inversion 数值实验

    这里使用真实 DDIM 重组公式，而不是一般旋转类比。我们人为设定
    \(x_0={_clean}\)、真实噪声 \(\epsilon={_true_epsilon}\)，并令预测器为
    \(\hat\epsilon=\epsilon+b\tanh(x_t)\)。

    当前最终重构误差={_error:.6f}。当 bias=0 时，预测器在整条路径上给出
    同一个真实噪声，inversion 与 reconstruction 应在浮点误差范围内闭环；
    bias>0 时，前进和后退在不同状态上重新评估预测器，误差会积累。
    步数增加不保证消除模型的系统偏差。"""),mo.hstack([inversion_steps,model_bias],widths="equal"),_fig])
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 8. 实现检查

    ```python
    trajectory = [x0]
    for t, t_next in zip(times[:-1], times[1:]):  # low noise -> high noise
        eps = model(x, t, condition)
        # 对应 x0_hat = (xt - sigma_t * eps) / alpha_t。
        x0_hat = predict_x0(x, eps, alpha_bar[t])
        # 用同一个 eps 按下一时刻的 alpha、sigma 重组状态。
        x = compose_state(x0_hat, eps, alpha_bar[t_next])
        trajectory.append(x)
    ```

    删除 trajectory 保存会让编辑难以从中间噪声强度开始，也不易诊断误差出现在哪一步。

    ## 10. 错误与反例

    - 把 inversion 当严格 inverse：网络与离散化使它通常只是近似。
    - inversion 与 reconstruction 使用不同 scheduler：无法闭环比较。
    - CFG 很大仍期待完美重构：guidance 改变向量场，逆过程更不一致。
    - 用任意可逆旋转演示后称其为 DDIM 数值验证：必须实际使用 alpha_bar 与噪声预测公式。

    工程参考：[Hugging Face DDIM Inversion](https://github.com/huggingface/diffusion-models-class/blob/main/unit4/01_ddim_inversion.ipynb)
    """)
    return


@app.cell
def _(exercise_block, mo):
    mo.md("## 11. 分层练习")
    exercise_block(
        ("inversion 的目的是什么？", "把真实样本定位到模型的 noisy trajectory，便于重构或在修改条件后编辑。"),
        ("每步误差 0.01，能否断言 50 步总误差 0.5？", "不能；误差是向量并经过非线性传播，可能相消或放大。"),
        ("为什么保存 trajectory？", "可从不同噪声强度编辑并逐步诊断重构偏差。"),
        ("增加 steps 与 bias，分别观察误差。", "bias=0 时本 toy 路径应几乎精确闭环；bias 表示状态相关模型误差。增加 steps 会减小单步状态变化，但不会自动消除系统偏差。"),
    )
    return


@app.cell
def _(CHAPTERS, chapter_footer):
    chapter_footer(CHAPTERS[20], ["DDIM inversion 近似把真实数据映射到噪声轨迹。", "确定性不等于数值严格可逆。", "scheduler、guidance 与步长影响重构。", "下一章统一比较 epsilon、x0、score 和 v 参数化。"])
    return


if __name__=="__main__": app.run()
