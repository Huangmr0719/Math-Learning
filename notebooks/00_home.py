import marimo

__generated_with = "0.23.13"
app = marimo.App(width="medium")


@app.cell
def _():
    import sys as _sys
    from pathlib import Path as _Path

    _root = _Path(__file__).resolve().parents[1]
    if str(_root) not in _sys.path:
        _sys.path.insert(0, str(_root))

    import marimo as mo
    from src.teaching import (
        CHAPTERS,
        PARTS,
        course_map_table,
        course_styles,
        terminology_table,
    )

    return (
        CHAPTERS,
        PARTS,
        course_map_table,
        course_styles,
        mo,
        terminology_table,
    )


@app.cell
def _(course_styles):
    course_styles()
    return


@app.cell
def _(mo):
    mo.Html(
        """
        <section class="gm-hero">
          <div class="gm-kicker">GENERATIVE MODEL LEARNING</div>
          <h1>从变分自编码器（VAE）到流匹配（Flow Matching）</h1>
          <p class="gm-question">
            一条为高中数学基础学习者设计的交互式路径：
            每次只学习眼前模型真正需要的数学，并立即用推导、代码和图形验证。
          </p>
        </section>
        """
    )
    return


@app.cell
def _(mo):
    mo.callout(
        mo.md(
            """
            ### 如何开始

            在工作区根目录运行：

            ```bash
            marimo edit notebooks
            ```

            第一次学习请从第 1 章开始。章节文件是教材的唯一教学源；
            `exports/` 中的 HTML 只是便于阅读的发布版本。
            """
        ),
        kind="info",
    )
    return


@app.cell
def _(PARTS, mo):
    part_filter = mo.ui.dropdown(
        options={
            "全部六个部分": "all",
            **{title: key for key, title in PARTS.items()},
        },
        value="全部六个部分",
        label="只查看某一部分",
    )
    completed_through = mo.ui.slider(
        0,
        30,
        value=0,
        step=1,
        show_value=True,
        label="我已经学完的最后一章",
    )
    return completed_through, part_filter


@app.cell
def _(CHAPTERS, PARTS, completed_through, course_map_table, mo, part_filter):
    _sections = []
    for _part_key, _part_title in PARTS.items():
        if part_filter.value != "all" and _part_key != part_filter.value:
            continue
        _rows = [
            {
                "章节": f"{_spec.number:02d}",
                "标题": _spec.title,
                "学习进度": (
                    "已完成"
                    if _spec.number <= completed_through.value
                    else "下一章"
                    if _spec.number == completed_through.value + 1
                    else "未开始"
                ),
                "入口": f"./{_spec.part}/{_spec.filename}",
            }
            for _spec in CHAPTERS.values()
            if _spec.part == _part_key
        ]
        _sections.append(course_map_table(_part_title, _rows))
    if completed_through.value < 30:
        _next_spec = CHAPTERS[completed_through.value + 1]
        _next_message = f"建议下一步：**第 {_next_spec.number} 章 · {_next_spec.title}**。"
    else:
        _next_message = "你已经走完整条主线；建议返回第 30 章完成综合研究设计。"
    _map_md = (
        f"## 交互式课程地图\n\n"
        f"当前记录：已完成 **{completed_through.value}/30** 章。\n"
        f"{_next_message}\n\n"
        "> 这里的进度控件用于规划当前学习会话，不会自动写入文件。"
    )
    mo.vstack(
        [
            mo.md(_map_md),
            mo.hstack([part_filter, completed_through], widths="equal"),
            *_sections,
        ],
        gap=1.2,
    )
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 统一符号索引

    | 符号 | 含义 | 首次完整教学 |
    |---|---|---|
    | \(x\) | 观测数据 | 第 1 章 |
    | \(z\) | 潜变量（latent variable） | 第 1–2 章 |
    | \(p(z)\) | 先验分布（prior） | 第 2–3 章 |
    | \(p_\theta(x\mid z)\) | 解码器似然（decoder likelihood） | 第 2 章 |
    | \(q_\phi(z\mid x)\) | 近似后验（approximate posterior）/ 编码器 | 第 3 章 |
    | \(D_{KL}(p\|q)\) | KL 散度，方向不可交换 | 第 4 章 |
    | \(\mathcal L_{\mathrm{ELBO}}\) | 证据下界（evidence lower bound） | 第 5 章 |
    | \(\mu,\log\sigma^2\) | 高斯后验参数 | 第 6 章 |
    | \(x_t\) | 扩散时间 \(t\) 的状态 | 第 14 章 |
    | \(\epsilon_\theta\) | 噪声预测网络 | 第 17 章 |
    | \(s_\theta\) | 得分场（score field） | 第 17 章 |
    | \(v_t(x)\) | 时间相关速度场（velocity field） | 第 27 章 |

    ### 跨部分符号切换

    | 符号 | DDPM / DDIM | 流匹配 / CNF |
    |---|---|---|
    | \(x_0\) | 干净数据 | 源分布 / 基础噪声（source/base noise） |
    | \(x_1\) | 第一步含噪状态 | 目标分布 / 数据（target/data） |
    | \(t\) | 通常 0 为数据端、T 为噪声端 | 本课程通常 0 为源端、1 为目标端 |
    | \(\alpha_t\) | 第 15–20 章通常指单步 \(1-\beta_t\) | 第 21 章特地改用 \(\sqrt{\bar\alpha_t}\)，并在章内警告 |

    同一个符号在不同论文传统中可能承担不同角色。每次跨部分时先看章节中的
    “符号切换提醒”，不要只凭字母猜含义。

    ## 学习纪律

    - 不把“数值实验相符”称为数学证明。
    - 不在公式中使用未定义符号。
    - 不跳过张量形状（shape）、定义域、KL 方向和时间方向检查。
    - 不把条件分布、边缘分布和单条样本路径混为一谈。
    - 不把训练目标、采样过程和似然计算混为一谈。
    - 不自动执行下载、完整训练或预训练模型加载。
    - 每章结束前先完成回忆问题，再进入下一章。
    """)
    return


@app.cell
def _(terminology_table):
    terminology_table()
    return


if __name__ == "__main__":
    app.run()
