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
    from src.teaching import CHAPTERS, PARTS, course_styles

    return CHAPTERS, PARTS, course_styles, mo


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
          <h1>从 VAE 到 Flow Matching</h1>
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
def _(CHAPTERS, PARTS, mo):
    _sections = []
    for _part_key, _part_title in PARTS.items():
        _rows = [
            {
                "章节": f"{_spec.number:02d}",
                "标题": _spec.title,
                "状态": _spec.status,
                "文件": f"{_spec.part}/{_spec.filename}",
            }
            for _spec in CHAPTERS.values()
            if _spec.part == _part_key
        ]
        _table = "\n".join(
            f"| {row['章节']} | {row['标题']} | {row['状态']} | `{row['文件']}` |"
            for row in _rows
        )
        _sections.append(
            mo.md(
                f"""
                ## {_part_title}

                | 章节 | 标题 | 状态 | 文件 |
                |---:|---|---|---|
                {_table}
                """
            )
        )
    mo.vstack(_sections, gap=1.2)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 统一符号索引

    | 符号 | 含义 | 首次完整教学 |
    |---|---|---|
    | \(x\) | 观测数据 | 第 1 章 |
    | \(z\) | latent variable | 第 1–2 章 |
    | \(p(z)\) | prior | 第 2–3 章 |
    | \(p_\theta(x\mid z)\) | decoder / likelihood | 第 2 章 |
    | \(q_\phi(z\mid x)\) | approximate posterior / encoder | 第 3 章 |
    | \(D_{KL}(p\|q)\) | KL divergence，方向不可交换 | 第 4 章 |
    | \(\mathcal L_{\mathrm{ELBO}}\) | evidence lower bound | 第 5 章 |
    | \(\mu,\log\sigma^2\) | Gaussian posterior 参数 | 第 6 章 |
    | \(x_t\) | 扩散时间 \(t\) 的状态 | 第 14 章 |
    | \(\epsilon_\theta\) | 噪声预测网络 | 第 17 章 |
    | \(s_\theta\) | score field | 第 17 章 |
    | \(v_t(x)\) | 时间相关 velocity field | 第 27 章 |

    ### 跨部分符号切换

    | 符号 | DDPM / DDIM | Flow Matching / CNF |
    |---|---|---|
    | \(x_0\) | 干净数据 | source/base noise |
    | \(x_1\) | 第一步 noisy state | target/data |
    | \(t\) | 通常 0 为数据端、T 为噪声端 | 本课程通常 0 为 source、1 为 target |
    | \(\alpha_t\) | 第 15–20 章通常指单步 \(1-\beta_t\) | 第 21 章特地改用 \(\sqrt{\bar\alpha_t}\)，并在章内警告 |

    同一个符号在不同论文传统中可能承担不同角色。每次跨部分时先看章节中的
    “符号切换提醒”，不要只凭字母猜含义。

    ## 学习纪律

    - 不把“数值实验相符”称为数学证明。
    - 不在公式中使用未定义符号。
    - 不跳过 shape、定义域、KL 方向和时间方向检查。
    - 不把 conditional distribution、marginal distribution 和单条样本路径混为一谈。
    - 不把训练目标、采样过程和 likelihood 计算混为一谈。
    - 不自动执行下载、完整训练或预训练模型加载。
    - 每章结束前先完成回忆问题，再进入下一章。
    """)
    return


if __name__ == "__main__":
    app.run()
