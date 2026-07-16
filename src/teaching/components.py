"""可复用的 marimo 教学组件。

组件刻意保持轻量：只负责一致的教学结构和视觉层级，不隐藏章节中的
数学推导与核心代码。重要推导仍应直接写在各章节 notebook 中。
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence

import marimo as mo

from .catalog import PARTS, ChapterSpec


def course_styles() -> mo.Html:
    return mo.Html(
        """
        <style>
          :root {
            --gm-ink: #1f2933;
            --gm-muted: #52606d;
            --gm-paper: #fffdf8;
            --gm-blue: #2563eb;
            --gm-green: #15803d;
            --gm-amber: #b45309;
            --gm-red: #b91c1c;
          }
          .gm-hero {
            padding: 1.4rem 1.6rem;
            border: 1px solid #d9e2ec;
            border-radius: 18px;
            background: linear-gradient(135deg, #fffdf8 0%, #eff6ff 100%);
            margin-bottom: 1rem;
          }
          .gm-kicker { color: var(--gm-blue); font-weight: 700; letter-spacing: .04em; }
          .gm-question { font-size: 1.25rem; line-height: 1.65; color: var(--gm-ink); }
          .gm-muted { color: var(--gm-muted); }
          .gm-flow {
            padding: .9rem 1rem;
            border-left: 4px solid var(--gm-blue);
            background: #f8fafc;
            line-height: 1.8;
          }
          .gm-code-map td, .gm-code-map th { padding: .45rem .7rem; }
          .gm-status {
            display: inline-block;
            padding: .16rem .55rem;
            border-radius: 999px;
            background: #e0f2fe;
            color: #075985;
            font-size: .82rem;
            font-weight: 700;
          }
        </style>
        """
    )


def chapter_header(spec: ChapterSpec, *, duration: str = "45–90 分钟") -> mo.Html:
    status_note = "内容已精写" if spec.status == "精写" else "结构骨架，待逐部分精写"
    return mo.vstack(
        [
            # 把样式与标题绑定，避免调用者在同一 cell 中先执行 course_styles()
            # 却只显示最后一个表达式，导致导出的 HTML 丢失课程样式。
            course_styles(),
            mo.Html(
                f"""
                <section class="gm-hero">
                  <div class="gm-kicker">{PARTS[spec.part]} · 第 {spec.number} 章</div>
                  <h1>{spec.title}</h1>
                  <p class="gm-question"><strong>本章核心问题：</strong>{spec.question}</p>
                  <p class="gm-muted">预计学习时间：{duration} ·
                    <span class="gm-status">{status_note}</span>
                  </p>
                </section>
                """
            ),
        ],
        gap=0,
    )


def intuition_and_rigor(intuition: str, rigor: str) -> mo.Html:
    return mo.tabs(
        {
            "先建立直觉": mo.callout(mo.md(intuition), kind="info"),
            "再看严格表述": mo.callout(mo.md(rigor), kind="neutral"),
        }
    )


def derivation_map(steps: Sequence[str]) -> mo.Html:
    chain = " <strong>→</strong> ".join(steps)
    return mo.Html(f'<div class="gm-flow">{chain}</div>')


def knowledge_checklist(items: Iterable[str]) -> mo.Html:
    text = "\n".join(f"- [ ] {item}" for item in items)
    return mo.md(text)


def exercise_block(
    understanding: tuple[str, str],
    calculation: tuple[str, str],
    coding: tuple[str, str],
    exploration: tuple[str, str],
) -> mo.Html:
    def item(label: str, pair: tuple[str, str]) -> mo.Html:
        question, answer = pair
        return mo.vstack(
            [
                mo.md(f"### {label}\n\n{question}"),
                mo.accordion({"展开参考答案": mo.md(answer)}, lazy=True),
            ],
            gap=0.5,
        )

    return mo.vstack(
        [
            item("理解题", understanding),
            item("手算题", calculation),
            item("代码题", coding),
            item("探索题", exploration),
        ],
        gap=1.0,
    )


def chapter_footer(spec: ChapterSpec, takeaways: Sequence[str]) -> mo.Html:
    summary = "\n".join(f"- {item}" for item in takeaways)
    return mo.vstack(
        [
            mo.md(f"## 本章总结\n\n{summary}"),
            mo.callout(
                mo.md(
                    f"""
                    **下一章为什么自然出现？**

                    {spec.bridge}
                    """
                ),
                kind="success",
            ),
            mo.md(
                """
                > 如果某一步仍不清楚，请把具体公式、图形或代码行告诉老师。
                > 不要带着一个模糊的“好像懂了”继续前进。
                """
            ),
        ],
        gap=0.8,
    )


def chapter_scaffold(spec: ChapterSpec, confidence: mo.ui.slider) -> mo.Html:
    """渲染尚未精写章节的完整教学骨架。

    骨架明确标出未来必须补齐的内容，避免空文件，也避免把占位内容误认为
    已完成教材。
    """

    prior = "\n".join(f"- {item}" for item in spec.prior)
    math_items = "\n".join(f"- **{item}**：从具体数字和图形开始解释，再给定义。" for item in spec.new_math)
    return mo.vstack(
        [
            course_styles(),
            chapter_header(spec),
            mo.callout(
                mo.md(
                    """
                    本章已经接入课程导航和统一教学协议，但正文尚未进入逐部分精写阶段。
                    当前文件可运行、可检查，并清楚规定本章需要完成的教学任务。
                    """
                ),
                kind="warn",
            ),
            mo.md(
                f"""
                ## 1. 本章为什么存在

                {spec.question}

                精写时必须从一个可观察的失败案例或生活问题开始，而不是直接抛出公式。

                ## 2. 你已经知道什么

                {prior}

                **回忆问题：** 请尝试用一句话说明上述知识如何帮助回答本章问题。

                ## 3. 本章即时数学

                {math_items}

                ## 4. 双层解释

                精写时分别提供“高中生可复述的直觉版本”和“声明假设、定义域、维度的严格版本”。

                ## 5. 推导地图
                """
            ),
            derivation_map(["明确已知量", "定义目标量", "列出中间恒等式", "逐步推导", "映射到代码"]),
            mo.md(
                """
                ## 6. 维度与定义域检查

                - [ ] 所有符号均在使用前定义
                - [ ] 标量、向量、矩阵、batch 维均明确
                - [ ] 概率、方差、对数或时间边界满足定义域
                - [ ] 代码的广播与公式一致

                ## 7. 可视化与交互

                当前自评控件用于确认 marimo 的反应式链路已经接通：
                """
            ),
            confidence,
            mo.md(
                f"""
                当前自评：**{confidence.value}/4**。

                精写时将替换为本章专属的核心交互实验，并包含重置、固定随机种子和必要的单步控制。

                ## 8. 代码与公式逐行对应

                精写时保留教学版核心实现；复杂训练器或求解器放入 `src/`。

                ## 9. 数值验证

                精写时至少加入一种解析—数值、自动微分—有限差分或采样—理论值对比。

                ## 10. 错误与反例

                精写时展示一个本章最常见的错误，并解释错误结果、根因和诊断方法。

                ## 11. 分层练习

                精写时补齐理解题、手算题、代码题、探索题，并在章内折叠答案。

                ## 12. 本章总结与桥梁

                {spec.bridge}
                """
            ),
        ],
        gap=0.8,
    )
