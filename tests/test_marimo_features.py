import ast
from html import unescape
from pathlib import Path

from src.teaching.catalog import CHAPTERS
from src.teaching.components import (
    chapter_navigation,
    chapter_paper_trail,
    chapter_terminology,
    course_map_table,
    course_styles,
    exercise_block,
    paper_guide_navigation,
    terminology_table,
)
from src.teaching.paper_guides import (
    PAPERS,
    PAPER_GUIDES,
    PAPER_GUIDES_AFTER_CHAPTER,
    PAPER_GUIDES_BEFORE_CHAPTER,
)
from src.teaching.terminology import TERMS, terms_for_chapter
from src.visualization import COLORS, configure_matplotlib


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOKS = ROOT / "notebooks"

DISPLAY_HELPERS = {
    "chapter_footer",
    "chapter_header",
    "course_styles",
    "derivation_map",
    "exercise_block",
    "intuition_and_rigor",
}


def test_paper_evidence_cards_keep_explicit_contrast_in_dark_marimo_theme():
    """浅色论文卡不能从深色 marimo 容器继承白色正文。"""

    styles = course_styles()._repr_html_()
    evidence_rule = styles.split(".gm-paper-evidence {", 1)[1].split("}", 1)[0]
    guide_rule = styles.split(".gm-figure-guide section {", 1)[1].split("}", 1)[0]

    assert "background: var(--gm-paper);" in evidence_rule
    assert "color: var(--gm-ink);" in evidence_rule
    assert "background: #f8fafc;" in guide_rule
    assert "color: var(--gm-ink);" in guide_rule
    caption_rule = styles.split(".gm-figure-caption {", 1)[1].split("}", 1)[0]
    assert "background: var(--gm-paper);" in caption_rule
    assert "color: var(--gm-ink);" in caption_rule


def test_all_custom_light_surfaces_define_their_own_foreground_colors():
    """自定义浅色表面不能依赖 marimo 当前主题的继承文字色。"""

    styles = course_styles()._repr_html_()
    surface_expectations = {
        ".gm-hero {": ("background:", "color: var(--gm-ink);"),
        ".gm-flow {": ("background:", "color: var(--gm-ink);"),
        ".gm-paper-trail {": ("background:", "color: var(--gm-ink);"),
        ".gm-paper-card {": ("background:", "color: var(--gm-ink);"),
        ".gm-paper-evidence {": ("background:", "color: var(--gm-ink);"),
        ".gm-paper-quote {": ("background:", "color: var(--gm-ink);"),
        ".gm-figure-caption {": ("background:", "color: var(--gm-ink);"),
        ".gm-figure-guide section {": ("background:", "color: var(--gm-ink);"),
        ".gm-term-intro {": ("background:", "color: var(--gm-ink);"),
        ".gm-chapter-nav-link {": ("background:", "color: var(--gm-ink);"),
    }
    for selector, declarations in surface_expectations.items():
        rule = styles.split(selector, 1)[1].split("}", 1)[0]
        for declaration in declarations:
            assert declaration in rule, f"{selector} 缺少 {declaration}"


def test_every_matplotlib_notebook_uses_theme_safe_shared_configuration():
    for path in sorted(NOTEBOOKS.rglob("*.py")):
        source = path.read_text(encoding="utf-8")
        if "plt." in source or "matplotlib" in source:
            assert "configure_matplotlib()" in source, path

    configure_matplotlib()
    import matplotlib.pyplot as plt

    expected = {
        "figure.facecolor": "#ffffff",
        "axes.facecolor": "#ffffff",
        "axes.labelcolor": "#1f2933",
        "axes.titlecolor": "#1f2933",
        "text.color": "#1f2933",
        "xtick.color": "#475569",
        "ytick.color": "#475569",
        "legend.labelcolor": "#1f2933",
    }
    for key, value in expected.items():
        assert plt.rcParams[key] == value


def test_shared_text_palette_meets_wcag_aa_on_course_surfaces():
    def luminance(hex_color: str) -> float:
        channels = [int(hex_color[index : index + 2], 16) / 255 for index in (1, 3, 5)]
        linear = [
            channel / 12.92
            if channel <= 0.04045
            else ((channel + 0.055) / 1.055) ** 2.4
            for channel in channels
        ]
        return 0.2126 * linear[0] + 0.7152 * linear[1] + 0.0722 * linear[2]

    def contrast(foreground: str, background: str) -> float:
        lighter, darker = sorted(
            (luminance(foreground), luminance(background)), reverse=True
        )
        return (lighter + 0.05) / (darker + 0.05)

    pairs = {
        "正文/纸张": ("#1f2933", "#fffdf8"),
        "辅助文字/纸张": ("#52606d", "#fffdf8"),
        "链接/纸张": ("#1d4ed8", "#fffdf8"),
        "下一章状态/浅蓝": ("#1d4ed8", "#dbeafe"),
        "完成状态/浅绿": (COLORS["success"], "#dcfce7"),
        "警告色/白色": (COLORS["warning"], "#ffffff"),
    }
    for label, (foreground, background) in pairs.items():
        assert contrast(foreground, background) >= 4.5, label


def _formal_notebooks() -> list[Path]:
    return [
        NOTEBOOKS / chapter.part / chapter.filename
        for chapter in CHAPTERS.values()
    ]


def _exercise_pairs(path: Path) -> tuple[tuple[str, str], ...]:
    """Extract the four literal question/answer pairs from one chapter."""

    tree = ast.parse(path.read_text(encoding="utf-8"))
    calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and _call_name(node) == "exercise_block"
    ]
    assert len(calls) == 1, f"{path} 必须且只能有一个章末检测"
    assert len(calls[0].args) == 4, f"{path} 必须包含四类练习"
    return tuple(ast.literal_eval(argument) for argument in calls[0].args)


def _is_marimo_cell(node: ast.AST) -> bool:
    if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
        return False
    return any(
        isinstance(decorator, ast.Attribute)
        and isinstance(decorator.value, ast.Name)
        and decorator.value.id == "app"
        and decorator.attr == "cell"
        for decorator in node.decorator_list
    )


def _call_name(call: ast.Call) -> str:
    function = call.func
    if isinstance(function, ast.Name):
        return function.id
    if isinstance(function, ast.Attribute) and isinstance(function.value, ast.Name):
        return f"{function.value.id}.{function.attr}"
    return ""


def _is_ui_call(node: ast.AST) -> bool:
    return (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and isinstance(node.func.value, ast.Attribute)
        and isinstance(node.func.value.value, ast.Name)
        and node.func.value.value.id == "mo"
        and node.func.value.attr == "ui"
    )


def _top_level_display_calls(cell: ast.FunctionDef) -> list[str]:
    calls = []
    for statement in cell.body:
        if not isinstance(statement, ast.Expr) or not isinstance(
            statement.value, ast.Call
        ):
            continue
        name = _call_name(statement.value)
        if name.startswith("mo.") or name in DISPLAY_HELPERS:
            calls.append(name)
    return calls


def _loaded_references(node: ast.AST) -> list[tuple[str, bool]]:
    """Return loaded names and whether they are used as objects, not attributes."""

    references: list[tuple[str, bool]] = []

    def visit(current: ast.AST, parent: ast.AST | None = None) -> None:
        if isinstance(current, ast.Name) and isinstance(current.ctx, ast.Load):
            references.append(
                (current.id, not isinstance(parent, ast.Attribute))
            )
        for child in ast.iter_child_nodes(current):
            visit(child, current)

    visit(node)
    return references


def test_home_uses_reactive_marimo_controls():
    source = (NOTEBOOKS / "00_home.py").read_text(encoding="utf-8")
    assert "mo.ui.dropdown(" in source
    assert "mo.ui.slider(" in source
    assert "part_filter.value" in source
    assert "completed_through.value" in source
    assert "PAPER_GUIDES" in source
    assert '"章节": (' in source
    assert "论文导读 {_guide_index}/{len(_matching_guides)}" in source


def test_home_exposes_the_shared_bilingual_glossary():
    source = (NOTEBOOKS / "00_home.py").read_text(encoding="utf-8")
    assert "terminology_table()" in source
    assert "潜变量（latent variable）" in source
    assert "得分场（score field）" in source

    html = terminology_table()._repr_html_()
    assert "专业术语中英对照" in html
    assert html.count("<tr>") == len(TERMS) + 1


def test_every_new_term_has_one_shared_first_chapter_explanation():
    assert len(TERMS) >= 90
    for key, term in TERMS.items():
        assert term.chinese.strip(), key
        assert term.english.strip(), key
        assert len(term.explanation) >= 12, key
        assert 1 <= term.introduced_in <= 29, key

    for number in range(1, 30):
        terms = terms_for_chapter(number)
        assert terms, f"第 {number} 章没有登记首次引入术语"
        html = unescape(chapter_terminology(CHAPTERS[number])._repr_html_())
        assert "先认中文，再熟悉英文" in html
        for term in terms:
            assert term.chinese in html
            assert term.english in html
            assert term.explanation in html


def test_ambiguous_terms_use_explicit_course_conventions():
    assert TERMS["latent_variable"].chinese == "潜变量"
    assert TERMS["likelihood"].chinese == "似然"
    assert TERMS["score"].chinese == "得分函数"
    assert "不是模型评分" in TERMS["score"].explanation
    assert TERMS["cfg"].chinese == "无分类器引导"
    assert "尚未完全统一" in TERMS["rectified_flow"].explanation


def test_course_map_uses_semantic_html_instead_of_dynamic_markdown_table():
    table = course_map_table(
        "测试部分",
        [
            {
                "章节": "01",
                "标题": "标题 <不会被当作 HTML>",
                "学习进度": "下一章",
                "入口": "./part/01.py",
            },
            {
                "章节": "02",
                "标题": "第二章",
                "学习进度": "未开始",
                "入口": "./part/02.py",
            },
        ],
    )
    html = table._repr_html_()

    assert '<table class="gm-course-table">' in html
    assert html.count("<tr>") == 3  # one header row and two chapter rows
    assert "标题 &lt;不会被当作 HTML&gt;" in html
    assert (
        f'href="?file={ROOT}/notebooks/part/01.py"'
        in html
    )
    assert 'target="_blank"' in html
    assert 'rel="noopener noreferrer"' in html
    assert "is-next" in html


def test_every_course_map_link_uses_an_existing_workspace_notebook():
    rows = [
        {
            "章节": f"{chapter.number:02d}",
            "标题": chapter.title,
            "学习进度": "未开始",
            "入口": f"./{chapter.part}/{chapter.filename}",
        }
        for chapter in CHAPTERS.values()
    ]
    html = course_map_table("全部章节", rows)._repr_html_()

    assert html.count('href="?file=') == len(CHAPTERS)
    for chapter in CHAPTERS.values():
        notebook_path = NOTEBOOKS / chapter.part / chapter.filename
        assert notebook_path.is_file()
        assert (
            f'href="?file={notebook_path}"' in html
        )


def test_chapter_navigation_handles_first_middle_and_last_chapters():
    first = chapter_navigation(CHAPTERS[1])._repr_html_()
    middle = chapter_navigation(CHAPTERS[15])._repr_html_()
    last = chapter_navigation(CHAPTERS[30])._repr_html_()

    assert f'href="?file={NOTEBOOKS / "00_home.py"}"' in first
    assert "← 上一章" not in first
    assert (
        f'href="?file={NOTEBOOKS / "part01_vae/02_latent_distribution.py"}"'
        in first
    )

    assert (
        f'href="?file={NOTEBOOKS / "part03_diffusion_ddpm/14_iterative_denoising.py"}"'
        in middle
    )
    assert (
        f'href="?file={NOTEBOOKS / "part03_diffusion_ddpm/16_ddpm_posterior_and_reverse.py"}"'
        in middle
    )
    assert "← 上一章" in middle
    assert "下一章 →" in middle

    assert (
        f'href="?file={NOTEBOOKS / "part06_flow_matching/29_optimal_transport_and_rectified_flow.py"}"'
        in last
    )
    assert "下一章 →" not in last
    assert 'aria-label="章节导航"' in last


def test_every_adjacent_chapter_navigation_link_targets_a_real_notebook():
    for number, chapter in CHAPTERS.items():
        html = chapter_navigation(chapter)._repr_html_()
        for adjacent_number in (number - 1, number + 1):
            guide = (
                PAPER_GUIDES_BEFORE_CHAPTER.get(number)
                if adjacent_number == number - 1
                else PAPER_GUIDES_AFTER_CHAPTER.get(number)
            )
            if guide is not None:
                if isinstance(guide, tuple):
                    guide = guide[0]
                notebook = NOTEBOOKS / guide.part / guide.filename
                assert notebook.is_file()
                assert f'href="?file={notebook}"' in html
                continue
            adjacent = CHAPTERS.get(adjacent_number)
            if adjacent is None:
                continue
            notebook = NOTEBOOKS / adjacent.part / adjacent.filename
            assert notebook.is_file()
            assert f'href="?file={notebook}"' in html


def test_vae_paper_guide_forms_a_navigation_bridge():
    first_guide = PAPER_GUIDES["vae_autoencoder_lineage"]
    second_guide = PAPER_GUIDES["vae_aevb"]
    first_path = NOTEBOOKS / first_guide.part / first_guide.filename
    second_path = NOTEBOOKS / second_guide.part / second_guide.filename

    from_chapter_seven = chapter_navigation(CHAPTERS[7])._repr_html_()
    from_chapter_eight = chapter_navigation(CHAPTERS[8])._repr_html_()
    first_navigation = paper_guide_navigation(first_guide)._repr_html_()
    second_navigation = paper_guide_navigation(second_guide)._repr_html_()

    assert f'href="?file={first_path}"' in from_chapter_seven
    assert f'href="?file={second_path}"' in from_chapter_eight
    assert "经典论文导读" in from_chapter_seven
    assert f'href="?file={NOTEBOOKS / CHAPTERS[7].part / CHAPTERS[7].filename}"' in first_navigation
    assert f'href="?file={second_path}"' in first_navigation
    assert f'href="?file={first_path}"' in second_navigation
    assert f'href="?file={NOTEBOOKS / CHAPTERS[8].part / CHAPTERS[8].filename}"' in second_navigation


def test_every_vae_chapter_exposes_curated_paper_coordinates():
    for number in range(1, 14):
        html = chapter_paper_trail(CHAPTERS[number])._repr_html_()
        assert "本章论文坐标" in html
        relevant = [paper for paper in PAPERS.values() if number in paper.chapters]
        assert relevant
        for paper in relevant:
            assert paper.chinese_title in html
            assert paper.url in html


def test_vae_paper_guide_uses_primary_source_math_code_and_feedback():
    guide = PAPER_GUIDES["vae_aevb"]
    path = NOTEBOOKS / guide.part / guide.filename
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source)
    cells = [node for node in tree.body if _is_marimo_cell(node)]

    assert len(cells) >= 18
    assert "https://arxiv.org/abs/1312.6114" in source
    assert "教学性意译" in source
    assert "式 (1)–(3) 严格推导" in source
    assert "可执行核验" in source
    assert "图 1 解剖" in source
    assert "图 4 复现思路" in source
    assert "torch.exp(0.5 * logvar)" in source
    assert source.count("mo.ui.") >= 10
    assert "exercise_block(" in source
    assert "paper_guide_footer(" in source


def test_added_vae_paper_seminars_are_interactive_and_source_grounded():
    expectations = {
        "vae_autoencoder_lineage": ("Jacobian 惩罚", "latent_gap", "hinton_autoencoder"),
        "vae_variants_landmarks": ("IWAE 为什么是下界", "iwae_samples", "normalizing_flows"),
    }
    for guide_key, fragments in expectations.items():
        guide = PAPER_GUIDES[guide_key]
        source = (NOTEBOOKS / guide.part / guide.filename).read_text(encoding="utf-8")
        assert "\t" not in source and "\r" not in source
        assert r"\(" in source
        assert source.count("mo.ui.") >= 3
        assert "exercise_block(" in source
        assert "paper_guide_footer(" in source
        for fragment in fragments:
            assert fragment in source
    variants = PAPER_GUIDES["vae_variants_landmarks"]
    variants_source = (
        NOTEBOOKS / variants.part / variants.filename
    ).read_text(encoding="utf-8")
    assert r"\[" in variants_source
    assert r"\mathcal L_K" in variants_source
    assert r"p_\theta(y,z\mid x)" in variants_source


def test_diffusion_paper_guides_form_navigation_bridge():
    first_guide = PAPER_GUIDES["diffusion_lineage"]
    second_guide = PAPER_GUIDES["ddpm_original"]
    first_path = NOTEBOOKS / first_guide.part / first_guide.filename
    second_path = NOTEBOOKS / second_guide.part / second_guide.filename

    from_chapter_eighteen = chapter_navigation(CHAPTERS[18])._repr_html_()
    from_chapter_nineteen = chapter_navigation(CHAPTERS[19])._repr_html_()
    first_navigation = paper_guide_navigation(first_guide)._repr_html_()
    second_navigation = paper_guide_navigation(second_guide)._repr_html_()

    assert f'href="?file={first_path}"' in from_chapter_eighteen
    assert f'href="?file={second_path}"' in from_chapter_nineteen
    assert f'href="?file={second_path}"' in first_navigation
    assert f'href="?file={first_path}"' in second_navigation
    assert f'href="?file={NOTEBOOKS / CHAPTERS[19].part / CHAPTERS[19].filename}"' in second_navigation


def test_diffusion_paper_guides_are_interactive_rigorous_and_source_grounded():
    expectations = {
        "diffusion_lineage": (
            "条件得分不等于边缘得分",
            "Fisher identity",
            "finite_difference_gradient",
        ),
        "ddpm_original": (
            "式 (4) 严格推导",
            "simple loss",
            "ddpm_posterior_mean_variance",
        ),
    }
    for guide_key, fragments in expectations.items():
        guide = PAPER_GUIDES[guide_key]
        source = (NOTEBOOKS / guide.part / guide.filename).read_text(encoding="utf-8")
        assert "\t" not in source and "\r" not in source
        assert source.count("mo.ui.") >= 3
        assert "exercise_block(" in source
        assert "paper_guide_footer(" in source
        for fragment in fragments:
            assert fragment in source


def test_ddim_and_diffusion_variant_paper_guides_form_navigation_bridges():
    ddim_guide = PAPER_GUIDES["ddim_original"]
    variants_guide = PAPER_GUIDES["diffusion_variants_landmarks"]
    sde_guide = PAPER_GUIDES["score_sde_original"]

    ddim_path = NOTEBOOKS / ddim_guide.part / ddim_guide.filename
    variants_path = NOTEBOOKS / variants_guide.part / variants_guide.filename
    sde_path = NOTEBOOKS / sde_guide.part / sde_guide.filename

    assert f'href="?file={ddim_path}"' in chapter_navigation(CHAPTERS[20])._repr_html_()
    assert f'href="?file={ddim_path}"' in chapter_navigation(CHAPTERS[21])._repr_html_()
    assert f'href="?file={variants_path}"' in chapter_navigation(CHAPTERS[26])._repr_html_()
    assert f'href="?file={sde_path}"' in paper_guide_navigation(variants_guide)._repr_html_()
    assert f'href="?file={variants_path}"' in paper_guide_navigation(sde_guide)._repr_html_()
    assert f'href="?file={NOTEBOOKS / CHAPTERS[27].part / CHAPTERS[27].filename}"' in paper_guide_navigation(sde_guide)._repr_html_()


def test_new_paper_guides_use_original_evidence_with_legends_and_boundaries():
    guide_keys = (
        "vae_autoencoder_lineage",
        "vae_aevb",
        "vae_variants_landmarks",
        "ddim_original",
        "diffusion_variants_landmarks",
        "score_sde_original",
        "flow_matching_lineage",
        "flow_matching_rectified_flow",
    )
    for guide_key in guide_keys:
        guide = PAPER_GUIDES[guide_key]
        source = (NOTEBOOKS / guide.part / guide.filename).read_text(encoding="utf-8")
        assert "paper_evidence_block(" in source
        assert "original_quote=" in source
        assert "figure_path=" in source
        assert "legend=(" in source
        assert "learning_goal=" in source
        assert "evidence_boundary=" in source
        assert source.count("mo.ui.") >= 3
        assert "exercise_block(" in source

    asset_paths = (
        "assets/paper_figures/vae/hinton_figure1_autoencoder.webp",
        "assets/paper_figures/vae/aevb_figure1_graphical_model.webp",
        "assets/paper_figures/vae/aevb_figure4_manifold.webp",
        "assets/paper_figures/vae/vq_vae_figure1_architecture.webp",
        "assets/paper_figures/vae/normalizing_flows_figure1.webp",
        "assets/paper_figures/ddim/figure1_graphical_models.webp",
        "assets/paper_figures/ddim/figure4_speed_quality.webp",
        "assets/paper_figures/diffusion_variants/cfg_figure2_gaussian_guidance.webp",
        "assets/paper_figures/diffusion_variants/ldm_figure3_architecture.webp",
        "assets/paper_figures/diffusion_variants/dit_figure2_scaling.webp",
        "assets/paper_figures/diffusion_variants/score_sde_figure2_overview.webp",
        "assets/paper_figures/flow_matching/neural_ode_figure1.webp",
        "assets/paper_figures/flow_matching/ffjord_figure1.webp",
        "assets/paper_figures/flow_matching/flow_matching_figures2_3.webp",
        "assets/paper_figures/flow_matching/rectified_flow_figure2.webp",
    )
    for relative_path in asset_paths:
        assert (ROOT / relative_path).is_file()


def test_flow_matching_paper_guides_form_final_navigation_bridge():
    lineage = PAPER_GUIDES["flow_matching_lineage"]
    deep_read = PAPER_GUIDES["flow_matching_rectified_flow"]
    lineage_path = NOTEBOOKS / lineage.part / lineage.filename
    deep_read_path = NOTEBOOKS / deep_read.part / deep_read.filename

    from_chapter_thirty = chapter_navigation(CHAPTERS[30])._repr_html_()
    lineage_navigation = paper_guide_navigation(lineage)._repr_html_()
    deep_read_navigation = paper_guide_navigation(deep_read)._repr_html_()

    assert f'href="?file={lineage_path}"' in from_chapter_thirty
    assert f'href="?file={deep_read_path}"' in lineage_navigation
    assert f'href="?file={lineage_path}"' in deep_read_navigation
    assert "进入下一部分" not in deep_read_navigation
    assert "返回首页" in deep_read_navigation


def test_end_of_chapter_assessment_hides_answers_until_requested():
    assessment = exercise_block(
        understanding=("理解问题", "理解答案"),
        calculation=("手算问题", "手算答案"),
        coding=("代码问题", "代码答案"),
        exploration=("探索问题", "探索答案"),
    )
    html = assessment._repr_html_()

    assert "章末学习检测｜先作答，再看答案" in html
    assert html.count("完成作答后，展开参考答案") == 4
    assert html.count("<marimo-accordion") == 4
    assert "<marimo-lazy" not in html
    for answer in ("理解答案", "手算答案", "代码答案", "探索答案"):
        assert answer in html
    assert "建议进入下一章的标准" in html


def test_every_chapter_places_assessment_before_summary_and_navigation():
    for path in _formal_notebooks():
        source = path.read_text(encoding="utf-8")
        assert source.count("exercise_block(") == 1, path
        assert source.count("chapter_footer(") == 1, path
        assert source.index("exercise_block(") < source.index("chapter_footer("), path


def test_every_chapter_has_four_substantive_assessment_pairs():
    for path in _formal_notebooks():
        pairs = _exercise_pairs(path)
        assert len(pairs) == 4
        for question, answer in pairs:
            assert isinstance(question, str) and len(question.strip()) >= 8, path
            assert isinstance(answer, str) and len(answer.strip()) >= 8, path


def test_every_code_exercise_names_a_concrete_code_expression():
    """第三题不能只是口头复述，至少要定位到可修改或诊断的代码。"""

    for path in _formal_notebooks():
        coding_question, coding_answer = _exercise_pairs(path)[2]
        assert "`" in coding_question, f"{path} 的代码题缺少具体代码对象"
        assert "`" in coding_answer, f"{path} 的代码答案缺少可核对实现"


def test_assessment_questions_are_unique_across_the_curriculum():
    questions = [
        question
        for path in _formal_notebooks()
        for question, _answer in _exercise_pairs(path)
    ]
    assert len(questions) == 120
    assert len(questions) == len(set(questions)), "课程中存在重复考题"


def test_every_formal_chapter_is_a_real_reactive_marimo_notebook():
    for path in _formal_notebooks():
        source = path.read_text(encoding="utf-8")
        tree = ast.parse(source)
        cells = [node for node in tree.body if _is_marimo_cell(node)]

        assert 'app = marimo.App(width="medium")' in source, path
        assert '__generated_with = "0.23.13"' in source, path
        assert len(cells) >= 8, f"{path} 的教学 cell 过少"
        assert "chapter_scaffold(" not in source, f"{path} 仍是通用骨架"
        assert "intuition_and_rigor(" in source, f"{path} 缺少双层解释 tabs"
        assert "exercise_block(" in source, f"{path} 缺少折叠练习答案"
        assert "plt.subplots(" in source or "plt.figure(" in source, (
            f"{path} 缺少可视化"
        )
        assert '"__main__"' in source and "app.run()" in source

        ui_definitions: dict[str, int] = {}
        for cell_index, cell in enumerate(cells):
            for node in ast.walk(cell):
                if (
                    isinstance(node, ast.Assign)
                    and len(node.targets) == 1
                    and isinstance(node.targets[0], ast.Name)
                    and _is_ui_call(node.value)
                ):
                    ui_definitions[node.targets[0].id] = cell_index

        assert ui_definitions, f"{path} 缺少本章专属 marimo UI"

        reactive_controls = set()
        for name, definition_index in ui_definitions.items():
            for later_cell in cells[definition_index + 1 :]:
                if any(
                    isinstance(node, ast.Name)
                    and isinstance(node.ctx, ast.Load)
                    and node.id == name
                    for node in ast.walk(later_cell)
                ):
                    reactive_controls.add(name)
                    break
        assert reactive_controls, f"{path} 的 UI 没有驱动下游响应式 cell"

        visible_controls = set()
        for cell in cells:
            local_assignments = {
                statement.targets[0].id: statement.value
                for statement in cell.body
                if isinstance(statement, ast.Assign)
                and len(statement.targets) == 1
                and isinstance(statement.targets[0], ast.Name)
            }
            display_expressions = [
                statement.value
                for statement in cell.body
                if isinstance(statement, ast.Expr)
                and isinstance(statement.value, ast.Call)
            ]
            pending = [
                reference
                for expression in display_expressions
                for reference in _loaded_references(expression)
            ]
            visited = set()
            while pending:
                name, used_as_object = pending.pop()
                if (name, used_as_object) in visited:
                    continue
                visited.add((name, used_as_object))
                if name in ui_definitions and used_as_object:
                    visible_controls.add(name)
                if name in local_assignments:
                    pending.extend(
                        _loaded_references(local_assignments[name])
                    )

        assert visible_controls == set(ui_definitions), (
            f"{path} 存在已定义但未显示的 UI："
            f"{sorted(set(ui_definitions) - visible_controls)}"
        )


def test_display_outputs_are_explicitly_composed():
    """marimo 每个 cell 只有最后一个表达式会成为主输出。

    多个并列展示调用必须放进 mo.vstack/mo.hstack 等容器，否则标题、
    推导地图或练习说明会在真实运行和 HTML 导出时消失。
    """

    for path in sorted(NOTEBOOKS.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for cell in (node for node in tree.body if _is_marimo_cell(node)):
            calls = _top_level_display_calls(cell)
            assert len(calls) <= 1, (
                f"{path}:{cell.lineno} 同一 cell 有多个未组合展示输出：{calls}"
            )


def test_shared_components_use_marimo_tabs_and_offline_accordions():
    source = (ROOT / "src/teaching/components.py").read_text(encoding="utf-8")
    assert "mo.tabs(" in source
    assert "mo.accordion(" in source
    assert "lazy=False" in source


def test_custom_light_surfaces_define_dark_theme_safe_text_colors():
    source = (ROOT / "src/teaching/components.py").read_text(encoding="utf-8")
    assert ".gm-hero h1 {" in source
    assert "font-size: clamp(1.75rem, 3vw, 2.5rem);" in source
    assert ".gm-flow {" in source and "color: var(--gm-ink);" in source
    assert ".gm-course-table {" in source
    assert ".gm-course-section > h2 {" in source
    assert ".markdown.prose table {" in source
    assert "overflow-x: auto;" in source
