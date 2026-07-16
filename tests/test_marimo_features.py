import ast
from pathlib import Path

from src.teaching.catalog import CHAPTERS
from src.teaching.components import course_map_table


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


def _formal_notebooks() -> list[Path]:
    return [
        NOTEBOOKS / chapter.part / chapter.filename
        for chapter in CHAPTERS.values()
    ]


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
    assert 'href="?file=part/01.py"' in html
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
            f'href="?file={chapter.part}/{chapter.filename}"' in html
        )


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


def test_shared_components_use_marimo_tabs_and_lazy_accordions():
    source = (ROOT / "src/teaching/components.py").read_text(encoding="utf-8")
    assert "mo.tabs(" in source
    assert "mo.accordion(" in source
    assert "lazy=True" in source


def test_custom_light_surfaces_define_dark_theme_safe_text_colors():
    source = (ROOT / "src/teaching/components.py").read_text(encoding="utf-8")
    assert ".gm-hero h1 {" in source
    assert "font-size: clamp(1.75rem, 3vw, 2.5rem);" in source
    assert ".gm-flow {" in source and "color: var(--gm-ink);" in source
    assert ".gm-course-table {" in source
    assert ".gm-course-section > h2 {" in source
    assert ".markdown.prose table {" in source
    assert "overflow-x: auto;" in source
