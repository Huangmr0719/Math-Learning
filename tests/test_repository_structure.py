import re
from pathlib import Path
from urllib.parse import unquote


ROOT = Path(__file__).resolve().parents[1]
MARKDOWN_LINK = re.compile(r"(?<!!)\[[^\]]+\]\(([^)]+)\)")


def test_learning_workspace_has_clear_top_level_sections():
    expected_directories = {
        "assets",
        "docs",
        "exports",
        "notebooks",
        "references",
        "scripts",
        "src",
        "tests",
    }
    actual_directories = {
        path.name
        for path in ROOT.iterdir()
        if path.is_dir() and not path.name.startswith(".")
    }
    assert expected_directories <= actual_directories


def test_reference_material_is_not_mixed_into_project_root():
    assert not list(ROOT.glob("*.pdf"))
    assert not (ROOT / "diffusion-models-class").exists()
    assert not (ROOT / "references/courses").exists()


def test_local_vae_papers_are_centralized_and_unique():
    paper_directory = ROOT / "references/papers/vae"
    expected = {
        "auto-encoding-variational-bayes.pdf",
        "from-autoencoder-to-beta-vae.pdf",
        "an-introduction-to-variational-autoencoders.pdf",
        "tutorial-on-variational-autoencoders.pdf",
    }
    assert {path.name for path in paper_directory.glob("*.pdf")} == expected


def test_learning_management_documents_are_grouped_under_docs():
    assert (ROOT / "docs/plans/VAE_学习_ToDO_清单.md").is_file()
    assert (ROOT / "docs/learning-profile/MISSION.md").is_file()
    assert (ROOT / "docs/learning-profile/NOTES.md").is_file()
    assert (ROOT / "docs/learning-profile/learning-records").is_dir()


def test_legacy_derivations_are_not_maintained_in_parallel():
    assert not (ROOT / "archive").exists()
    assert not (ROOT / "math-foundations").exists()
    assert not (ROOT / "vae-derivations").exists()
    assert not (ROOT / "diffusion-derivations").exists()


def test_markdown_files_have_consistent_basic_formatting():
    problems = []
    for path in sorted(ROOT.rglob("*.md")):
        if any(part.startswith(".") for part in path.relative_to(ROOT).parts):
            continue

        text = path.read_text(encoding="utf-8")
        lines = text.splitlines()
        if any(line.rstrip() != line for line in lines):
            problems.append(f"{path.relative_to(ROOT)}: 行尾存在多余空白")
        if any("\t" in line for line in lines):
            problems.append(f"{path.relative_to(ROOT)}: 包含 tab 字符")
        if sum(line.lstrip().startswith("```") for line in lines) % 2:
            problems.append(f"{path.relative_to(ROOT)}: 代码围栏未闭合")

        for target in MARKDOWN_LINK.findall(text):
            target = target.strip().strip("<>")
            if (
                not target
                or target.startswith(("#", "http://", "https://", "mailto:"))
            ):
                continue
            local_target = unquote(target.split("#", 1)[0])
            if not (path.parent / local_target).resolve().exists():
                problems.append(
                    f"{path.relative_to(ROOT)}: 本地链接不存在：{target}"
                )

    assert not problems, "\n".join(problems)


def test_documented_course_launch_uses_a_directory_workspace():
    for relative_path in ("README.md", "learning-path.md"):
        text = (ROOT / relative_path).read_text(encoding="utf-8")
        bash_blocks = re.findall(r"```bash\s*\n(.*?)```", text, re.DOTALL)
        documented_commands = {
            line.strip()
            for block in bash_blocks
            for line in block.splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        }
        assert "marimo edit notebooks/00_home.py" not in documented_commands
        assert "marimo edit notebooks" in documented_commands
