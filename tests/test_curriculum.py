import re
from pathlib import Path

from src.teaching.catalog import CHAPTERS, PARTS


ROOT = Path(__file__).resolve().parents[1]


def test_curriculum_has_six_parts_and_thirty_chapters():
    assert len(PARTS) == 6
    assert list(CHAPTERS) == list(range(1, 31))


def test_every_catalog_entry_has_a_notebook_file():
    for number, chapter in CHAPTERS.items():
        path = ROOT / "notebooks" / chapter.part / chapter.filename
        assert path.is_file(), f"第 {number} 章文件不存在：{path}"


def test_formal_notebook_set_exactly_matches_catalog():
    """防止重号、漏号或遗留旧文件悄悄进入正式课程目录。"""

    expected = {
        (ROOT / "notebooks" / chapter.part / chapter.filename).resolve()
        for chapter in CHAPTERS.values()
    }
    actual = {
        path.resolve()
        for path in (ROOT / "notebooks").glob("part*/*.py")
    }
    assert actual == expected


def test_chapter_numbers_match_catalog_filenames_and_headers():
    for number, chapter in CHAPTERS.items():
        assert chapter.number == number
        assert chapter.filename.startswith(f"{number:02d}_")

        path = ROOT / "notebooks" / chapter.part / chapter.filename
        source = path.read_text(encoding="utf-8")
        assert f"chapter_header(CHAPTERS[{number}]" in source, (
            f"第 {number} 章标题使用了错误的章节号"
        )


def test_internal_headings_do_not_expose_protocol_numbers():
    """章号负责全课程导航；章内标题只表达学习任务，不显示合并协议号。"""

    numbered_heading = re.compile(
        r"##\s+\d+(?:[–-]\d+)?(?:、\d+(?:[–-]\d+)?)*\."
    )
    for chapter in CHAPTERS.values():
        path = ROOT / "notebooks" / chapter.part / chapter.filename
        source = path.read_text(encoding="utf-8")
        assert numbered_heading.search(source) is None, (
            f"{path} 仍包含章内协议编号"
        )


def test_written_parts_and_remaining_scaffolds_match_current_plan():
    assert all(CHAPTERS[number].status == "精写" for number in range(1, 31))


def test_notebooks_do_not_contain_automatic_network_downloads():
    forbidden_fragments = (
        "download=True",
        "requests.get(",
        "urlretrieve(",
        "snapshot_download(",
        "hf_hub_download(",
    )
    for path in (ROOT / "notebooks").rglob("*.py"):
        source = path.read_text(encoding="utf-8")
        for fragment in forbidden_fragments:
            assert fragment not in source, f"{path} 包含自动下载调用：{fragment}"


def test_every_formal_chapter_contains_core_teaching_protocol():
    required_fragments = (
        "## 本章为什么存在",
        "## 你已经知道什么",
        "derivation_map(",
        "mo.ui.",
        "exercise_block(",
        "chapter_footer(",
    )
    for number, chapter in CHAPTERS.items():
        path = ROOT / "notebooks" / chapter.part / chapter.filename
        source = path.read_text(encoding="utf-8")
        for fragment in required_fragments:
            assert fragment in source, f"第 {number} 章缺少教学协议：{fragment}"
        assert ("错误" in source or "反例" in source), (
            f"第 {number} 章缺少错误或反例教学"
        )


def test_reviewed_math_regressions_remain_fixed():
    kl_source = (
        ROOT / "notebooks/part01_vae/04_kl_divergence.py"
    ).read_text(encoding="utf-8")
    assert r"\sum_{i:p_i>0}q_i\le 1" in kl_source

    flow_source = (
        ROOT / "notebooks/part02_vae_variants/13_flexible_posterior.py"
    ).read_text(encoding="utf-8")
    assert r"f(u)=u^3" in flow_source

    sde_source = (
        ROOT
        / "notebooks/part05_diffusion_variants"
        / "26_score_sde_and_probability_flow_ode.py"
    ).read_text(encoding="utf-8")
    assert "simulate_ornstein_uhlenbeck" in sde_source
    assert "_x=-.5*_x*_dt" not in sde_source


def test_reviewed_assessment_regressions_remain_fixed():
    collapse_source = (
        ROOT / "notebooks/part02_vae_variants/09_posterior_collapse.py"
    ).read_text(encoding="utf-8")
    assert "KL-threshold 代理" in collapse_source
    assert "posterior mean 跨数据的方差" in collapse_source

    markov_source = (
        ROOT / "notebooks/part03_diffusion_ddpm/14_iterative_denoising.py"
    ).read_text(encoding="utf-8")
    assert "有限 T 或噪声不足时" in markov_source

    ddpm_source = (
        ROOT / "notebooks/part03_diffusion_ddpm/18_minimal_ddpm.py"
    ).read_text(encoding="utf-8")
    assert "零基代码索引" in ddpm_source
    assert "论文的一基 `t=1`" in ddpm_source

    sde_source = (
        ROOT
        / "notebooks/part05_diffusion_variants"
        / "26_score_sde_and_probability_flow_ode.py"
    ).read_text(encoding="utf-8")
    assert "标准 Brownian motion" in sde_source
    assert "0.5 * g(t)**2 * score(x,t)" in sde_source

    transport_source = (
        ROOT
        / "notebooks/part06_flow_matching"
        / "29_optimal_transport_and_rectified_flow.py"
    ).read_text(encoding="utf-8")
    assert "排序配对是最优的" in transport_source
    assert "更高维或不同约束下" in transport_source
