import re
from pathlib import Path

from src.teaching.catalog import CHAPTERS, PARTS
from src.teaching.paper_guides import PAPERS, PAPER_GUIDES, PAPER_GUIDES_AFTER_CHAPTER


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


def test_paper_guides_are_nonnumeric_part_end_supplements():
    identities = []
    for key, guide in PAPER_GUIDES.items():
        assert guide.key == key
        assert guide.after_chapter in CHAPTERS
        assert CHAPTERS[guide.after_chapter].part == guide.part
        assert Path(guide.filename).parent.name == "paper_guides"
        path = ROOT / "notebooks" / guide.part / guide.filename
        assert path.is_file(), f"论文导读不存在：{path}"
        assert guide.order >= 1
        assert guide.paper_keys
        assert all(paper_key in PAPERS for paper_key in guide.paper_keys)
        identities.append((guide.part, guide.filename))
    assert len(identities) == len(set(identities))
    for chapter, guides in PAPER_GUIDES_AFTER_CHAPTER.items():
        assert tuple(guide.order for guide in guides) == tuple(
            sorted(guide.order for guide in guides)
        )
        assert all(guide.after_chapter == chapter for guide in guides)


def test_paper_catalog_has_primary_sources_and_reading_targets():
    assert len(PAPERS) >= 36
    assert sum(paper.primary for paper in PAPERS.values()) >= 35
    for key, paper in PAPERS.items():
        assert paper.key == key
        assert paper.title and paper.chinese_title
        assert paper.url.startswith("https://")
        assert len(paper.reading_targets) >= 3
        assert all(1 <= chapter <= 30 for chapter in paper.chapters)


def test_diffusion_chapters_have_primary_paper_coordinates():
    for chapter in range(14, 19):
        relevant = [paper for paper in PAPERS.values() if chapter in paper.chapters]
        assert relevant, f"第 {chapter} 章缺少论文坐标"
        assert any(paper.primary for paper in relevant)

    assert set(PAPER_GUIDES["diffusion_lineage"].paper_keys) >= {
        "diffusion_thermodynamics",
        "ncsn",
        "ddpm",
    }
    assert PAPER_GUIDES["ddpm_original"].paper_keys[0] == "ddpm"


def test_ddim_and_diffusion_variant_chapters_have_primary_paper_coordinates():
    for chapter in range(19, 27):
        relevant = [paper for paper in PAPERS.values() if chapter in paper.chapters]
        assert relevant, f"第 {chapter} 章缺少论文坐标"
        assert any(paper.primary for paper in relevant)

    assert PAPER_GUIDES["ddim_original"].paper_keys[0] == "ddim"
    assert set(PAPER_GUIDES["diffusion_variants_landmarks"].paper_keys) >= {
        "classifier_free_guidance",
        "latent_diffusion",
        "dit",
        "dpm_solver",
        "score_sde",
    }
    assert PAPER_GUIDES["score_sde_original"].paper_keys == ("score_sde",)


def test_flow_matching_chapters_have_primary_paper_coordinates():
    for chapter in range(27, 31):
        relevant = [paper for paper in PAPERS.values() if chapter in paper.chapters]
        assert relevant, f"第 {chapter} 章缺少论文坐标"
        assert any(paper.primary for paper in relevant)

    assert set(PAPER_GUIDES["flow_matching_lineage"].paper_keys) == {
        "neural_ode",
        "ffjord",
        "flow_matching",
    }
    assert PAPER_GUIDES["flow_matching_rectified_flow"].paper_keys == (
        "flow_matching",
        "rectified_flow",
    )


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
