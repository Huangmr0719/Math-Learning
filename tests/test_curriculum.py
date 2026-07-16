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
        "## 1.",
        "## 2.",
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
