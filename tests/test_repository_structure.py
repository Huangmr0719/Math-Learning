from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


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
