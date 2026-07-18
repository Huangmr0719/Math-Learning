"""统一教学组件与课程目录。"""

from .catalog import CHAPTERS, PARTS, ChapterSpec
from .paper_guides import PAPERS, PAPER_GUIDES, PaperGuideSpec, PaperSource
from .components import (
    chapter_footer,
    chapter_header,
    chapter_navigation,
    chapter_paper_trail,
    chapter_scaffold,
    chapter_terminology,
    course_map_table,
    course_styles,
    derivation_map,
    exercise_block,
    intuition_and_rigor,
    knowledge_checklist,
    paper_guide_footer,
    paper_guide_header,
    paper_guide_navigation,
    paper_evidence_block,
    terminology_table,
)
from .terminology import TERMS, TermSpec, terms_for_chapter

__all__ = [
    "CHAPTERS",
    "PARTS",
    "PAPER_GUIDES",
    "PAPERS",
    "ChapterSpec",
    "PaperGuideSpec",
    "PaperSource",
    "TERMS",
    "TermSpec",
    "chapter_footer",
    "chapter_header",
    "chapter_navigation",
    "chapter_paper_trail",
    "chapter_scaffold",
    "chapter_terminology",
    "course_map_table",
    "course_styles",
    "derivation_map",
    "exercise_block",
    "intuition_and_rigor",
    "knowledge_checklist",
    "paper_guide_footer",
    "paper_guide_header",
    "paper_guide_navigation",
    "paper_evidence_block",
    "terminology_table",
    "terms_for_chapter",
]
