"""统一教学组件与课程目录。"""

from .catalog import CHAPTERS, PARTS, ChapterSpec
from .components import (
    chapter_footer,
    chapter_header,
    chapter_navigation,
    chapter_scaffold,
    chapter_terminology,
    course_map_table,
    course_styles,
    derivation_map,
    exercise_block,
    intuition_and_rigor,
    knowledge_checklist,
    terminology_table,
)
from .terminology import TERMS, TermSpec, terms_for_chapter

__all__ = [
    "CHAPTERS",
    "PARTS",
    "ChapterSpec",
    "TERMS",
    "TermSpec",
    "chapter_footer",
    "chapter_header",
    "chapter_navigation",
    "chapter_scaffold",
    "chapter_terminology",
    "course_map_table",
    "course_styles",
    "derivation_map",
    "exercise_block",
    "intuition_and_rigor",
    "knowledge_checklist",
    "terminology_table",
    "terms_for_chapter",
]
