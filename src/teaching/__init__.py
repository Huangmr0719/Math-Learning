"""统一教学组件与课程目录。"""

from .catalog import CHAPTERS, PARTS, ChapterSpec
from .components import (
    chapter_footer,
    chapter_header,
    chapter_navigation,
    chapter_scaffold,
    course_map_table,
    course_styles,
    derivation_map,
    exercise_block,
    intuition_and_rigor,
    knowledge_checklist,
)

__all__ = [
    "CHAPTERS",
    "PARTS",
    "ChapterSpec",
    "chapter_footer",
    "chapter_header",
    "chapter_navigation",
    "chapter_scaffold",
    "course_map_table",
    "course_styles",
    "derivation_map",
    "exercise_block",
    "intuition_and_rigor",
    "knowledge_checklist",
]
