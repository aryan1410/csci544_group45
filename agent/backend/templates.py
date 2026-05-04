from typing import Final

from constants import (
    TEMPLATE_BULLET_SUMMARY,
    TEMPLATE_TWO_COLUMN,
    TEMPLATE_DETAILED_REPORT,
)


ONE_WORD_ANSWER_TEMPLATE: Final[str] = (
    "You are a precise fact-answering assistant. "
    "Answer the question with EXACTLY one word or the shortest possible phrase (1-3 words maximum). "
    "Do NOT write explanations, sentences, bullet points, tables, or research reports. "
    "Output ONLY the answer word or phrase, nothing else."
)

BULLET_SUMMARY_TEMPLATE: Final[str] = ONE_WORD_ANSWER_TEMPLATE

TWO_COLUMN_TEMPLATE: Final[str] = ONE_WORD_ANSWER_TEMPLATE

DETAILED_REPORT_TEMPLATE: Final[str] = ONE_WORD_ANSWER_TEMPLATE


REPORT_TEMPLATES: Final[dict[str, str]] = {
    TEMPLATE_BULLET_SUMMARY: BULLET_SUMMARY_TEMPLATE,
    TEMPLATE_TWO_COLUMN: TWO_COLUMN_TEMPLATE,
    TEMPLATE_DETAILED_REPORT: DETAILED_REPORT_TEMPLATE,
}


def get_template(template_name: str) -> str:
    return REPORT_TEMPLATES[template_name]


def get_available_templates() -> list[str]:
    return list(REPORT_TEMPLATES.keys())


def add_provider_specific_instructions(
    template: str,
    is_gemini: bool,
    template_name: str
) -> str:
    if not is_gemini:
        return template

    if template_name == TEMPLATE_TWO_COLUMN:
        additional = (
            "\n\n**CRITICAL INSTRUCTIONS FOR THIS TASK**: "
            "You MUST output ONLY a markdown table, nothing else. "
            "NO introduction, NO explanation, NO conclusion. "
            "Maximum 12 rows. Each cell: 1-2 sentences maximum. "
            "Start directly with: | Claim | Evidence |"
        )
    else:
        additional = "\nBe concise and focused. Prioritize quality over length."

    return template + additional