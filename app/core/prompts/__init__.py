"""
Prompt Engineering Module

Centralized prompt templates and utilities for consistent,
maintainable prompt engineering across the application.

Usage:
    from app.core.prompts import get_template, SUMMARIZATION_TEMPLATE

    # Use pre-defined template
    prompt = SUMMARIZATION_TEMPLATE.format(
        text="...",
        focus_area="key findings",
        max_length=200,
        style="professional",
        audience="developers"
    )

    # Or get by name
    template = get_template("summarization")
    prompt = template.format(...)
"""

from app.core.prompts.templates import (
    # Content templates
    BLOG_POST_TEMPLATE,
    # Advanced templates
    CHAIN_OF_THOUGHT_TEMPLATE,
    CLASSIFICATION_TEMPLATE,
    CODE_EXPLANATION_TEMPLATE,
    # Code templates
    CODE_GENERATION_TEMPLATE,
    CODE_REVIEW_TEMPLATE,
    # Support templates
    CUSTOMER_SUPPORT_TEMPLATE,
    # Business templates
    DATA_ANALYSIS_TEMPLATE,
    DECISION_ANALYSIS_TEMPLATE,
    EMAIL_TEMPLATE,
    EXTRACTION_TEMPLATE,
    FAQ_GENERATION_TEMPLATE,
    QA_TEMPLATE,
    REPORT_GENERATION_TEMPLATE,
    SELF_REFLECTION_TEMPLATE,
    SOCIAL_MEDIA_TEMPLATE,
    # General templates
    SUMMARIZATION_TEMPLATE,
    TRANSLATION_TEMPLATE,
    PromptTemplate,
    get_template,
    get_template_info,
    list_templates,
)

__all__ = [
    "PromptTemplate",
    "get_template",
    "list_templates",
    "get_template_info",
    # Templates
    "SUMMARIZATION_TEMPLATE",
    "CLASSIFICATION_TEMPLATE",
    "EXTRACTION_TEMPLATE",
    "QA_TEMPLATE",
    "TRANSLATION_TEMPLATE",
    "CODE_GENERATION_TEMPLATE",
    "CODE_REVIEW_TEMPLATE",
    "CODE_EXPLANATION_TEMPLATE",
    "BLOG_POST_TEMPLATE",
    "EMAIL_TEMPLATE",
    "SOCIAL_MEDIA_TEMPLATE",
    "DATA_ANALYSIS_TEMPLATE",
    "REPORT_GENERATION_TEMPLATE",
    "DECISION_ANALYSIS_TEMPLATE",
    "CUSTOMER_SUPPORT_TEMPLATE",
    "FAQ_GENERATION_TEMPLATE",
    "CHAIN_OF_THOUGHT_TEMPLATE",
    "SELF_REFLECTION_TEMPLATE",
]
