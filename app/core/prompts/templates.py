"""
Prompt Template System

Reusable prompt templates with variable substitution for consistent,
maintainable prompt engineering across the application.

Usage:
    from app.core.prompts.templates import SUMMARIZATION_TEMPLATE

    prompt = SUMMARIZATION_TEMPLATE.format(
        text="Long article here...",
        focus_area="key findings",
        max_length=200,
        style="professional",
        audience="technical stakeholders"
    )
"""

import json
from string import Template
from typing import Any, Dict, List, Optional


class PromptTemplate:
    """Reusable prompt template with variable substitution"""

    def __init__(
        self,
        template: str,
        name: str = "",
        description: str = "",
        variables: Optional[List[str]] = None,
        default_values: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize prompt template

        Args:
            template: Template string with $variables
            name: Template name for identification
            description: What this template does
            variables: List of required variable names
            default_values: Default values for optional variables
        """
        self.template = Template(template)
        self.raw_template = template
        self.name = name
        self.description = description
        self.variables = variables or []
        self.default_values = default_values or {}

    def format(self, **kwargs) -> str:
        """
        Format template with variables

        Args:
            **kwargs: Variable values to substitute

        Returns:
            Formatted prompt string

        Raises:
            ValueError: If required variables are missing
        """
        # Merge with defaults
        values = {**self.default_values, **kwargs}

        # Check for missing required variables
        missing = [v for v in self.variables if v not in values]
        if missing:
            raise ValueError(f"Missing required variables: {missing}")

        return self.template.safe_substitute(**values)

    def __repr__(self):
        return f"PromptTemplate(name='{self.name}', variables={self.variables})"

    def to_dict(self) -> dict:
        """Convert template to dictionary for serialization"""
        return {
            "name": self.name,
            "description": self.description,
            "template": self.raw_template,
            "variables": self.variables,
            "default_values": self.default_values,
        }


# ============================================================================
# General Purpose Templates
# ============================================================================

SUMMARIZATION_TEMPLATE = PromptTemplate(
    name="summarization",
    description="Summarize text with specific focus and constraints",
    template="""Summarize the following text focusing on $focus_area.

Requirements:
- Maximum length: $max_length words
- Writing style: $style
- Target audience: $audience

Text to summarize:
$text

Summary:""",
    variables=["text", "focus_area", "max_length", "style", "audience"],
)


CLASSIFICATION_TEMPLATE = PromptTemplate(
    name="classification",
    description="Classify text into predefined categories",
    template="""Classify the following $item_type into one of these categories:
$categories

$item_type to classify:
$text

Respond with only the category name.

Category:""",
    variables=["item_type", "categories", "text"],
)


EXTRACTION_TEMPLATE = PromptTemplate(
    name="extraction",
    description="Extract structured information from unstructured text",
    template="""Extract the following information from the text:
$fields_to_extract

Text:
$text

Respond in JSON format with this schema:
{
$json_schema
}

Extracted information:""",
    variables=["fields_to_extract", "text", "json_schema"],
)


QA_TEMPLATE = PromptTemplate(
    name="qa",
    description="Answer questions based on provided context",
    template="""Answer the question based on the following context.
If the answer cannot be found in the context, respond with "I don't have enough information to answer this question."

Context:
$context

Question: $question

Answer:""",
    variables=["context", "question"],
)


TRANSLATION_TEMPLATE = PromptTemplate(
    name="translation",
    description="Translate text between languages",
    template="""Translate the following text from $source_language to $target_language.
Maintain the original tone and style.

Text to translate:
$text

Translation:""",
    variables=["source_language", "target_language", "text"],
)


# ============================================================================
# Code-Related Templates
# ============================================================================

CODE_GENERATION_TEMPLATE = PromptTemplate(
    name="code_generation",
    description="Generate code with specifications and requirements",
    template="""Generate $language code for the following task.

Task: $task

Requirements:
$requirements

Guidelines:
- Include clear comments
- Add error handling
- Use type hints where applicable
- Follow best practices

Code:""",
    variables=["language", "task", "requirements"],
)


CODE_REVIEW_TEMPLATE = PromptTemplate(
    name="code_review",
    description="Review code for issues and improvements",
    template="""Review this $language code for:
1. Bugs and errors
2. Performance issues
3. Security vulnerabilities
4. Code style and best practices
5. Potential improvements

Code:
```$language
$code
```

Provide your review in this format:

## Issues Found
- [List any issues with severity: critical/high/medium/low]

## Suggestions
- [List improvement suggestions]

## Overall Assessment
[Brief summary and rating 1-10]

Review:""",
    variables=["language", "code"],
)


CODE_EXPLANATION_TEMPLATE = PromptTemplate(
    name="code_explanation",
    description="Explain code in plain language",
    template="""Explain this $language code to a $audience_level programmer.

Code:
```$language
$code
```

Include:
1. What the code does (high-level overview)
2. How it works (step-by-step explanation)
3. Key concepts used
4. Potential use cases

Explanation:""",
    variables=["language", "code", "audience_level"],
)


# ============================================================================
# Content Generation Templates
# ============================================================================

BLOG_POST_TEMPLATE = PromptTemplate(
    name="blog_post",
    description="Generate blog posts with SEO optimization",
    template="""Write a blog post about: $topic

Requirements:
- Target audience: $audience
- Tone: $tone
- Length: $word_count words
- SEO keywords to include: $keywords

Structure:
1. Compelling headline
2. Introduction with hook
3. 3-5 main sections with subheadings
4. Conclusion with call-to-action
5. Meta description (150 characters max)

Blog post:""",
    variables=["topic", "audience", "tone", "word_count", "keywords"],
    default_values={"tone": "professional", "word_count": "800"},
)


EMAIL_TEMPLATE = PromptTemplate(
    name="email",
    description="Compose professional emails",
    template="""Compose a professional email with the following details:

Purpose: $purpose
Recipient: $recipient
Tone: $tone
Key points to include:
$key_points

Write a clear, concise email that achieves the purpose while maintaining a $tone tone.

Email:""",
    variables=["purpose", "recipient", "tone", "key_points"],
    default_values={"tone": "professional"},
)


SOCIAL_MEDIA_TEMPLATE = PromptTemplate(
    name="social_media",
    description="Create social media posts",
    template="""Create a $platform post about: $topic

Requirements:
- Target audience: $audience
- Tone: $tone
- Include: $must_include
- Character limit: $char_limit

Make it engaging and shareable!

Post:""",
    variables=["platform", "topic", "audience", "tone", "must_include", "char_limit"],
    default_values={"tone": "casual", "char_limit": "280"},
)


# ============================================================================
# Business & Analytics Templates
# ============================================================================

DATA_ANALYSIS_TEMPLATE = PromptTemplate(
    name="data_analysis",
    description="Analyze data and provide insights",
    template="""Analyze the following dataset and provide insights.

Dataset: $dataset_name
Data:
$data

Analysis requirements:
- Identify $num_insights key insights
- Find trends and patterns
- Flag anomalies or outliers
- Provide actionable recommendations

Output format (JSON):
{
  "summary": "One paragraph overview",
  "insights": [
    {"insight": "description", "importance": "high|medium|low", "evidence": "supporting data"}
  ],
  "trends": ["trend1", "trend2"],
  "anomalies": ["anomaly1", "anomaly2"],
  "recommendations": ["recommendation1", "recommendation2"]
}

Analysis:""",
    variables=["dataset_name", "data", "num_insights"],
    default_values={"num_insights": "5"},
)


REPORT_GENERATION_TEMPLATE = PromptTemplate(
    name="report_generation",
    description="Generate business reports",
    template="""Generate a $report_type report for $time_period.

Key metrics:
$metrics

Context:
$context

Include:
1. Executive summary
2. Key findings
3. Detailed analysis
4. Recommendations
5. Next steps

Report:""",
    variables=["report_type", "time_period", "metrics", "context"],
)


DECISION_ANALYSIS_TEMPLATE = PromptTemplate(
    name="decision_analysis",
    description="Analyze decisions with pros/cons",
    template="""Analyze the following decision:

Decision: $decision
Options: $options
Context: $context
Criteria: $criteria

Provide:
1. Pros and cons for each option
2. Risk assessment
3. Recommendation with reasoning
4. Potential outcomes

Analysis:""",
    variables=["decision", "options", "context", "criteria"],
)


# ============================================================================
# Customer Support Templates
# ============================================================================

CUSTOMER_SUPPORT_TEMPLATE = PromptTemplate(
    name="customer_support",
    description="Generate customer support responses",
    template="""You are a customer support agent for $company_name.

Customer information:
- Name: $customer_name
- Account type: $account_type
- Issue history: $previous_issues

Guidelines:
- Be empathetic and professional
- Provide clear, actionable solutions
- Escalate if: $escalation_criteria

Customer message:
$customer_message

Your response:""",
    variables=[
        "company_name",
        "customer_name",
        "account_type",
        "previous_issues",
        "escalation_criteria",
        "customer_message",
    ],
    default_values={
        "previous_issues": "None",
        "escalation_criteria": "Issue cannot be resolved or customer is very upset",
    },
)


FAQ_GENERATION_TEMPLATE = PromptTemplate(
    name="faq_generation",
    description="Generate FAQ entries",
    template="""Generate a comprehensive FAQ entry for: $topic

Context: $context
Target audience: $audience

Include:
- Clear, concise question
- Detailed answer (2-3 paragraphs)
- Related questions/topics
- Links or references (if applicable)

FAQ Entry:""",
    variables=["topic", "context", "audience"],
    default_values={"audience": "general users"},
)


# ============================================================================
# Advanced Reasoning Templates
# ============================================================================

CHAIN_OF_THOUGHT_TEMPLATE = PromptTemplate(
    name="chain_of_thought",
    description="Step-by-step reasoning for complex problems",
    template="""Solve this problem using step-by-step reasoning:

Problem: $problem

Let's break this down:
1. First, identify what information we have
2. Then, determine what we need to find
3. Next, plan our approach
4. Finally, execute and verify

Step 1 - Given Information:""",
    variables=["problem"],
)


SELF_REFLECTION_TEMPLATE = PromptTemplate(
    name="self_reflection",
    description="Self-critique and improvement",
    template="""Initial response:
$initial_response

Now, critically review this response:
1. What could be improved?
2. What assumptions were made?
3. What edge cases were missed?
4. How can we make it better?

Reflection:""",
    variables=["initial_response"],
)


# ============================================================================
# Template Registry
# ============================================================================

ALL_TEMPLATES = {
    # General
    "summarization": SUMMARIZATION_TEMPLATE,
    "classification": CLASSIFICATION_TEMPLATE,
    "extraction": EXTRACTION_TEMPLATE,
    "qa": QA_TEMPLATE,
    "translation": TRANSLATION_TEMPLATE,
    # Code
    "code_generation": CODE_GENERATION_TEMPLATE,
    "code_review": CODE_REVIEW_TEMPLATE,
    "code_explanation": CODE_EXPLANATION_TEMPLATE,
    # Content
    "blog_post": BLOG_POST_TEMPLATE,
    "email": EMAIL_TEMPLATE,
    "social_media": SOCIAL_MEDIA_TEMPLATE,
    # Business
    "data_analysis": DATA_ANALYSIS_TEMPLATE,
    "report_generation": REPORT_GENERATION_TEMPLATE,
    "decision_analysis": DECISION_ANALYSIS_TEMPLATE,
    # Support
    "customer_support": CUSTOMER_SUPPORT_TEMPLATE,
    "faq_generation": FAQ_GENERATION_TEMPLATE,
    # Advanced
    "chain_of_thought": CHAIN_OF_THOUGHT_TEMPLATE,
    "self_reflection": SELF_REFLECTION_TEMPLATE,
}


def get_template(name: str) -> PromptTemplate:
    """
    Get a template by name

    Args:
        name: Template name

    Returns:
        PromptTemplate instance

    Raises:
        ValueError: If template not found
    """
    if name not in ALL_TEMPLATES:
        available = ", ".join(ALL_TEMPLATES.keys())
        raise ValueError(f"Template '{name}' not found. Available: {available}")

    return ALL_TEMPLATES[name]


def list_templates() -> List[str]:
    """List all available template names"""
    return list(ALL_TEMPLATES.keys())


def get_template_info(name: str) -> dict:
    """
    Get information about a template

    Args:
        name: Template name

    Returns:
        Dictionary with template information
    """
    template = get_template(name)
    return template.to_dict()
