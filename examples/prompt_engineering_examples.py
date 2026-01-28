"""
Prompt Engineering Examples

Demonstrates various prompt engineering techniques and best practices
using the template system.

Usage:
    python examples/prompt_engineering_examples.py
"""

import asyncio
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.core.prompts import (
    CHAIN_OF_THOUGHT_TEMPLATE,
    CLASSIFICATION_TEMPLATE,
    CODE_GENERATION_TEMPLATE,
    DATA_ANALYSIS_TEMPLATE,
    QA_TEMPLATE,
    SUMMARIZATION_TEMPLATE,
    get_template,
    list_templates,
)


def print_section(title: str):
    """Print a formatted section header"""
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70 + "\n")


def example_1_summarization():
    """Example 1: Text Summarization"""
    print_section("Example 1: Text Summarization")

    article = """
    Artificial Intelligence has made remarkable progress in recent years,
    particularly in natural language processing and computer vision. Large
    language models like GPT-4 and Claude can now perform tasks that were
    impossible just a few years ago, from writing code to analyzing complex
    documents. However, concerns remain about AI safety, bias, and the
    potential for misuse. Researchers are working on making AI systems more
    transparent, controllable, and aligned with human values. The field
    continues to evolve rapidly, with new breakthroughs announced regularly.
    """

    # Generate prompt with different styles
    styles = ["professional", "casual", "technical", "executive"]

    for style in styles:
        prompt = SUMMARIZATION_TEMPLATE.format(
            text=article.strip(),
            focus_area="key developments and challenges",
            max_length=50,
            style=style,
            audience=f"{style} readers",
        )

        print(f"📝 {style.capitalize()} Style Prompt:")
        print("-" * 70)
        print(prompt)
        print()


def example_2_classification():
    """Example 2: Text Classification"""
    print_section("Example 2: Text Classification")

    examples = [
        "I love this product! Best purchase ever!",
        "Terrible quality, waste of money.",
        "It's okay, nothing special.",
        "Amazing customer service and fast shipping!",
        "Completely broken when it arrived.",
    ]

    categories = "positive, negative, neutral"

    for text in examples:
        prompt = CLASSIFICATION_TEMPLATE.format(
            item_type="customer review",
            categories=categories,
            text=text,
        )

        print(f"💬 Review: '{text}'")
        print("-" * 70)
        print(prompt)
        print()


def example_3_question_answering():
    """Example 3: Question Answering with Context"""
    print_section("Example 3: Question Answering")

    context = """
    FastAPI is a modern, fast (high-performance) web framework for building APIs
    with Python 3.7+ based on standard Python type hints. It's built on top of
    Starlette for the web parts and Pydantic for the data parts. Key features
    include automatic interactive API documentation, data validation using Python
    type hints, async/await support, and dependency injection.
    """

    questions = [
        "What is FastAPI?",
        "What Python version does it require?",
        "What are the key features?",
        "What is it built on?",
    ]

    for question in questions:
        prompt = QA_TEMPLATE.format(context=context.strip(), question=question)

        print(f"❓ Question: {question}")
        print("-" * 70)
        print(prompt)
        print()


def example_4_code_generation():
    """Example 4: Code Generation"""
    print_section("Example 4: Code Generation")

    task = "Create a function that validates email addresses"
    requirements = """
    - Accept email string as input
    - Check for valid format (user@domain.com)
    - Return True if valid, False otherwise
    - Handle edge cases (empty, None, special characters)
    """

    languages = ["python", "javascript", "typescript"]

    for language in languages:
        prompt = CODE_GENERATION_TEMPLATE.format(
            language=language,
            task=task,
            requirements=requirements.strip(),
        )

        print(f"💻 {language.capitalize()} Code Generation:")
        print("-" * 70)
        print(prompt)
        print()


def example_5_data_analysis():
    """Example 5: Data Analysis"""
    print_section("Example 5: Data Analysis")

    sales_data = """
    Month | Revenue | Orders | Avg Order Value
    Jan   | $45,000 | 150    | $300
    Feb   | $52,000 | 173    | $300
    Mar   | $38,000 | 127    | $299
    Apr   | $61,000 | 203    | $300
    May   | $58,000 | 193    | $300
    """

    prompt = DATA_ANALYSIS_TEMPLATE.format(
        dataset_name="Monthly Sales Q1-Q2",
        data=sales_data.strip(),
        num_insights=5,
    )

    print("📊 Sales Data Analysis:")
    print("-" * 70)
    print(prompt)
    print()


def example_6_chain_of_thought():
    """Example 6: Chain-of-Thought Reasoning"""
    print_section("Example 6: Chain-of-Thought Reasoning")

    problems = [
        "If a train travels 120 miles in 2 hours, how long will it take to travel 300 miles at the same speed?",
        "A store sells apples at $3 per pound. If I buy 5 pounds and have a 20% discount coupon, how much do I pay?",
        "If 5 workers can build a wall in 8 days, how many days will it take 8 workers to build the same wall?",
    ]

    for problem in problems:
        prompt = CHAIN_OF_THOUGHT_TEMPLATE.format(problem=problem)

        print(f"🧮 Problem: {problem}")
        print("-" * 70)
        print(prompt)
        print()


def example_7_few_shot_learning():
    """Example 7: Few-Shot Learning Pattern"""
    print_section("Example 7: Few-Shot Learning")

    # Manual few-shot prompt (not using template)
    prompt = """Learn from these examples and then classify the new input:

Example 1:
Input: "I need to return this item"
Intent: return_request

Example 2:
Input: "When will my package arrive?"
Intent: delivery_inquiry

Example 3:
Input: "The product is broken"
Intent: quality_complaint

Example 4:
Input: "How do I use this feature?"
Intent: support_question

Now classify this:
Input: "Can I change my shipping address?"
Intent: """

    print("🎯 Few-Shot Classification:")
    print("-" * 70)
    print(prompt)
    print()


def example_8_structured_output():
    """Example 8: Structured Output (JSON)"""
    print_section("Example 8: Structured Output")

    prompt = """Analyze the following customer feedback and respond in JSON format.

Customer Feedback:
"The product quality is excellent, but shipping took too long. Customer service
was helpful when I called about the delay. Overall satisfied but shipping needs
improvement."

Respond with this exact JSON structure:
{
  "sentiment": "positive|negative|neutral",
  "satisfaction_score": 1-10,
  "aspects": {
    "product_quality": {"rating": 1-5, "comment": "brief comment"},
    "shipping": {"rating": 1-5, "comment": "brief comment"},
    "customer_service": {"rating": 1-5, "comment": "brief comment"}
  },
  "key_issues": ["issue1", "issue2"],
  "recommendations": ["recommendation1", "recommendation2"]
}

Analysis:"""

    print("📋 Structured JSON Output:")
    print("-" * 70)
    print(prompt)
    print()


def example_9_role_based_prompting():
    """Example 9: Role-Based Prompting"""
    print_section("Example 9: Role-Based Prompting")

    roles = [
        {
            "role": "senior software architect",
            "task": "design a user authentication system",
        },
        {"role": "data scientist", "task": "explain gradient descent"},
        {"role": "technical writer", "task": "document this API endpoint"},
        {"role": "security expert", "task": "review this code for vulnerabilities"},
    ]

    for example in roles:
        prompt = f"""You are a {example["role"]} with 10+ years of experience.

Task: {example["task"]}

Provide your professional perspective:"""

        print(f"👤 Role: {example['role'].title()}")
        print("-" * 70)
        print(prompt)
        print()


def example_10_template_comparison():
    """Example 10: Compare Different Template Approaches"""
    print_section("Example 10: Template Comparison")

    text = "Machine learning is transforming industries worldwide."

    # Approach 1: Simple/Direct
    simple = f"Translate this to French: {text}"

    # Approach 2: Structured
    structured = f"""Task: Translation

Source Language: English
Target Language: French
Text: {text}

Translation:"""

    # Approach 3: Detailed with context
    detailed = f"""You are a professional translator specializing in technical content.

Task: Translate the following technical text from English to French.
Requirements:
- Maintain technical accuracy
- Use appropriate technical terminology
- Preserve the original meaning and tone

Source text:
{text}

French translation:"""

    print("🔄 Approach 1 (Simple):")
    print("-" * 70)
    print(simple)
    print()

    print("🔄 Approach 2 (Structured):")
    print("-" * 70)
    print(structured)
    print()

    print("🔄 Approach 3 (Detailed):")
    print("-" * 70)
    print(detailed)
    print()


def example_11_list_all_templates():
    """Example 11: List Available Templates"""
    print_section("Example 11: Available Templates")

    templates = list_templates()

    print("📚 All Available Templates:")
    print("-" * 70)
    for i, template_name in enumerate(templates, 1):
        template = get_template(template_name)
        print(f"{i}. {template_name}")
        print(f"   Description: {template.description}")
        print(f"   Variables: {', '.join(template.variables)}")
        print()


def example_12_prompt_chaining():
    """Example 12: Prompt Chaining Pattern"""
    print_section("Example 12: Prompt Chaining")

    print("🔗 Multi-Step Analysis (Chained Prompts):")
    print("-" * 70)

    # Step 1: Extract key information
    step1 = """Step 1: Extract key entities and topics

Text: "Apple announced a new iPhone with improved camera and longer battery life.
The device will be available next month for $999."

Extract:
- Entities (companies, products, prices)
- Topics (themes, subjects)

Extracted Information:"""

    print("Step 1 Prompt:")
    print(step1)
    print()

    # Step 2: Analyze sentiment (uses output from step 1)
    step2 = """Step 2: Analyze sentiment

Based on extracted entities: Apple, iPhone, $999
Topics: product announcement, features, pricing

Analyze the overall sentiment of the announcement.

Sentiment Analysis:"""

    print("Step 2 Prompt:")
    print(step2)
    print()

    # Step 3: Generate summary (uses outputs from steps 1 and 2)
    step3 = """Step 3: Generate comprehensive summary

Entities: Apple, iPhone, $999
Sentiment: Positive
Topics: product announcement, features, pricing

Create a concise summary incorporating all analyzed information.

Final Summary:"""

    print("Step 3 Prompt:")
    print(step3)
    print()


def main():
    """Run all examples"""
    print("\n" + "=" * 70)
    print("  🎯 PROMPT ENGINEERING EXAMPLES")
    print("  Demonstrating best practices and techniques")
    print("=" * 70)

    # Run all examples
    example_1_summarization()
    example_2_classification()
    example_3_question_answering()
    example_4_code_generation()
    example_5_data_analysis()
    example_6_chain_of_thought()
    example_7_few_shot_learning()
    example_8_structured_output()
    example_9_role_based_prompting()
    example_10_template_comparison()
    example_11_list_all_templates()
    example_12_prompt_chaining()

    # Final summary
    print_section("Summary")
    print("✅ Demonstrated 12 prompt engineering techniques:")
    print("   1. Text Summarization with style variations")
    print("   2. Classification with categories")
    print("   3. Question Answering with context")
    print("   4. Code Generation across languages")
    print("   5. Data Analysis with structured output")
    print("   6. Chain-of-Thought reasoning")
    print("   7. Few-Shot learning patterns")
    print("   8. Structured JSON output")
    print("   9. Role-based prompting")
    print("   10. Template comparison (simple vs detailed)")
    print("   11. Available template exploration")
    print("   12. Prompt chaining for complex tasks")
    print()
    print("💡 Key Takeaways:")
    print("   - Be specific and provide clear instructions")
    print("   - Include relevant context and examples")
    print("   - Structure your prompts for consistent output")
    print("   - Use templates for reusability and maintainability")
    print("   - Test different approaches to find what works best")
    print()
    print("📚 Learn more: See wiki/Prompt-Engineering.md")
    print("=" * 70)


if __name__ == "__main__":
    main()
