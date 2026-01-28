"""
Cost Optimization Examples

Demonstrates various techniques for reducing LLM API costs while
maintaining quality and performance.

Usage:
    python examples/cost_optimization_examples.py
"""

import sys
from datetime import datetime, timedelta
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.services.monitoring import usage_tracker


def print_section(title: str):
    """Print formatted section header"""
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70 + "\n")


def example_1_track_usage():
    """Example 1: Track API Usage"""
    print_section("Example 1: Track API Usage")

    print("📊 Logging sample API requests...")

    # Simulate some API requests
    requests = [
        {"model": "gpt-3.5-turbo", "input": 500, "output": 200, "cost": 0.0014},
        {"model": "gpt-4-turbo", "input": 800, "output": 300, "cost": 0.0170},
        {"model": "gpt-3.5-turbo", "input": 300, "output": 150, "cost": 0.0009},
        {"model": "gpt-4", "input": 1000, "output": 500, "cost": 0.0400},
        {"model": "claude-3-haiku", "input": 600, "output": 250, "cost": 0.0005},
    ]

    for req in requests:
        usage_tracker.log_request(
            model=req["model"],
            input_tokens=req["input"],
            output_tokens=req["output"],
            cost=req["cost"],
            user_id="demo_user",
            endpoint="/api/chat",
        )

    print("✅ Logged 5 requests\n")

    # Get statistics
    stats = usage_tracker.get_stats()

    print("📈 Statistics:")
    print(f"   Total requests: {stats['total_requests']}")
    print(f"   Total cost: ${stats['total_cost']:.4f}")
    print(f"   Total tokens: {stats['total_tokens']:,}")
    print(f"   Avg cost/request: ${stats['avg_cost_per_request']:.4f}")
    print(f"   Avg tokens/request: {stats['avg_tokens_per_request']:.1f}")

    print("\n📊 By Model:")
    for model, data in stats["by_model"].items():
        print(f"   {model}:")
        print(f"      Requests: {data['requests']}")
        print(f"      Cost: ${data['cost']:.4f}")
        print(f"      Tokens: {data['tokens']:,}")


def example_2_calculate_costs():
    """Example 2: Calculate and Compare Costs"""
    print_section("Example 2: Calculate and Compare Costs")

    def calculate_cost(input_tokens: int, output_tokens: int, model: str) -> float:
        """Calculate cost for API call"""
        pricing = {
            "gpt-4-turbo": {"input": 10, "output": 30},
            "gpt-3.5-turbo": {"input": 0.5, "output": 1.5},
            "claude-3-opus": {"input": 15, "output": 75},
            "claude-3-haiku": {"input": 0.25, "output": 1.25},
        }

        if model not in pricing:
            return 0.0

        prices = pricing[model]
        input_cost = (input_tokens / 1_000_000) * prices["input"]
        output_cost = (output_tokens / 1_000_000) * prices["output"]

        return input_cost + output_cost

    # Example request
    input_tokens = 1000
    output_tokens = 500

    print(f"📝 Request: {input_tokens} input + {output_tokens} output tokens\n")

    print("💰 Cost Comparison:")
    models = ["gpt-3.5-turbo", "gpt-4-turbo", "claude-3-haiku", "claude-3-opus"]

    for model in models:
        cost = calculate_cost(input_tokens, output_tokens, model)
        print(f"   {model:20} ${cost:.6f}")

    # Calculate savings
    expensive = calculate_cost(input_tokens, output_tokens, "claude-3-opus")
    cheap = calculate_cost(input_tokens, output_tokens, "gpt-3.5-turbo")
    savings_pct = ((expensive - cheap) / expensive) * 100

    print(f"\n💡 Savings using GPT-3.5 vs Claude Opus: {savings_pct:.0f}%")


def example_3_prompt_compression():
    """Example 3: Prompt Compression"""
    print_section("Example 3: Prompt Compression")

    def compress_prompt(prompt: str) -> str:
        """Compress prompt while maintaining meaning"""
        # Remove extra whitespace
        compressed = " ".join(prompt.split())

        # Remove filler words
        fillers = ["please", "kindly", "very", "really", "just", "actually"]
        words = compressed.split()
        compressed = " ".join(w for w in words if w.lower() not in fillers)

        # Use abbreviations
        replacements = {
            "for example": "e.g.",
            "that is": "i.e.",
            "and so forth": "etc.",
        }
        for long, short in replacements.items():
            compressed = compressed.replace(long, short)

        return compressed

    original = """Please, could you very kindly summarize this document for me?
    I would really appreciate it if you could just focus on the main points,
    for example the key findings and recommendations. Thank you very much!"""

    compressed = compress_prompt(original)

    # Estimate tokens (rough: 1 token ≈ 4 chars)
    orig_tokens = len(original) // 4
    comp_tokens = len(compressed) // 4
    savings_pct = ((orig_tokens - comp_tokens) / orig_tokens) * 100

    print("📝 Original prompt:")
    print(f"   {original}")
    print(f"   Estimated tokens: {orig_tokens}")

    print("\n✂️  Compressed prompt:")
    print(f"   {compressed}")
    print(f"   Estimated tokens: {comp_tokens}")

    print(f"\n💰 Token savings: {savings_pct:.0f}%")


def example_4_context_management():
    """Example 4: Smart Context Management"""
    print_section("Example 4: Smart Context Management")

    # Simulate conversation history
    conversation = [
        {"role": "user", "content": "What is machine learning?"},
        {"role": "assistant", "content": "Machine learning is a subset of AI..."},
        {"role": "user", "content": "Tell me about neural networks"},
        {"role": "assistant", "content": "Neural networks are computing systems..."},
        {"role": "user", "content": "What about deep learning?"},
        {"role": "assistant", "content": "Deep learning uses multiple layers..."},
        {"role": "user", "content": "Can you give examples?"},
    ]

    def sliding_window(messages: list, window_size: int = 4) -> list:
        """Keep only recent N messages"""
        return messages[-window_size:]

    def estimate_tokens(messages: list) -> int:
        """Estimate tokens in message list"""
        total_text = " ".join([m["content"] for m in messages])
        return len(total_text) // 4

    full_tokens = estimate_tokens(conversation)
    windowed = sliding_window(conversation, window_size=4)
    windowed_tokens = estimate_tokens(windowed)
    savings_pct = ((full_tokens - windowed_tokens) / full_tokens) * 100

    print(f"💬 Full conversation: {len(conversation)} messages")
    print(f"   Estimated tokens: {full_tokens}")

    print(f"\n🪟 Sliding window (4 messages):")
    print(f"   Messages kept: {len(windowed)}")
    print(f"   Estimated tokens: {windowed_tokens}")

    print(f"\n💰 Token savings: {savings_pct:.0f}%")


def example_5_model_selection():
    """Example 5: Smart Model Selection"""
    print_section("Example 5: Smart Model Selection")

    def route_to_model(task: str) -> str:
        """Select appropriate model for task"""
        task_lower = task.lower()

        # Simple tasks
        if any(word in task_lower for word in ["classify", "extract", "yes/no"]):
            return "gpt-3.5-turbo"

        # Complex reasoning
        if any(word in task_lower for word in ["analyze", "explain why", "reason"]):
            return "gpt-4-turbo"

        # Default
        return "gpt-3.5-turbo"

    tasks = [
        "Classify this email as spam or not spam",
        "Analyze the economic implications of this policy",
        "Extract names and dates from this text",
        "Explain why the stock market crashed in 2008",
        "Is this review positive? (yes/no)",
    ]

    print("🎯 Task Routing:\n")

    for task in tasks:
        model = route_to_model(task)
        cost_indicator = "💲" if "gpt-4" in model else "💰"
        print(f"{cost_indicator} {model:20} ← {task}")


def example_6_budget_monitoring():
    """Example 6: Budget Monitoring"""
    print_section("Example 6: Budget Monitoring")

    daily_budget = 10.0

    # Check budget status
    budget_status = usage_tracker.check_budget(daily_budget)

    print(f"📊 Budget Status for {budget_status['date']}:")
    print(f"   Daily budget: ${budget_status['budget']:.2f}")
    print(f"   Spent today: ${budget_status['spent']:.4f}")
    print(f"   Remaining: ${budget_status['remaining']:.4f}")
    print(f"   Usage: {budget_status['percentage_used']:.1f}%")
    print(f"   Requests: {budget_status['requests_today']}")
    print(f"   Tokens: {budget_status['tokens_today']:,}")

    if budget_status["over_budget"]:
        print("\n⚠️  WARNING: Over budget!")
    elif budget_status["percentage_used"] > 80:
        print("\n⚠️  WARNING: Approaching budget limit!")
    else:
        print("\n✅ Budget status: OK")


def example_7_cost_projection():
    """Example 7: Cost Projection"""
    print_section("Example 7: Cost Projection")

    projection = usage_tracker.get_cost_projection(days_ahead=30)

    print("📈 Cost Projection (based on last 7 days):\n")
    print(f"   Average daily cost: ${projection['avg_daily_cost']:.4f}")
    print(f"   30-day projection: ${projection['projected_cost']:.2f}")
    print(f"   Monthly estimate: ${projection['projected_monthly_cost']:.2f}")

    # Compare with targets
    monthly_target = 100.0
    if projection["projected_monthly_cost"] > monthly_target:
        overage = projection["projected_monthly_cost"] - monthly_target
        print(f"\n⚠️  Projected to exceed monthly target by ${overage:.2f}")
    else:
        savings = monthly_target - projection["projected_monthly_cost"]
        print(f"\n✅ Under monthly target by ${savings:.2f}")


def example_8_caching_simulation():
    """Example 8: Caching Benefits Simulation"""
    print_section("Example 8: Caching Benefits")

    # Simulate requests with some duplicates
    total_requests = 1000
    cache_hit_rate = 0.35  # 35% of requests are duplicate

    cost_per_request = 0.002  # GPT-3.5 average

    # Without caching
    cost_no_cache = total_requests * cost_per_request

    # With caching
    actual_api_calls = int(total_requests * (1 - cache_hit_rate))
    cost_with_cache = actual_api_calls * cost_per_request

    savings = cost_no_cache - cost_with_cache
    savings_pct = (savings / cost_no_cache) * 100

    print(f"📊 Scenario: {total_requests:,} requests\n")

    print("❌ Without caching:")
    print(f"   API calls: {total_requests:,}")
    print(f"   Cost: ${cost_no_cache:.2f}")

    print(f"\n✅ With caching ({cache_hit_rate * 100:.0f}% hit rate):")
    print(f"   API calls: {actual_api_calls:,}")
    print(f"   Cache hits: {total_requests - actual_api_calls:,}")
    print(f"   Cost: ${cost_with_cache:.2f}")

    print(f"\n💰 Savings: ${savings:.2f} ({savings_pct:.0f}%)")


def example_9_batch_vs_individual():
    """Example 9: Batch Processing Savings"""
    print_section("Example 9: Batch Processing")

    num_requests = 100
    tokens_per_request = 50

    # Individual requests
    individual_overhead = 10  # tokens overhead per request
    individual_total_tokens = num_requests * (tokens_per_request + individual_overhead)

    # Batched requests (10 per batch)
    batch_size = 10
    num_batches = num_requests // batch_size
    batch_overhead = 20  # tokens overhead per batch
    batch_total_tokens = (num_batches * batch_overhead) + (
        num_requests * tokens_per_request
    )

    savings_tokens = individual_total_tokens - batch_total_tokens
    savings_pct = (savings_tokens / individual_total_tokens) * 100

    print(f"📊 Processing {num_requests} small requests:\n")

    print("❌ Individual requests:")
    print(f"   Total tokens: {individual_total_tokens:,}")
    print(f"   Overhead: {num_requests * individual_overhead:,} tokens")

    print(f"\n✅ Batched (10 per batch):")
    print(f"   Batches: {num_batches}")
    print(f"   Total tokens: {batch_total_tokens:,}")
    print(f"   Overhead: {num_batches * batch_overhead:,} tokens")

    print(f"\n💰 Token savings: {savings_tokens:,} ({savings_pct:.0f}%)")


def example_10_optimization_summary():
    """Example 10: Optimization Strategy Summary"""
    print_section("Example 10: Optimization Strategy")

    baseline_cost = 1000.0  # Monthly baseline

    strategies = [
        {"name": "Use GPT-3.5 instead of GPT-4", "savings": 0.80},
        {"name": "Implement caching (35% hit rate)", "savings": 0.35},
        {"name": "Compress prompts (20% reduction)", "savings": 0.20},
        {"name": "Sliding window context", "savings": 0.30},
        {"name": "Batch processing", "savings": 0.15},
    ]

    print(f"📊 Monthly Baseline Cost: ${baseline_cost:.2f}\n")
    print("💡 Optimization Strategies:\n")

    current_cost = baseline_cost

    for i, strategy in enumerate(strategies, 1):
        strategy_savings = current_cost * strategy["savings"]
        current_cost -= strategy_savings

        print(f"{i}. {strategy['name']}")
        print(
            f"   Savings: ${strategy_savings:.2f} (-{strategy['savings'] * 100:.0f}%)"
        )
        print(f"   New cost: ${current_cost:.2f}")
        print()

    total_savings = baseline_cost - current_cost
    total_savings_pct = (total_savings / baseline_cost) * 100

    print("=" * 50)
    print(f"💰 Total Savings: ${total_savings:.2f} ({total_savings_pct:.0f}%)")
    print(f"📉 Final Monthly Cost: ${current_cost:.2f}")


def main():
    """Run all examples"""
    print("\n" + "=" * 70)
    print("  💰 COST OPTIMIZATION EXAMPLES")
    print("  Learn how to reduce LLM costs by 70-90%")
    print("=" * 70)

    try:
        example_1_track_usage()
        example_2_calculate_costs()
        example_3_prompt_compression()
        example_4_context_management()
        example_5_model_selection()
        example_6_budget_monitoring()
        example_7_cost_projection()
        example_8_caching_simulation()
        example_9_batch_vs_individual()
        example_10_optimization_summary()

        print_section("Summary")
        print("✅ Covered 10 cost optimization techniques:")
        print("   1. Track usage and monitor costs")
        print("   2. Calculate and compare model costs")
        print("   3. Compress prompts (remove filler)")
        print("   4. Manage context with sliding windows")
        print("   5. Route to appropriate models")
        print("   6. Monitor budgets with alerts")
        print("   7. Project future costs")
        print("   8. Cache responses (35-50% savings)")
        print("   9. Batch process requests")
        print("   10. Combine strategies for 70-90% savings")
        print()
        print("💡 Key Takeaway:")
        print("   Combining multiple techniques can reduce costs by 70-90%")
        print("   while maintaining quality and performance!")
        print()
        print("📚 Learn more: See wiki/Cost-Optimization.md")
        print("=" * 70)

    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
