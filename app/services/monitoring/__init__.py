"""
Monitoring Module

Usage tracking and cost monitoring for LLM applications.

Usage:
    from app.services.monitoring import usage_tracker

    # Log a request
    usage_tracker.log_request(
        model="gpt-3.5-turbo",
        input_tokens=500,
        output_tokens=200,
        cost=0.0014
    )

    # Get statistics
    stats = usage_tracker.get_stats()
    print(f"Total cost: ${stats['total_cost']:.2f}")
"""

from app.services.monitoring.usage_tracker import UsageTracker, usage_tracker

__all__ = ["UsageTracker", "usage_tracker"]
