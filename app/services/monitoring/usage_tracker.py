"""
Usage Tracking Service - Monitor LLM API usage and costs

This service tracks all LLM API calls, calculates costs, and provides
analytics for cost optimization and budget management.

Usage:
    from app.services.monitoring.usage_tracker import usage_tracker

    # Log a request
    usage_tracker.log_request(
        model="gpt-3.5-turbo",
        input_tokens=500,
        output_tokens=200,
        cost=0.0014,
        user_id="user123",
        endpoint="/api/chat"
    )

    # Get statistics
    stats = usage_tracker.get_stats()
    print(f"Total cost: ${stats['total_cost']:.2f}")
"""

import logging
from collections import defaultdict
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class UsageTracker:
    """Track and analyze LLM usage and costs"""

    def __init__(self):
        """Initialize usage tracker"""
        self.usage_log: List[Dict[str, Any]] = []
        self.daily_totals: Dict[str, Dict[str, Any]] = defaultdict(
            lambda: {
                "requests": 0,
                "input_tokens": 0,
                "output_tokens": 0,
                "total_tokens": 0,
                "cost": 0.0,
            }
        )
        self.model_totals: Dict[str, Dict[str, Any]] = defaultdict(
            lambda: {
                "requests": 0,
                "input_tokens": 0,
                "output_tokens": 0,
                "total_tokens": 0,
                "cost": 0.0,
            }
        )
        self.user_totals: Dict[str, Dict[str, Any]] = defaultdict(
            lambda: {"requests": 0, "cost": 0.0, "tokens": 0}
        )

    def log_request(
        self,
        model: str,
        input_tokens: int,
        output_tokens: int,
        cost: float,
        user_id: Optional[str] = None,
        endpoint: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """
        Log an API request

        Args:
            model: Model name (e.g., "gpt-3.5-turbo")
            input_tokens: Number of input tokens
            output_tokens: Number of output tokens
            cost: Cost in USD
            user_id: Optional user identifier
            endpoint: Optional API endpoint
            metadata: Optional additional metadata
        """
        total_tokens = input_tokens + output_tokens

        entry = {
            "timestamp": datetime.utcnow(),
            "model": model,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": total_tokens,
            "cost": cost,
            "user_id": user_id,
            "endpoint": endpoint,
            "metadata": metadata or {},
        }

        # Add to log
        self.usage_log.append(entry)

        # Update daily totals
        date_key = entry["timestamp"].date().isoformat()
        self.daily_totals[date_key]["requests"] += 1
        self.daily_totals[date_key]["input_tokens"] += input_tokens
        self.daily_totals[date_key]["output_tokens"] += output_tokens
        self.daily_totals[date_key]["total_tokens"] += total_tokens
        self.daily_totals[date_key]["cost"] += cost

        # Update model totals
        self.model_totals[model]["requests"] += 1
        self.model_totals[model]["input_tokens"] += input_tokens
        self.model_totals[model]["output_tokens"] += output_tokens
        self.model_totals[model]["total_tokens"] += total_tokens
        self.model_totals[model]["cost"] += cost

        # Update user totals
        if user_id:
            self.user_totals[user_id]["requests"] += 1
            self.user_totals[user_id]["cost"] += cost
            self.user_totals[user_id]["tokens"] += total_tokens

        logger.info(
            f"LLM Request: {model} | "
            f"Tokens: {input_tokens}→{output_tokens} | "
            f"Cost: ${cost:.6f} | "
            f"User: {user_id or 'anonymous'}"
        )

    def get_stats(
        self,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        user_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Get usage statistics for date range

        Args:
            start_date: Start date (default: 30 days ago)
            end_date: End date (default: now)
            user_id: Optional user filter

        Returns:
            Dictionary with usage statistics
        """
        # Set defaults
        if start_date is None:
            start_date = datetime.utcnow() - timedelta(days=30)
        if end_date is None:
            end_date = datetime.utcnow()

        # Filter by date range and user
        filtered = [
            entry
            for entry in self.usage_log
            if start_date <= entry["timestamp"] <= end_date
            and (user_id is None or entry["user_id"] == user_id)
        ]

        if not filtered:
            return {
                "period": {
                    "start": start_date.isoformat(),
                    "end": end_date.isoformat(),
                },
                "total_requests": 0,
                "total_cost": 0.0,
                "total_tokens": 0,
                "by_model": {},
                "by_user": {},
                "by_day": {},
            }

        # Calculate totals
        total_cost = sum(e["cost"] for e in filtered)
        total_tokens = sum(e["total_tokens"] for e in filtered)
        total_input_tokens = sum(e["input_tokens"] for e in filtered)
        total_output_tokens = sum(e["output_tokens"] for e in filtered)

        # Model breakdown
        model_stats = defaultdict(lambda: {"requests": 0, "cost": 0.0, "tokens": 0})
        for entry in filtered:
            model = entry["model"]
            model_stats[model]["requests"] += 1
            model_stats[model]["cost"] += entry["cost"]
            model_stats[model]["tokens"] += entry["total_tokens"]

        # User breakdown
        user_stats = defaultdict(lambda: {"requests": 0, "cost": 0.0, "tokens": 0})
        for entry in filtered:
            if entry["user_id"]:
                uid = entry["user_id"]
                user_stats[uid]["requests"] += 1
                user_stats[uid]["cost"] += entry["cost"]
                user_stats[uid]["tokens"] += entry["total_tokens"]

        # Daily breakdown
        daily_stats = defaultdict(lambda: {"requests": 0, "cost": 0.0, "tokens": 0})
        for entry in filtered:
            date_key = entry["timestamp"].date().isoformat()
            daily_stats[date_key]["requests"] += 1
            daily_stats[date_key]["cost"] += entry["cost"]
            daily_stats[date_key]["tokens"] += entry["total_tokens"]

        # Endpoint breakdown
        endpoint_stats = defaultdict(lambda: {"requests": 0, "cost": 0.0, "tokens": 0})
        for entry in filtered:
            if entry["endpoint"]:
                endpoint = entry["endpoint"]
                endpoint_stats[endpoint]["requests"] += 1
                endpoint_stats[endpoint]["cost"] += entry["cost"]
                endpoint_stats[endpoint]["tokens"] += entry["total_tokens"]

        return {
            "period": {
                "start": start_date.isoformat(),
                "end": end_date.isoformat(),
                "days": (end_date - start_date).days,
            },
            "total_requests": len(filtered),
            "total_cost": round(total_cost, 6),
            "total_tokens": total_tokens,
            "input_tokens": total_input_tokens,
            "output_tokens": total_output_tokens,
            "avg_cost_per_request": round(total_cost / len(filtered), 6),
            "avg_tokens_per_request": round(total_tokens / len(filtered), 2),
            "by_model": dict(model_stats),
            "by_user": dict(user_stats),
            "by_day": dict(daily_stats),
            "by_endpoint": dict(endpoint_stats),
        }

    def get_top_users(self, limit: int = 10, by: str = "cost") -> List[Dict]:
        """
        Get top users by cost or usage

        Args:
            limit: Number of users to return
            by: Sort by 'cost', 'requests', or 'tokens'

        Returns:
            List of top users with their statistics
        """
        if by not in ["cost", "requests", "tokens"]:
            raise ValueError("by must be 'cost', 'requests', or 'tokens'")

        sorted_users = sorted(
            self.user_totals.items(),
            key=lambda x: x[1][by],
            reverse=True,
        )

        return [
            {
                "user_id": user,
                "requests": stats["requests"],
                "cost": round(stats["cost"], 6),
                "tokens": stats["tokens"],
            }
            for user, stats in sorted_users[:limit]
        ]

    def get_top_endpoints(self, limit: int = 10) -> List[Dict]:
        """Get top endpoints by cost"""
        endpoint_stats = defaultdict(lambda: {"requests": 0, "cost": 0.0})

        for entry in self.usage_log:
            if entry["endpoint"]:
                endpoint_stats[entry["endpoint"]]["requests"] += 1
                endpoint_stats[entry["endpoint"]]["cost"] += entry["cost"]

        sorted_endpoints = sorted(
            endpoint_stats.items(),
            key=lambda x: x[1]["cost"],
            reverse=True,
        )

        return [
            {
                "endpoint": endpoint,
                "requests": stats["requests"],
                "cost": round(stats["cost"], 6),
            }
            for endpoint, stats in sorted_endpoints[:limit]
        ]

    def check_budget(self, daily_budget: float) -> Dict[str, Any]:
        """
        Check if current daily usage exceeds budget

        Args:
            daily_budget: Daily budget limit in USD

        Returns:
            Dictionary with budget status
        """
        today = datetime.utcnow().date().isoformat()
        today_usage = self.daily_totals[today]

        spent = today_usage["cost"]
        remaining = daily_budget - spent
        percentage_used = (spent / daily_budget) * 100 if daily_budget > 0 else 0

        return {
            "date": today,
            "budget": daily_budget,
            "spent": round(spent, 6),
            "remaining": round(remaining, 6),
            "percentage_used": round(percentage_used, 2),
            "over_budget": spent > daily_budget,
            "requests_today": today_usage["requests"],
            "tokens_today": today_usage["total_tokens"],
        }

    def get_cost_projection(self, days_ahead: int = 30) -> Dict[str, Any]:
        """
        Project future costs based on recent usage

        Args:
            days_ahead: Number of days to project

        Returns:
            Cost projection
        """
        # Get last 7 days average
        today = datetime.utcnow().date()
        last_7_days = [(today - timedelta(days=i)).isoformat() for i in range(7)]

        total_cost_7days = sum(
            self.daily_totals[day]["cost"]
            for day in last_7_days
            if day in self.daily_totals
        )

        avg_daily_cost = total_cost_7days / 7
        projected_cost = avg_daily_cost * days_ahead

        return {
            "based_on_days": 7,
            "avg_daily_cost": round(avg_daily_cost, 6),
            "projection_days": days_ahead,
            "projected_cost": round(projected_cost, 2),
            "projected_monthly_cost": round(avg_daily_cost * 30, 2),
        }

    def clear_old_data(self, days_to_keep: int = 90):
        """
        Clear data older than specified days

        Args:
            days_to_keep: Number of days of data to keep
        """
        cutoff_date = datetime.utcnow() - timedelta(days=days_to_keep)

        # Filter usage log
        self.usage_log = [
            entry for entry in self.usage_log if entry["timestamp"] > cutoff_date
        ]

        # Clear old daily totals
        cutoff_date_str = cutoff_date.date().isoformat()
        keys_to_remove = [
            key for key in self.daily_totals.keys() if key < cutoff_date_str
        ]
        for key in keys_to_remove:
            del self.daily_totals[key]

        logger.info(f"Cleared usage data older than {days_to_keep} days")

    def export_to_dict(self) -> Dict[str, Any]:
        """Export all data to dictionary for persistence"""
        return {
            "usage_log": [
                {
                    **entry,
                    "timestamp": entry["timestamp"].isoformat(),
                }
                for entry in self.usage_log
            ],
            "daily_totals": dict(self.daily_totals),
            "model_totals": dict(self.model_totals),
            "user_totals": dict(self.user_totals),
        }

    def import_from_dict(self, data: Dict[str, Any]):
        """Import data from dictionary"""
        self.usage_log = [
            {
                **entry,
                "timestamp": datetime.fromisoformat(entry["timestamp"]),
            }
            for entry in data.get("usage_log", [])
        ]
        self.daily_totals = defaultdict(
            lambda: {
                "requests": 0,
                "input_tokens": 0,
                "output_tokens": 0,
                "total_tokens": 0,
                "cost": 0.0,
            },
            data.get("daily_totals", {}),
        )
        self.model_totals = defaultdict(
            lambda: {
                "requests": 0,
                "input_tokens": 0,
                "output_tokens": 0,
                "total_tokens": 0,
                "cost": 0.0,
            },
            data.get("model_totals", {}),
        )
        self.user_totals = defaultdict(
            lambda: {"requests": 0, "cost": 0.0, "tokens": 0},
            data.get("user_totals", {}),
        )


# Global instance
usage_tracker = UsageTracker()
