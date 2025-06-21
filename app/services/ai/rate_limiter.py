# app/services/ai/rate_limiter.py
import asyncio
import time
import logging
from typing import Dict, Any, Optional
from collections import deque
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


class RateLimiter:
    """Simple rate limiter for API calls"""

    def __init__(self, calls_per_minute: int = 60, calls_per_hour: int = 1000):
        self.calls_per_minute = calls_per_minute
        self.calls_per_hour = calls_per_hour

        # Track call timestamps
        self.minute_calls = deque()  # Calls in the last minute
        self.hour_calls = deque()  # Calls in the last hour

        self.last_call_time = 0
        self.min_delay = 1.0  # Minimum delay between calls (seconds)

    async def wait_if_needed(self):
        """Wait if rate limit would be exceeded"""
        current_time = time.time()

        # Clean old entries
        self._clean_old_calls(current_time)

        # Check if we need to wait
        wait_time = self._calculate_wait_time(current_time)

        if wait_time > 0:
            logger.info(f"Rate limiting: waiting {wait_time:.2f} seconds")
            await asyncio.sleep(wait_time)
            current_time = time.time()

        # Record this call
        self.minute_calls.append(current_time)
        self.hour_calls.append(current_time)
        self.last_call_time = current_time

    def _clean_old_calls(self, current_time: float):
        """Remove old call timestamps"""
        minute_cutoff = current_time - 60
        hour_cutoff = current_time - 3600

        # Clean minute calls
        while self.minute_calls and self.minute_calls[0] < minute_cutoff:
            self.minute_calls.popleft()

        # Clean hour calls
        while self.hour_calls and self.hour_calls[0] < hour_cutoff:
            self.hour_calls.popleft()

    def _calculate_wait_time(self, current_time: float) -> float:
        """Calculate how long to wait before making the next call"""
        wait_times = []

        # Check minimum delay between calls
        time_since_last = current_time - self.last_call_time
        if time_since_last < self.min_delay:
            wait_times.append(self.min_delay - time_since_last)

        # Check minute limit
        if len(self.minute_calls) >= self.calls_per_minute:
            # Need to wait until the oldest call in the minute expires
            oldest_minute_call = self.minute_calls[0]
            wait_until = oldest_minute_call + 60
            wait_times.append(wait_until - current_time)

        # Check hour limit
        if len(self.hour_calls) >= self.calls_per_hour:
            # Need to wait until the oldest call in the hour expires
            oldest_hour_call = self.hour_calls[0]
            wait_until = oldest_hour_call + 3600
            wait_times.append(wait_until - current_time)

        return max(0, max(wait_times)) if wait_times else 0

    def get_stats(self) -> Dict[str, Any]:
        """Get rate limiter statistics"""
        current_time = time.time()
        self._clean_old_calls(current_time)

        return {
            "calls_last_minute": len(self.minute_calls),
            "calls_last_hour": len(self.hour_calls),
            "minute_limit": self.calls_per_minute,
            "hour_limit": self.calls_per_hour,
            "time_since_last_call": current_time - self.last_call_time,
            "min_delay": self.min_delay
        }


class AdaptiveRateLimiter(RateLimiter):
    """Rate limiter that adapts based on API responses"""

    def __init__(self, calls_per_minute: int = 60, calls_per_hour: int = 1000):
        super().__init__(calls_per_minute, calls_per_hour)
        self.consecutive_errors = 0
        self.backoff_multiplier = 1.0
        self.max_backoff = 30.0
        self.success_count = 0

    async def wait_if_needed(self):
        """Wait with adaptive backoff"""
        # Apply backoff multiplier
        original_delay = self.min_delay
        self.min_delay = min(self.min_delay * self.backoff_multiplier, self.max_backoff)

        await super().wait_if_needed()

        # Restore original delay
        self.min_delay = original_delay

    def record_success(self):
        """Record a successful API call"""
        self.consecutive_errors = 0
        self.success_count += 1

        # Gradually reduce backoff on success
        if self.success_count > 5:
            self.backoff_multiplier = max(1.0, self.backoff_multiplier * 0.9)
            self.success_count = 0

    def record_error(self, error_type: str = "unknown"):
        """Record an API error"""
        self.consecutive_errors += 1
        self.success_count = 0

        # Increase backoff on errors
        if error_type in ["rate_limit", "timeout", "server_error"]:
            self.backoff_multiplier = min(5.0, self.backoff_multiplier * 1.5)

        logger.warning(
            f"API error recorded ({error_type}). "
            f"Consecutive errors: {self.consecutive_errors}, "
            f"Backoff multiplier: {self.backoff_multiplier:.2f}"
        )

    def get_stats(self) -> Dict[str, Any]:
        """Get enhanced statistics"""
        stats = super().get_stats()
        stats.update({
            "consecutive_errors": self.consecutive_errors,
            "backoff_multiplier": self.backoff_multiplier,
            "effective_delay": self.min_delay * self.backoff_multiplier
        })
        return stats


class ProviderRateLimiter:
    """Manages rate limiters for different AI providers"""

    def __init__(self):
        self.limiters: Dict[str, RateLimiter] = {}
        self._initialize_default_limiters()

    def _initialize_default_limiters(self):
        """Initialize rate limiters for different providers"""
        # HuggingFace free tier is quite restrictive
        self.limiters["huggingface"] = AdaptiveRateLimiter(
            calls_per_minute=20,
            calls_per_hour=100
        )

        # OpenAI has higher limits (adjust based on your tier)
        self.limiters["openai"] = AdaptiveRateLimiter(
            calls_per_minute=500,
            calls_per_hour=10000
        )

        # Anthropic limits (adjust based on your tier)
        self.limiters["anthropic"] = AdaptiveRateLimiter(
            calls_per_minute=100,
            calls_per_hour=5000
        )

    def get_limiter(self, provider: str) -> RateLimiter:
        """Get rate limiter for a specific provider"""
        if provider not in self.limiters:
            # Default limiter for unknown providers
            self.limiters[provider] = AdaptiveRateLimiter(
                calls_per_minute=30,
                calls_per_hour=500
            )

        return self.limiters[provider]

    async def wait_if_needed(self, provider: str):
        """Wait if needed for the specified provider"""
        limiter = self.get_limiter(provider)
        await limiter.wait_if_needed()

    def record_success(self, provider: str):
        """Record successful call for provider"""
        limiter = self.get_limiter(provider)
        if isinstance(limiter, AdaptiveRateLimiter):
            limiter.record_success()

    def record_error(self, provider: str, error_type: str = "unknown"):
        """Record error for provider"""
        limiter = self.get_limiter(provider)
        if isinstance(limiter, AdaptiveRateLimiter):
            limiter.record_error(error_type)

    def get_all_stats(self) -> Dict[str, Dict[str, Any]]:
        """Get stats for all providers"""
        return {
            provider: limiter.get_stats()
            for provider, limiter in self.limiters.items()
        }


# Global instance
provider_rate_limiter = ProviderRateLimiter()