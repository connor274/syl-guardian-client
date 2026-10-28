"""Python client for Syl Guardian, a causal guardrail for AI agent decisions."""

__version__ = "0.1.0"

from syl_guardian_client.client import (  # noqa: E402
    DEFAULT_BASE_URL,
    GuardianClient,
    GuardianError,
    ReviewResult,
)

__all__ = [
    "DEFAULT_BASE_URL",
    "GuardianClient",
    "GuardianError",
    "ReviewResult",
    "__version__",
]
