"""CrewAI tool: ``pip install 'syl-guardian-client[crewai]'``."""

from __future__ import annotations

from typing import Any

from crewai.tools import BaseTool, EnvVar
from pydantic import BaseModel, Field

from syl_guardian_client.client import API_KEY_ENV, DEFAULT_BASE_URL, GuardianClient
from syl_guardian_client.integrations._schema import (
    TOOL_DESCRIPTION,
    GuardianReviewInput,
)


class SylGuardianTool(BaseTool):
    """Checks that a claimed effect is causal before an agent acts on it."""

    name: str = "Syl Guardian causal check"
    description: str = TOOL_DESCRIPTION
    args_schema: type[BaseModel] = GuardianReviewInput
    base_url: str = DEFAULT_BASE_URL
    timeout: int = 120
    env_vars: list[EnvVar] = Field(
        default_factory=lambda: [
            EnvVar(
                name=API_KEY_ENV,
                description="API key for Syl Guardian (https://sylai.io/guardian)",
                required=True,
            )
        ]
    )

    def _run(self, **kwargs: Any) -> str:
        client = GuardianClient(base_url=self.base_url, timeout=self.timeout)
        cause, effect = kwargs.pop("cause"), kwargs.pop("effect")
        dataset = {k: kwargs.pop(k, None) for k in ("data", "csv", "dataset_id")}
        return client.review(cause, effect, **dataset, **kwargs).summary()
