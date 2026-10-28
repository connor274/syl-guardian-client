"""LangChain tool: ``pip install 'syl-guardian-client[langchain]'``."""

from __future__ import annotations

from typing import Any

from langchain_core.tools import StructuredTool, ToolException

from syl_guardian_client.client import GuardianClient, GuardianError
from syl_guardian_client.integrations._schema import (
    TOOL_DESCRIPTION,
    TOOL_NAME,
    GuardianReviewInput,
)


def guardian_review_tool(client: GuardianClient | None = None) -> StructuredTool:
    """Build a LangChain tool that runs a Guardian causal check.

    Args:
        client: Optional preconfigured client. Defaults to one built from the
            ``SYL_GUARDIAN_API_KEY`` environment variable.
    """
    guardian = client or GuardianClient()

    def _review(cause: str, effect: str, **kwargs: Any) -> str:
        dataset = {k: kwargs.pop(k, None) for k in ("data", "csv", "dataset_id")}
        try:
            return guardian.review(cause, effect, **dataset, **kwargs).summary()
        except GuardianError as exc:
            raise ToolException(str(exc)) from exc

    return StructuredTool.from_function(
        func=_review,
        name=TOOL_NAME,
        description=TOOL_DESCRIPTION,
        args_schema=GuardianReviewInput,
        handle_tool_error=True,
    )
