from unittest.mock import MagicMock

import pytest

from syl_guardian_client import GuardianError
from syl_guardian_client.client import ReviewResult

RESULT = ReviewResult.from_response(
    {"verdict": "pass", "causal_effect": {"ate": 1.0, "ci": [0.5, 1.5]}}
)


def test_schema_requires_exactly_one_dataset():
    from pydantic import ValidationError

    from syl_guardian_client.integrations._schema import GuardianReviewInput

    with pytest.raises(ValidationError):
        GuardianReviewInput(cause="a", effect="b")
    GuardianReviewInput(cause="a", effect="b", dataset_id="x")


def test_langchain_tool(monkeypatch):
    pytest.importorskip("langchain_core")
    from syl_guardian_client.integrations.langchain import guardian_review_tool

    client = MagicMock()
    client.review.return_value = RESULT
    tool = guardian_review_tool(client)

    out = tool.invoke({"cause": "spend", "effect": "value", "dataset_id": "abc"})
    assert "Verdict: PASS" in out
    client.review.assert_called_once()

    client.review.side_effect = GuardianError(429, "quota_exceeded", "Quota used.")
    out = tool.invoke({"cause": "spend", "effect": "value", "dataset_id": "abc"})
    assert "quota_exceeded" in out


def test_crewai_tool(monkeypatch):
    pytest.importorskip("crewai")
    from syl_guardian_client.integrations import crewai as mod

    client = MagicMock()
    client.review.return_value = RESULT
    monkeypatch.setattr(mod, "GuardianClient", lambda **kw: client)

    out = mod.SylGuardianTool()._run(cause="spend", effect="value", dataset_id="abc")

    assert "Verdict: PASS" in out
    assert client.review.call_args.kwargs["dataset_id"] == "abc"
