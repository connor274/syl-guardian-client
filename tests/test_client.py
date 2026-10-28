from unittest.mock import MagicMock

import pytest
import requests

from syl_guardian_client import GuardianClient, GuardianError

HOLD = {
    "verdict": "hold",
    "confounded": True,
    "correlational_effect": 3.2,
    "causal_effect": {"ate": 0.4, "ci": [-0.3, 1.1]},
    "confidence": 0.78,
    "top_confounders": ["brand_query_click_pct"],
    "recommended_action": "Do not act on this claim.",
}
ROWS = [{"spend": 1, "value": 2}, {"spend": 2, "value": 3}]


def _resp(status=200, body=None):
    r = MagicMock(spec=requests.Response)
    r.status_code = status
    r.ok = 200 <= status < 400
    r.text = str(body)
    r.json.return_value = body if body is not None else {}
    return r


def _client(session):
    return GuardianClient("test-key", session=session)


def test_requires_api_key(monkeypatch):
    monkeypatch.delenv("SYL_GUARDIAN_API_KEY", raising=False)
    with pytest.raises(ValueError, match="SYL_GUARDIAN_API_KEY"):
        GuardianClient()


def test_api_key_from_env(monkeypatch):
    monkeypatch.setenv("SYL_GUARDIAN_API_KEY", "env-key")
    assert GuardianClient().api_key == "env-key"


def test_review_builds_request_and_parses_result():
    session = MagicMock()
    session.post.return_value = _resp(200, HOLD)

    result = _client(session).review(
        "spend", "value", data=ROWS, known_confounders=["brand_query_click_pct"], step=500
    )

    (url,), kwargs = session.post.call_args
    assert url == "https://guardian.sylai.io/v1/guardrail/review"
    assert kwargs["headers"]["X-API-Key"] == "test-key"
    assert kwargs["headers"]["User-Agent"].startswith("syl-guardian-client/")
    assert kwargs["json"] == {
        "claim": {"cause": "spend", "effect": "value"},
        "data": ROWS,
        "known_confounders": ["brand_query_click_pct"],
        "step": 500,
    }
    assert result.held and not result.passed and not result.caution
    assert result.ate == 0.4 and result.ci == [-0.3, 1.1]
    assert "Verdict: HOLD" in result.summary()
    assert result.raw is not None


def test_review_accepts_dataframe_like():
    class Frame:
        columns = ["spend", "value"]

        def to_dict(self, orient):
            assert orient == "records"
            return ROWS

    session = MagicMock()
    session.post.return_value = _resp(200, HOLD)
    _client(session).review("spend", "value", data=Frame())
    assert session.post.call_args.kwargs["json"]["data"] == ROWS


@pytest.mark.parametrize(
    "kwargs",
    [{}, {"data": ROWS, "csv": "a,b"}, {"data": ROWS, "bogus": 1}],
)
def test_review_validates_arguments(kwargs):
    with pytest.raises(ValueError):
        _client(MagicMock()).review("spend", "value", **kwargs)


def test_api_error_raises_guardian_error():
    session = MagicMock()
    session.post.return_value = _resp(
        403, {"error": "trial_expired", "detail": "This trial key has expired."}
    )
    with pytest.raises(GuardianError) as exc:
        _client(session).review("spend", "value", data=ROWS)
    assert exc.value.status_code == 403 and exc.value.error == "trial_expired"


def test_upload_dataset_flow_does_not_send_key_with_put(tmp_path):
    csv = tmp_path / "d.csv"
    csv.write_text("spend,value\n1,2\n")
    session = MagicMock()
    session.post.return_value = _resp(
        200, {"upload_url": "https://u.example/v1/uploads/abc?token=t", "dataset_id": "abc"}
    )
    session.put.return_value = _resp(200, {"dataset_id": "abc"})

    assert _client(session).upload_dataset(csv) == "abc"

    put_args, put_kwargs = session.put.call_args
    assert put_args[0] == "https://u.example/v1/uploads/abc?token=t"
    assert put_kwargs["data"] == b"spend,value\n1,2\n"
    assert "X-API-Key" not in put_kwargs["headers"]


def test_review_file_reviews_by_dataset_id(tmp_path):
    csv = tmp_path / "d.csv"
    csv.write_text("spend,value\n1,2\n")
    session = MagicMock()
    session.post.side_effect = [
        _resp(200, {"upload_url": "/v1/uploads/abc?token=t", "dataset_id": "abc"}),
        _resp(200, HOLD),
    ]
    session.put.return_value = _resp(200, {})

    _client(session).review_file("spend", "value", csv)

    assert session.put.call_args.args[0] == "https://guardian.sylai.io/v1/uploads/abc?token=t"
    assert session.post.call_args.kwargs["json"]["dataset_id"] == "abc"
