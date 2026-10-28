"""Thin HTTP client for the hosted Syl Guardian API.

All estimation happens server-side; this module only builds requests and parses
responses. Numbers in a result are computed deterministically by statistical
code on the server, never by an LLM.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import requests

from syl_guardian_client import __version__

DEFAULT_BASE_URL = "https://guardian.sylai.io"
API_KEY_ENV = "SYL_GUARDIAN_API_KEY"

_OPTIONAL_FIELDS = (
    "known_confounders",
    "context",
    "units",
    "step",
    "time_col",
    "adstock",
    "test_design",
    "cate",
    "sensitivity",
    "seed",
    "config",
    "estimator",
)


class GuardianError(Exception):
    """An error response from the Guardian API (auth, quota, validation, ...)."""

    def __init__(self, status_code: int, error: str, detail: str) -> None:
        super().__init__(f"Syl Guardian error ({status_code}) {error}: {detail}")
        self.status_code = status_code
        self.error = error
        self.detail = detail


@dataclass
class ReviewResult:
    """A parsed ``guardian_review`` response. The full body is kept in ``raw``."""

    verdict: str
    confounded: bool | None
    correlational_effect: float | None
    ate: float | None
    ci: list[float] | None
    confidence: float | None
    top_confounders: list[str] = field(default_factory=list)
    recommended_action: str | None = None
    interpretation: str | None = None
    method: str | None = None
    n: int | None = None
    raw: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_response(cls, body: dict[str, Any]) -> "ReviewResult":
        causal = body.get("causal_effect") or {}
        return cls(
            verdict=str(body.get("verdict", "")).lower(),
            confounded=body.get("confounded"),
            correlational_effect=body.get("correlational_effect"),
            ate=causal.get("ate"),
            ci=causal.get("ci"),
            confidence=body.get("confidence"),
            top_confounders=list(body.get("top_confounders") or []),
            recommended_action=body.get("recommended_action"),
            interpretation=body.get("interpretation"),
            method=body.get("method"),
            n=body.get("n"),
            raw=body,
        )

    @property
    def passed(self) -> bool:
        return self.verdict == "pass"

    @property
    def caution(self) -> bool:
        return self.verdict == "caution"

    @property
    def held(self) -> bool:
        return self.verdict == "hold"

    def summary(self) -> str:
        """Plain-text summary suitable for returning to an agent."""
        return "\n".join(
            [
                f"Verdict: {self.verdict.upper()}",
                f"Confounded: {self.confounded}",
                f"Naive (correlational) effect: {self.correlational_effect}",
                f"Adjusted causal effect: {self.ate} (CI: {self.ci})",
                f"Confidence: {self.confidence}",
                f"Top confounders: {', '.join(self.top_confounders) or 'none'}",
                f"Recommended action: {self.recommended_action}",
            ]
        )


def _records(data: Any) -> Any:
    """Accept a list of row dicts or a pandas-like DataFrame without importing pandas."""
    if hasattr(data, "to_dict") and hasattr(data, "columns"):
        return data.to_dict(orient="records")
    return data


class GuardianClient:
    """Client for ``https://guardian.sylai.io``.

    Args:
        api_key: Guardian API key. Defaults to the ``SYL_GUARDIAN_API_KEY``
            environment variable. Get a trial key at https://sylai.io/guardian.
        base_url: API origin, for self-hosted or test deployments.
        timeout: Request timeout in seconds.
        session: Optional ``requests.Session`` to reuse.
    """

    def __init__(
        self,
        api_key: str | None = None,
        *,
        base_url: str = DEFAULT_BASE_URL,
        timeout: float = 120,
        session: requests.Session | None = None,
    ) -> None:
        self.api_key = api_key or os.environ.get(API_KEY_ENV)
        if not self.api_key:
            raise ValueError(
                f"No API key: pass api_key= or set {API_KEY_ENV} "
                "(trial keys: https://sylai.io/guardian)"
            )
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self._session = session or requests.Session()

    def _headers(self) -> dict[str, str]:
        return {
            "X-API-Key": self.api_key or "",
            "User-Agent": f"syl-guardian-client/{__version__}",
        }

    @staticmethod
    def _raise_for_error(response: requests.Response) -> None:
        if response.ok:
            return
        try:
            err = response.json()
            error, detail = str(err.get("error")), str(err.get("detail"))
        except ValueError:
            error, detail = "http_error", response.text
        raise GuardianError(response.status_code, error, detail)

    def review(
        self,
        cause: str,
        effect: str,
        *,
        data: Any = None,
        csv: str | None = None,
        dataset_id: str | None = None,
        **options: Any,
    ) -> ReviewResult:
        """Check whether ``cause`` really drives ``effect`` in the dataset.

        Provide exactly one of ``data`` (list of row dicts or a DataFrame),
        ``csv`` (CSV text) or ``dataset_id`` (from :meth:`upload_dataset`).
        Optional keyword arguments are passed through: ``known_confounders``,
        ``context``, ``units``, ``step``, ``time_col``, ``adstock``,
        ``test_design``, ``cate``, ``sensitivity``, ``seed``, ``config`` and
        ``estimator``. Send only the cause, effect, ``time_col`` and confounder
        columns; downstream columns (e.g. ``roas``, ``clicks``) bias the estimate.

        Raises:
            ValueError: if the dataset arguments or option names are invalid.
            GuardianError: on an API error response.
        """
        sources = {"data": data, "csv": csv, "dataset_id": dataset_id}
        given = {k: v for k, v in sources.items() if v is not None}
        if len(given) != 1:
            raise ValueError("Provide exactly one of data, csv or dataset_id.")
        unknown = set(options) - set(_OPTIONAL_FIELDS)
        if unknown:
            raise ValueError(f"Unknown review option(s): {', '.join(sorted(unknown))}")

        payload: dict[str, Any] = {"claim": {"cause": cause, "effect": effect}}
        (key, value), = given.items()
        payload[key] = _records(value) if key == "data" else value
        payload.update({k: v for k, v in options.items() if v is not None})

        response = self._session.post(
            f"{self.base_url}/v1/guardrail/review",
            json=payload,
            headers=self._headers(),
            timeout=self.timeout,
        )
        self._raise_for_error(response)
        return ReviewResult.from_response(response.json())

    def upload_dataset(self, path: str | Path) -> str:
        """Upload a CSV for a large dataset and return its single-use ``dataset_id``.

        Uploads don't count toward review quota. The dataset is deleted when it
        is reviewed (or after 30 minutes), so upload again to re-run a review.
        """
        body = Path(path).read_bytes()
        response = self._session.post(
            f"{self.base_url}/v1/uploads", headers=self._headers(), timeout=self.timeout
        )
        self._raise_for_error(response)
        ticket = response.json()
        upload_url = ticket["upload_url"]
        if upload_url.startswith("/"):
            upload_url = f"{self.base_url}{upload_url}"
        # The one-time token is in the URL, so the API key is not sent here.
        put = self._session.put(
            upload_url,
            data=body,
            headers={"User-Agent": self._headers()["User-Agent"]},
            timeout=self.timeout,
        )
        self._raise_for_error(put)
        return str(ticket["dataset_id"])

    def review_file(
        self, cause: str, effect: str, path: str | Path, **options: Any
    ) -> ReviewResult:
        """Upload a CSV file and review it in one call."""
        return self.review(cause, effect, dataset_id=self.upload_dataset(path), **options)
