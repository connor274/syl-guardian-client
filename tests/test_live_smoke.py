"""Live smoke test against the hosted API. Uses a little trial quota.

Run with: SYL_GUARDIAN_LIVE=1 SYL_GUARDIAN_API_KEY=... pytest tests/test_live_smoke.py
"""

import os
import random

import pytest

from syl_guardian_client import GuardianClient

pytestmark = pytest.mark.skipif(
    not (os.environ.get("SYL_GUARDIAN_LIVE") and os.environ.get("SYL_GUARDIAN_API_KEY")),
    reason="set SYL_GUARDIAN_LIVE=1 and SYL_GUARDIAN_API_KEY to run",
)


def _confounded_rows(n=120, seed=7):
    """Brand search share drives both spend and revenue; spend itself does nothing."""
    rng = random.Random(seed)
    rows = []
    for week in range(n):
        brand = rng.uniform(0.1, 0.9)
        rows.append(
            {
                "week": week,
                "brand_query_click_pct": brand,
                "ad_spend": 1000 + 4000 * brand + rng.gauss(0, 150),
                "conversion_value": 5000 + 20000 * brand + rng.gauss(0, 600),
            }
        )
    return rows


def test_inline_review():
    result = GuardianClient().review(
        "ad_spend",
        "conversion_value",
        data=_confounded_rows(),
        time_col="week",
        known_confounders=["brand_query_click_pct"],
        step=500,
    )
    assert result.verdict in {"pass", "caution", "hold"}
    assert result.correlational_effect is not None
    assert result.ate is not None and result.ci and len(result.ci) == 2
    assert result.raw["stamp"]["service"]["service"]
    print(result.summary())


def test_upload_review(tmp_path):
    rows = _confounded_rows()
    path = tmp_path / "campaign.csv"
    cols = ["week", "brand_query_click_pct", "ad_spend", "conversion_value"]
    path.write_text(
        ",".join(cols) + "\n" + "\n".join(",".join(str(r[c]) for c in cols) for r in rows)
    )
    result = GuardianClient().review_file(
        "ad_spend",
        "conversion_value",
        path,
        time_col="week",
        known_confounders=["brand_query_click_pct"],
    )
    assert result.verdict in {"pass", "caution", "hold"}
    assert result.n == len(rows)
