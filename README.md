# syl-guardian-client

Python client for [Syl Guardian](https://sylai.io/guardian), a causal guardrail for AI agent decisions.

Before an agent acts on a claim such as "this campaign's ROAS is 4x, scale it up", Guardian checks whether the effect is causal or merely confounded (for example, branded search that would have converted anyway) and returns a `pass`, `caution` or `hold` verdict, with the naive vs. confounder-adjusted effect and a confidence interval.

Estimates are computed deterministically by statistical code on the hosted service, never by an LLM. This package is a thin MIT-licensed client: it only makes HTTP calls to `https://guardian.sylai.io`.

## Install

```bash
pip install syl-guardian-client                # client only
pip install 'syl-guardian-client[crewai]'      # + CrewAI tool
pip install 'syl-guardian-client[langchain]'   # + LangChain tool
```

Get a trial API key at <https://sylai.io/guardian> and set it:

```bash
export SYL_GUARDIAN_API_KEY=<your key>
```

Each review counts against your plan or trial quota. Uploads do not.

## Use

```python
from syl_guardian_client import GuardianClient

guardian = GuardianClient()  # reads SYL_GUARDIAN_API_KEY

result = guardian.review(
    "ad_spend",
    "conversion_value",
    data=rows,                      # list of row dicts or a DataFrame
    time_col="date",
    step=500,                       # the proposed budget change
    known_confounders=["brand_query_click_pct", "holiday_season_index"],
)

print(result.verdict)               # "pass" | "caution" | "hold"
print(result.correlational_effect, result.ate, result.ci)

if result.held:
    raise RuntimeError(result.recommended_action)
```

Send only the cause, effect, `time_col` and confounder columns. Downstream columns such as `roas`, `clicks` or `conversions` are bad controls and bias the estimate.

For datasets of more than about 200 rows, upload the CSV instead of sending rows inline:

```python
result = guardian.review_file("ad_spend", "conversion_value", "campaign.csv", time_col="date")
```

Uploaded datasets are single-use: each is deleted when reviewed, so upload again to re-run.

### Review arguments

| Argument | Description |
|---|---|
| `cause`, `effect` | Required. Column names for the claim. |
| `data`, `csv` or `dataset_id` | Dataset as row dicts or a DataFrame, CSV text, or an uploaded dataset id. Exactly one. |
| `known_confounders` | Columns to adjust for; otherwise discovered automatically. |
| `time_col`, `adstock` | Time-series handling: row ordering and carryover effects. |
| `step`, `units`, `context` | Restate the effect in business terms. |
| `test_design`, `cate`, `sensitivity` | Optional extras: spend-cut test design, subgroup effects, sensitivity checks. |
| `seed`, `config`, `estimator` | Reproducibility and engine settings. Set `config={"cleaning": "rule", "discovery": "rule"}` for strict reproducibility. |

Errors raise `GuardianError` with `status_code`, `error` (for example `trial_expired`, `quota_exceeded`) and `detail`.

## CrewAI

```python
from crewai import Agent
from syl_guardian_client.integrations.crewai import SylGuardianTool

media_buyer = Agent(
    role="Media Buyer",
    goal="Change ad budgets only when the lift is incremental",
    backstory="Checks every budget change with Syl Guardian before applying it.",
    tools=[SylGuardianTool()],
)
```

## LangChain

```python
from syl_guardian_client.integrations.langchain import guardian_review_tool

tools = [guardian_review_tool()]  # pass to your agent; API errors become tool messages
```

## MCP

Guardian is also a hosted MCP server (`https://guardian.sylai.io/mcp`). See <https://github.com/connor274/syl-guardian-mcp>.

## License

MIT. The Guardian service itself is hosted and closed; this package contains no engine code.
