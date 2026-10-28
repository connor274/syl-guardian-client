from __future__ import annotations

from typing import Any, Optional

from pydantic import BaseModel, Field, model_validator

TOOL_NAME = "syl_guardian_causal_check"
TOOL_DESCRIPTION = (
    "Verifies that a claimed effect is causal before acting on it, e.g. before "
    "increasing an ad budget because ROAS looks high. Takes the cause and effect "
    "column names plus the dataset, and returns a pass/caution/hold verdict, the "
    "naive vs. confounder-adjusted effect and a recommended action. On 'hold', "
    "do not act on the claim."
)


class GuardianReviewInput(BaseModel):
    """Arguments an agent supplies to the Guardian tool."""

    cause: str = Field(..., description="Column name of the cause (e.g. ad spend).")
    effect: str = Field(
        ..., description="Column name of the claimed effect (e.g. conversion value)."
    )
    data: Optional[list[dict[str, Any]]] = Field(
        default=None, description="Dataset as a list of row objects (small datasets)."
    )
    csv: Optional[str] = Field(default=None, description="Dataset as CSV text.")
    dataset_id: Optional[str] = Field(
        default=None, description="ID of a previously uploaded dataset."
    )
    known_confounders: Optional[list[str]] = Field(
        default=None,
        description="Columns to adjust for (e.g. seasonality, brand search share).",
    )
    time_col: Optional[str] = Field(
        default=None, description="Date/period column used to order the rows."
    )
    step: Optional[float] = Field(
        default=None, description="Proposed change in the cause, to restate the effect."
    )
    context: Optional[str] = Field(default=None, description="Short business context.")

    @model_validator(mode="after")
    def _exactly_one_dataset(self) -> "GuardianReviewInput":
        given = [v for v in (self.data, self.csv, self.dataset_id) if v is not None]
        if len(given) != 1:
            raise ValueError("Provide exactly one of `data`, `csv` or `dataset_id`.")
        return self
