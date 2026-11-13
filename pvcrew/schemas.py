"""The structured case record every stage reads and writes."""
from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator

Sex = Literal["male", "female", "unknown"]
Outcome = Literal[
    "recovered",
    "recovering",
    "not_recovered",
    "recovered_with_sequelae",
    "fatal",
    "unknown",
]


class Case(BaseModel):
    model_config = ConfigDict(extra="forbid")

    patient_age: Optional[int] = Field(
        default=None, ge=0, le=120, description="Patient age in whole years; null if not stated"
    )
    patient_sex: Sex = Field(default="unknown", description="male, female or unknown")
    suspect_drug: str = Field(min_length=1, description="The medicine the reporter suspects caused the event")
    reactions: list[str] = Field(min_length=1, description="Each adverse reaction or symptom described")
    serious: bool = Field(
        description="true if the event was life-threatening, fatal, disabling, or needed hospital care"
    )
    outcome: Outcome = Field(default="unknown", description="Patient outcome at the time of the report")

    @field_validator("suspect_drug")
    @classmethod
    def _clean_drug(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("suspect_drug is empty")
        return v

    @field_validator("reactions")
    @classmethod
    def _clean_reactions(cls, v: list[str]) -> list[str]:
        seen, out = set(), []
        for r in v:
            r = r.strip()
            if r and r.lower() not in seen:
                seen.add(r.lower())
                out.append(r)
        if not out:
            raise ValueError("reactions is empty")
        return out
