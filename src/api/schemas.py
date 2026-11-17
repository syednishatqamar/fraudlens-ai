"""Request and response shapes, with the validation rules of the analyst workflow."""
from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field, model_validator

Band = Literal["REVIEW", "BLOCK"]               # CLEAR transactions are logged and auto-cleared, never queued
Action = Literal["CLEAR", "REVIEW", "BLOCK"]


class CaseIn(BaseModel):
    case_ref: str = Field(min_length=1, max_length=64)
    band: Band
    probability: float = Field(ge=0, le=1)
    amount: float = Field(ge=0)
    card_last4: str = Field(pattern=r"^\d{1,4}$")  # never accept a full card number
    merchant: str = Field(max_length=200)
    category: str = Field(max_length=64)
    txn_time: datetime
    case: dict
    note: str
    note_source: Literal["llm", "template"]


class DecisionIn(BaseModel):
    decision: Literal["APPROVE", "REJECT", "ESCALATE"]
    analyst: str = Field(min_length=1, max_length=50)
    final_action: Action | None = None
    comment: str | None = Field(default=None, max_length=2000)

    @model_validator(mode="after")
    def check_rules(self):
        has_comment = bool(self.comment and len(self.comment.strip()) >= 3)
        if self.decision == "APPROVE" and self.final_action is not None:
            raise ValueError("APPROVE accepts the system's action; do not send final_action")
        if self.decision == "REJECT" and (self.final_action is None or not has_comment):
            raise ValueError("REJECT needs final_action (the action you chose instead) and a comment")
        if self.decision == "ESCALATE" and (self.final_action is not None or not has_comment):
            raise ValueError("ESCALATE needs a comment and no final_action")
        return self


class DecisionOut(BaseModel):
    id: int
    decision: str
    final_action: str | None
    analyst: str
    comment: str | None
    created_at: datetime


class CaseSummary(BaseModel):
    id: int
    case_ref: str
    band: str
    probability: float
    amount: float
    card_last4: str
    merchant: str
    category: str
    txn_time: datetime
    status: str
    final_action: str | None
    created_at: datetime


class CaseDetail(CaseSummary):
    note: str
    note_source: str
    case: dict
    decisions: list[DecisionOut]
