"""FastAPI service: queue of REVIEW/BLOCK cases and the analyst decision workflow.

Run:  DATABASE_URL=sqlite:///./fraudlens.db uvicorn src.api.app:create_app --factory --reload
Set ANALYST_API_KEY to require an X-API-Key header on every endpoint except /health (minimal access control;
a real deployment needs proper authentication). Set CORS_ORIGINS (comma separated) for the dashboard origin.
"""
from __future__ import annotations

import json
import os
import secrets

from fastapi import Depends, FastAPI, Header, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import func, select
from sqlalchemy.exc import IntegrityError

from src.api.db import CaseRow, DecisionRow, make_session_factory
from src.api.schemas import CaseDetail, CaseIn, CaseSummary, DecisionIn, DecisionOut


def create_app(database_url: str | None = None) -> FastAPI:
    url = database_url or os.environ.get("DATABASE_URL", "sqlite:///./fraudlens.db")
    sessions = make_session_factory(url)
    app = FastAPI(title="FraudLens API", version="0.1.0")
    origins = [o.strip() for o in os.environ.get("CORS_ORIGINS", "http://localhost:5173,http://localhost:3000").split(",") if o.strip()]
    app.add_middleware(CORSMiddleware, allow_origins=origins, allow_methods=["*"], allow_headers=["*"])

    def get_db():
        with sessions() as db:
            yield db

    def require_key(x_api_key: str | None = Header(default=None)):
        expected = os.environ.get("ANALYST_API_KEY")
        if expected and not (x_api_key and secrets.compare_digest(x_api_key, expected)):
            raise HTTPException(401, "missing or invalid API key")

    def detail(row: CaseRow) -> CaseDetail:
        return CaseDetail(
            **{c: getattr(row, c) for c in CaseSummary.model_fields},
            note=row.note, note_source=row.note_source, case=json.loads(row.case_json),
            decisions=[DecisionOut(**{c: getattr(d, c) for c in DecisionOut.model_fields}) for d in row.decisions],
        )

    @app.get("/health")
    def health():
        return {"status": "ok"}

    @app.post("/cases", response_model=CaseDetail, status_code=201, dependencies=[Depends(require_key)])
    def create_case(body: CaseIn, db=Depends(get_db)):
        row = CaseRow(**body.model_dump(exclude={"case"}), case_json=json.dumps(body.case), status="PENDING")
        db.add(row)
        try:
            db.commit()
        except IntegrityError:
            db.rollback()
            raise HTTPException(409, f"case_ref {body.case_ref!r} already exists")
        return detail(row)

    @app.get("/cases", response_model=list[CaseSummary], dependencies=[Depends(require_key)])
    def list_cases(
        status: str | None = Query(None, pattern="^(PENDING|ESCALATED|DECIDED)$"),
        band: str | None = Query(None, pattern="^(REVIEW|BLOCK)$"),
        limit: int = Query(50, ge=1, le=200),
        offset: int = Query(0, ge=0),
        db=Depends(get_db),
    ):
        q = select(CaseRow)
        if status:
            q = q.where(CaseRow.status == status)
        if band:
            q = q.where(CaseRow.band == band)
        # highest risk first: BLOCK before REVIEW, then by probability, then oldest
        q = q.order_by(CaseRow.band.asc(), CaseRow.probability.desc(), CaseRow.id.asc()).limit(limit).offset(offset)
        return [CaseSummary(**{c: getattr(r, c) for c in CaseSummary.model_fields}) for r in db.scalars(q)]

    @app.get("/cases/{case_id}", response_model=CaseDetail, dependencies=[Depends(require_key)])
    def get_case(case_id: int, db=Depends(get_db)):
        row = db.get(CaseRow, case_id)
        if row is None:
            raise HTTPException(404, "case not found")
        return detail(row)

    @app.post("/cases/{case_id}/decision", response_model=CaseDetail, status_code=201, dependencies=[Depends(require_key)])
    def decide_case(case_id: int, body: DecisionIn, db=Depends(get_db)):
        row = db.get(CaseRow, case_id)
        if row is None:
            raise HTTPException(404, "case not found")
        if row.status == "DECIDED":
            raise HTTPException(409, "case is already decided; decisions are final")
        final = row.band if body.decision == "APPROVE" else body.final_action
        if body.decision == "REJECT" and final == row.band:
            raise HTTPException(422, "REJECT must choose a different action than the system's; use APPROVE to accept it")
        db.add(DecisionRow(case_id=row.id, decision=body.decision, final_action=final, analyst=body.analyst, comment=body.comment))
        if body.decision == "ESCALATE":
            row.status, row.final_action = "ESCALATED", None
        else:
            row.status, row.final_action = "DECIDED", final
        db.commit()
        db.refresh(row)
        return detail(row)

    @app.get("/stats", dependencies=[Depends(require_key)])
    def stats(db=Depends(get_db)):
        by_band = dict(db.execute(select(CaseRow.band, func.count()).group_by(CaseRow.band)).all())
        by_status = dict(db.execute(select(CaseRow.status, func.count()).group_by(CaseRow.status)).all())
        decided = db.scalars(select(CaseRow).where(CaseRow.status == "DECIDED")).all()
        agree = sum(1 for r in decided if r.final_action == r.band)
        return {
            "cases": sum(by_band.values()),
            "by_band": by_band,
            "by_status": by_status,
            "decided": len(decided),
            "analyst_agrees_with_system": agree,
            "analyst_overrides": len(decided) - agree,
            "agreement_rate": (agree / len(decided)) if decided else None,
        }

    return app

