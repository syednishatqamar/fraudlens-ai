"""Database tables and engine. The URL decides the database: SQLite for tests, PostgreSQL in production."""
from __future__ import annotations

from datetime import datetime, timezone

from sqlalchemy import DateTime, Float, ForeignKey, String, Text, create_engine
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship, sessionmaker
from sqlalchemy.pool import StaticPool


def utcnow() -> datetime:
    return datetime.now(timezone.utc).replace(tzinfo=None)


class Base(DeclarativeBase):
    pass


class CaseRow(Base):
    __tablename__ = "cases"

    id: Mapped[int] = mapped_column(primary_key=True)
    case_ref: Mapped[str] = mapped_column(String(64), unique=True, index=True)
    band: Mapped[str] = mapped_column(String(10), index=True)
    probability: Mapped[float] = mapped_column(Float)
    amount: Mapped[float] = mapped_column(Float)
    card_last4: Mapped[str] = mapped_column(String(4))
    merchant: Mapped[str] = mapped_column(String(200))
    category: Mapped[str] = mapped_column(String(64))
    txn_time: Mapped[datetime] = mapped_column(DateTime)
    case_json: Mapped[str] = mapped_column(Text)          # JSON text keeps the schema portable across databases
    note: Mapped[str] = mapped_column(Text)
    note_source: Mapped[str] = mapped_column(String(10))
    status: Mapped[str] = mapped_column(String(10), index=True, default="PENDING")
    final_action: Mapped[str | None] = mapped_column(String(10), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)
    decisions: Mapped[list["DecisionRow"]] = relationship(back_populates="case", order_by="DecisionRow.id")


class DecisionRow(Base):
    """Append-only: rows are never updated or deleted."""

    __tablename__ = "decisions"

    id: Mapped[int] = mapped_column(primary_key=True)
    case_id: Mapped[int] = mapped_column(ForeignKey("cases.id"), index=True)
    decision: Mapped[str] = mapped_column(String(10))
    final_action: Mapped[str | None] = mapped_column(String(10), nullable=True)
    analyst: Mapped[str] = mapped_column(String(50))
    comment: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)
    case: Mapped[CaseRow] = relationship(back_populates="decisions")


def make_session_factory(url: str):
    kwargs = {}
    if url.startswith("sqlite"):
        kwargs["connect_args"] = {"check_same_thread": False}
        if ":memory:" in url or url == "sqlite://":
            kwargs["poolclass"] = StaticPool            # one shared in-memory database for all connections
    engine = create_engine(url, **kwargs)
    Base.metadata.create_all(engine)
    return sessionmaker(engine, expire_on_commit=False)
