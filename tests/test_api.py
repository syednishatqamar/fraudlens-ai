import pytest
from fastapi.testclient import TestClient

from src.api.app import create_app


@pytest.fixture()
def client(monkeypatch):
    monkeypatch.delenv("ANALYST_API_KEY", raising=False)
    return TestClient(create_app("sqlite://"))     # fresh in-memory database per test


def case_body(ref="c1", band="BLOCK", p=0.93, amount=884.02, last4="1064"):
    return {
        "case_ref": ref, "band": band, "probability": p, "amount": amount, "card_last4": last4,
        "merchant": "Acme", "category": "shopping_pos", "txn_time": "2020-05-31T23:03:03",
        "case": {"why": "test", "nan_free": 1}, "note": "BLOCK recommended.", "note_source": "template",
    }


def make(client, **kw):
    r = client.post("/cases", json=case_body(**kw))
    assert r.status_code == 201, r.text
    return r.json()


def test_health(client):
    assert client.get("/health").json() == {"status": "ok"}


def test_create_and_fetch_case_roundtrip(client):
    c = make(client)
    assert c["status"] == "PENDING" and c["final_action"] is None and c["decisions"] == []
    got = client.get(f"/cases/{c['id']}").json()
    assert got["case"] == {"why": "test", "nan_free": 1} and got["note_source"] == "template"
    assert client.get("/cases/999").status_code == 404


def test_create_validation(client):
    assert client.post("/cases", json={**case_body(), "band": "CLEAR"}).status_code == 422       # CLEAR is never queued
    assert client.post("/cases", json={**case_body(), "card_last4": "1234567890123456"}).status_code == 422   # no full card numbers
    assert client.post("/cases", json={**case_body(), "probability": 1.5}).status_code == 422
    make(client, ref="dup")
    assert client.post("/cases", json=case_body(ref="dup")).status_code == 409


def test_list_filters_and_risk_order(client):
    make(client, ref="a", band="REVIEW", p=0.30)
    make(client, ref="b", band="BLOCK", p=0.90)
    make(client, ref="c", band="BLOCK", p=0.95)
    refs = [c["case_ref"] for c in client.get("/cases").json()]
    assert refs == ["c", "b", "a"]                                  # BLOCK before REVIEW, higher probability first
    assert [c["case_ref"] for c in client.get("/cases?band=REVIEW").json()] == ["a"]
    assert len(client.get("/cases?status=PENDING").json()) == 3 and client.get("/cases?status=DECIDED").json() == []
    assert client.get("/cases?status=bogus").status_code == 422
    assert len(client.get("/cases?limit=1&offset=1").json()) == 1


def test_approve_accepts_the_system_action(client):
    c = make(client, band="BLOCK")
    r = client.post(f"/cases/{c['id']}/decision", json={"decision": "APPROVE", "analyst": "ana"})
    body = r.json()
    assert r.status_code == 201 and body["status"] == "DECIDED" and body["final_action"] == "BLOCK"
    assert body["decisions"][0]["analyst"] == "ana"


def test_reject_needs_a_different_action_and_a_comment(client):
    c = make(client, band="BLOCK")
    url = f"/cases/{c['id']}/decision"
    assert client.post(url, json={"decision": "REJECT", "analyst": "ana"}).status_code == 422
    assert client.post(url, json={"decision": "REJECT", "analyst": "ana", "final_action": "REVIEW"}).status_code == 422   # no comment
    same = client.post(url, json={"decision": "REJECT", "analyst": "ana", "final_action": "BLOCK", "comment": "no"})
    assert same.status_code == 422
    ok = client.post(url, json={"decision": "REJECT", "analyst": "ana", "final_action": "CLEAR", "comment": "customer confirmed purchase"})
    assert ok.status_code == 201 and ok.json()["final_action"] == "CLEAR" and ok.json()["status"] == "DECIDED"


def test_approve_must_not_carry_a_final_action(client):
    c = make(client)
    r = client.post(f"/cases/{c['id']}/decision", json={"decision": "APPROVE", "analyst": "ana", "final_action": "CLEAR"})
    assert r.status_code == 422


def test_escalate_then_senior_decides_and_history_is_kept(client):
    c = make(client, band="REVIEW", p=0.3)
    url = f"/cases/{c['id']}/decision"
    assert client.post(url, json={"decision": "ESCALATE", "analyst": "ana"}).status_code == 422          # comment required
    e = client.post(url, json={"decision": "ESCALATE", "analyst": "ana", "comment": "unclear, need senior view"}).json()
    assert e["status"] == "ESCALATED" and e["final_action"] is None
    assert client.get("/cases?status=ESCALATED").json()[0]["id"] == c["id"]
    d = client.post(url, json={"decision": "APPROVE", "analyst": "sam"}).json()
    assert d["status"] == "DECIDED" and d["final_action"] == "REVIEW"
    assert [(x["decision"], x["analyst"]) for x in d["decisions"]] == [("ESCALATE", "ana"), ("APPROVE", "sam")]


def test_decided_cases_are_final(client):
    c = make(client)
    url = f"/cases/{c['id']}/decision"
    assert client.post(url, json={"decision": "APPROVE", "analyst": "ana"}).status_code == 201
    assert client.post(url, json={"decision": "APPROVE", "analyst": "ana"}).status_code == 409
    assert client.post("/cases/999/decision", json={"decision": "APPROVE", "analyst": "ana"}).status_code == 404


def test_stats_agreement_rate(client):
    assert client.get("/stats").json()["agreement_rate"] is None
    ids = [make(client, ref=f"s{i}")["id"] for i in range(4)]
    for i in ids[:3]:
        client.post(f"/cases/{i}/decision", json={"decision": "APPROVE", "analyst": "ana"})
    client.post(f"/cases/{ids[3]}/decision", json={"decision": "REJECT", "analyst": "ana", "final_action": "CLEAR", "comment": "customer confirmed"})
    s = client.get("/stats").json()
    assert s["cases"] == 4 and s["decided"] == 4 and s["analyst_overrides"] == 1 and s["agreement_rate"] == pytest.approx(0.75)


def test_api_key_is_enforced_only_when_configured(monkeypatch):
    monkeypatch.setenv("ANALYST_API_KEY", "secret-key")
    c = TestClient(create_app("sqlite://"))
    assert c.get("/health").status_code == 200
    assert c.get("/cases").status_code == 401
    assert c.get("/cases", headers={"X-API-Key": "wrong"}).status_code == 401
    assert c.get("/cases", headers={"X-API-Key": "secret-key"}).status_code == 200


def test_case_from_the_real_builder_can_be_stored(client):
    """A CaseFile built by the investigation code serialises to valid JSON and is accepted by the API."""
    import json

    from src.explainability.reasons import CaseExplanation
    from src.investigation.case import build_case
    from src.investigation.serialize import case_to_dict
    from src.models.decision import BLOCK
    from tests.test_investigation import card_rows, make_history

    case = build_case(make_history(card_rows()), 6, 0.93, BLOCK, CaseExplanation(0.9, 0.02, [], [], ""))
    d = case_to_dict(case)
    json.dumps(d, allow_nan=False)                    # raises if anything is not JSON-safe
    body = {**case_body(ref="real"), "case": d, "card_last4": d["card_last4"], "txn_time": d["timestamp"].replace(" ", "T")}
    assert client.post("/cases", json=body).status_code == 201
