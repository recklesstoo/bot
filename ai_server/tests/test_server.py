import json
import math
import sys
from pathlib import Path

import pytest
import requests
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import server  # noqa: E402


def make_bars(n=100, start=20000.0):
    bars = []
    for i in range(n):
        c = start + 20 * math.sin(i / 7) + i * 0.5
        bars.append({"t": f"2026-09-24T09:{i % 60:02d}:00", "o": c - 1, "h": c + 3,
                     "l": c - 3, "c": c, "v": 1000 + (i % 10) * 50})
    return bars


def payload(**kw):
    base = {"instrument": "MNQ 12-26", "timeframe": "5 Minute", "tick_size": 0.25,
            "point_value": 2.0, "bars": make_bars()}
    base.update(kw)
    return base


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(server, "LOG_FILE", tmp_path / "log.jsonl")
    monkeypatch.setattr(server, "API_TOKEN", "")
    monkeypatch.setattr(server, "ALLOWED_HOSTS", {"testserver", "127.0.0.1", "localhost"})
    server.STATE.update(paused=False, nt=None, nt_seen=0.0, last_request=None)
    for q in (server.DECISIONS, server.EVENTS, server.COMMANDS, server.PENDING):
        q.clear()
    return TestClient(server.app)


PANEL = {"X-Panel": "1"}


def fake_llm(monkeypatch, content):
    monkeypatch.setattr(server, "ask_ollama", lambda prompt: content)


def test_buy_decision(client, monkeypatch):
    fake_llm(monkeypatch, json.dumps({"action": "buy", "confidence": 0.8, "reason": "tendencia"}))
    r = client.post("/decide", json=payload())
    assert r.status_code == 200
    assert r.json() == {"action": "BUY", "confidence": 0.8, "reason": "tendencia"}


def test_invalid_json_is_hold(client, monkeypatch):
    fake_llm(monkeypatch, "no json")
    assert client.post("/decide", json=payload()).json()["action"] == "HOLD"


def test_unknown_action_is_hold(client, monkeypatch):
    fake_llm(monkeypatch, json.dumps({"action": "YOLO", "confidence": 1}))
    assert client.post("/decide", json=payload()).json()["action"] == "HOLD"


def test_ollama_down_is_hold(client, monkeypatch):
    def boom(prompt):
        raise requests.ConnectionError("down")
    monkeypatch.setattr(server, "ask_ollama", boom)
    d = client.post("/decide", json=payload()).json()
    assert d["action"] == "HOLD" and "Ollama" in d["reason"]


def test_too_few_bars_is_hold(client, monkeypatch):
    fake_llm(monkeypatch, json.dumps({"action": "BUY", "confidence": 1}))
    assert client.post("/decide", json=payload(bars=make_bars(10))).json()["action"] == "HOLD"


def test_short_blocked_when_disabled(client, monkeypatch):
    fake_llm(monkeypatch, json.dumps({"action": "SELL", "confidence": 0.9}))
    assert client.post("/decide", json=payload(allow_shorts=False)).json()["action"] == "HOLD"
    # Con posición larga, SELL sigue permitido (sirve para salir)
    assert client.post("/decide", json=payload(allow_shorts=False, position="LONG")).json()["action"] == "SELL"


def test_exit_when_flat_is_hold(client, monkeypatch):
    fake_llm(monkeypatch, json.dumps({"action": "EXIT", "confidence": 0.9}))
    assert client.post("/decide", json=payload()).json()["action"] == "HOLD"


def test_confidence_clamped(client, monkeypatch):
    fake_llm(monkeypatch, json.dumps({"action": "BUY", "confidence": 7}))
    assert client.post("/decide", json=payload()).json()["confidence"] == 1.0


def test_token_required(client, monkeypatch):
    monkeypatch.setattr(server, "API_TOKEN", "secret")
    fake_llm(monkeypatch, json.dumps({"action": "BUY", "confidence": 1}))
    assert client.post("/decide", json=payload()).status_code == 401
    assert client.post("/decide", json=payload(), headers={"X-Api-Token": "secret"}).status_code == 200


def test_prompt_contains_indicators():
    req = server.DecideRequest(**payload())
    prompt = server.build_prompt(req, server.to_frame(req.bars))
    assert "EMA21" in prompt and "RSI14" in prompt and "n/a" not in prompt


# ---------------- Panel y latido de NinjaTrader ----------------

def heartbeat(client, **kw):
    body = {"account": "Sim101", "instrument": "MNQ 12-26", "position": "FLAT"}
    body.update(kw)
    return client.post("/nt/heartbeat", json=body)


def test_dashboard_served(client):
    r = client.get("/")
    assert r.status_code == 200 and "Panel del Bot" in r.text


def test_foreign_host_rejected(client):
    assert client.get("/api/state", headers={"Host": "evil.example.com"}).status_code == 403


def test_panel_header_required(client):
    heartbeat(client)
    assert client.post("/api/command", json={"type": "PING"}).status_code == 403


def test_command_requires_nt(client):
    r = client.post("/api/command", json={"type": "PING"}, headers=PANEL)
    assert r.status_code == 409


def test_unknown_command(client):
    heartbeat(client)
    assert client.post("/api/command", json={"type": "YOLO"}, headers=PANEL).status_code == 400


def test_ping_round_trip(client):
    assert heartbeat(client).json() == {"commands": []}
    state = client.get("/api/state").json()
    assert state["nt"]["connected"] and state["nt"]["status"]["account"] == "Sim101"

    cmd = client.post("/api/command", json={"type": "PING"}, headers=PANEL).json()
    got = heartbeat(client).json()["commands"]
    assert got == [{"id": cmd["id"], "type": "PING"}]
    assert heartbeat(client).json()["commands"] == []          # no se entrega dos veces

    heartbeat(client, acks=[{"id": cmd["id"], "result": "PONG"}], events=["Ejecutada: Buy 1 @ 20000"])
    state = client.get("/api/state").json()
    c = next(c for c in state["commands"] if c["id"] == cmd["id"])
    assert c["status"] == "done" and c["result"] == "PONG" and c["latency_ms"] is not None
    assert any("Ejecutada" in e["text"] for e in state["events"])


def test_stale_command_expires(client, monkeypatch):
    heartbeat(client)
    cmd = client.post("/api/command", json={"type": "BUY"}, headers=PANEL).json()
    server.COMMANDS[cmd["id"]]["created"] -= server.COMMAND_TTL + 1
    assert heartbeat(client).json()["commands"] == []
    assert server.COMMANDS[cmd["id"]]["status"] == "expired"


def test_pause_skips_ollama(client, monkeypatch):
    def boom(prompt):
        raise AssertionError("no debe llamar a Ollama en pausa")
    monkeypatch.setattr(server, "ask_ollama", boom)
    assert client.post("/api/pause", json={"paused": True}, headers=PANEL).json() == {"paused": True}
    d = client.post("/decide", json=payload()).json()
    assert d["action"] == "HOLD" and "pausa" in d["reason"]


def test_decisions_listed(client, monkeypatch):
    fake_llm(monkeypatch, json.dumps({"action": "BUY", "confidence": 0.8, "reason": "x"}))
    client.post("/decide", json=payload())
    state = client.get("/api/state").json()
    assert state["decisions"][0]["action"] == "BUY"


def test_ai_test_uses_demo_then_real(client, monkeypatch):
    fake_llm(monkeypatch, json.dumps({"action": "HOLD", "confidence": 0.3, "reason": "lateral"}))
    r = client.post("/api/test-ai", headers=PANEL).json()
    assert r["action"] == "HOLD" and "ejemplo" in r["source"]
    client.post("/decide", json=payload())
    r = client.post("/api/test-ai", headers=PANEL).json()
    assert "reales" in r["source"]


def test_chat(client, monkeypatch):
    class Resp:
        def raise_for_status(self): pass
        def json(self): return {"message": {"content": "Hola, todo bien"}}
    monkeypatch.setattr(server.requests, "post", lambda *a, **k: Resp())
    r = client.post("/api/chat", json={"message": "hola"}, headers=PANEL)
    assert r.status_code == 200 and r.json()["reply"] == "Hola, todo bien"
