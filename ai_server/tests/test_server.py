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
    return TestClient(server.app)


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
