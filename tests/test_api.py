"""HTTP API, end to end through FastAPI (offline engine; no API key needed)."""

import io

from PIL import Image


def chat(client, message, session="api"):
    r = client.post("/api/chat", data={"message": message, "session_id": session})
    assert r.status_code == 200, r.text
    return r.json()


def test_health_and_metrics(client):
    from backend.core.telemetry import recorder

    recorder.reset()  # the recorder is process-wide
    h = client.get("/health").json()
    assert h["status"] == "ok" and h["llm"]["mode"] == "offline" and h["rag"]["foods"] >= 400
    chat(client, "had a samosa")
    m = client.get("/metrics").json()
    assert m["chat"]["text"]["count"] == 1 and "retrieve" in m["node"]


def test_full_conversation_flow(client):
    r = chat(client, "had 2 parathas and chai for breakfast")
    assert r["mode"] == "offline" and r["meals_changed"][0]["meal"]["calories"] == 640
    assert {g["kb_id"] for g in r["groundings"]} == {"paratha", "masala_chai"}

    r = chat(client, "actually that was 3 parathas not 2")
    assert r["meals_changed"][0]["action"] == "updated" and r["day"]["total_calories"] == 900

    chat(client, "I'm vegetarian btw")
    r = chat(client, "leftover biryani, maybe two thirds of the box")
    assert r["meals_changed"][0]["meal"]["items"][0]["kb_id"] == "veg_biryani"  # profile-aware RAG

    r = chat(client, "how am I doing?")
    assert "1200 kcal" in r["response"]

    dash = client.get("/api/dashboard").json()
    assert dash["day"]["meal_count"] == 2 and dash["grounded_share"] == 1.0
    assert dash["profile"]["diet"] == "vegetarian" and len(dash["trend"]) == 7

    history = client.get("/api/history", params={"session_id": "api"}).json()["messages"]
    assert len(history) == 10 and history[-1]["meta"]["mode"] == "offline"


def test_meal_and_memory_endpoints(client):
    meal_id = chat(client, "a bowl of dal and 2 rotis")["meals_changed"][0]["meal"]["id"]
    assert client.get("/api/meals").json()["meals"][0]["id"] == meal_id
    assert client.delete(f"/api/meals/{meal_id}").status_code == 200
    assert client.delete(f"/api/meals/{meal_id}").status_code == 404

    chat(client, "I'm allergic to peanuts")
    facts = client.get("/api/memory").json()
    assert facts["profile"]["allergens"] == ["peanut"]
    key = facts["memories"][0]["key"]
    assert client.delete(f"/api/memory/{key}").status_code == 200
    assert client.get("/api/memory").json()["count"] == 0


def test_rag_endpoints(client):
    s = client.get("/api/rag/search", params={"q": "biryani", "diet": "vegetarian"}).json()
    assert s["candidates"][0]["id"] == "veg_biryani" and s["profile_promoted"]
    a = client.get("/api/rag/analyze", params={"text": "2 rotis and a bowl of dal"}).json()
    assert a["grounded_kcal"] == 400
    assert client.get("/api/rag/stats").json()["foods"] >= 400
    assert client.get("/api/rag/foods/paratha", params={"quantity": 2}).json()["scaled"]["kcal"] == 520
    assert client.get("/api/rag/foods/nope").status_code == 404
    assert client.get("/api/rag/search", params={"q": "x", "diet": "carnivore"}).status_code == 422


def test_input_validation(client):
    assert client.post("/api/chat", data={"message": ""}).status_code == 400
    assert client.post("/api/chat", data={"message": "x" * 5000}).status_code == 413
    bad = client.post("/api/chat", data={"message": "lunch"}, files={"image": ("a.jpg", b"not an image", "image/jpeg")})
    assert bad.status_code == 415
    assert client.get("/api/dashboard", params={"date": "2999-01-01"}).status_code == 422


def test_photo_without_key_is_handled_gracefully(client):
    buf = io.BytesIO()
    Image.new("RGB", (64, 64), "orange").save(buf, format="PNG")
    r = client.post(
        "/api/chat",
        data={"message": "2 rotis", "session_id": "img"},
        files={"image": ("plate.png", buf.getvalue(), "image/png")},
    )
    assert r.status_code == 200
    assert r.json()["meals_changed"]  # caption still logged; vision skipped without a key
