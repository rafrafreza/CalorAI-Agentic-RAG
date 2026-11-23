"""
Shared fixtures. Every test gets a fresh SQLite file and runs fully offline:
no API key is read, the dense channel is disabled, and LLM-path tests inject a
scripted model through :func:`backend.core.llm.override_chat_model`.
"""

from __future__ import annotations

import os

os.environ["CALORAI_LLM_MODE"] = "offline"
os.environ["GOOGLE_API_KEY"] = ""
os.environ["GEMINI_API_KEY"] = ""
os.environ["CALORAI_ENABLE_DENSE"] = "false"

import pytest

from backend.core.config import get_settings
from backend.db.database import init_db

get_settings.cache_clear()


@pytest.fixture(autouse=True)
def fresh_db(tmp_path, monkeypatch):
    path = str(tmp_path / "calorai_test.db")
    monkeypatch.setenv("CALORAI_DB_PATH", path)
    monkeypatch.setenv("CALORAI_UPLOAD_DIR", str(tmp_path / "uploads"))
    get_settings.cache_clear()
    init_db(path)
    yield path
    get_settings.cache_clear()


@pytest.fixture
def client():
    from fastapi.testclient import TestClient

    from backend.main import app

    with TestClient(app) as c:
        yield c
