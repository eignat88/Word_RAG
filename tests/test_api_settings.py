from word_rag import api


class DummyRagService:
    def __init__(self, settings):
        self.settings = settings


def _reset_runtime():
    api.settings = api.Settings()
    api.service = DummyRagService(api.settings)


def test_ui_settings_payload_contains_only_editable_fields(monkeypatch):
    monkeypatch.setattr(api, "RagService", DummyRagService)
    _reset_runtime()

    payload = api._ui_settings_payload()

    assert set(payload.keys()) == api.EDITABLE_SETTINGS_FIELDS
    assert "database_url" not in payload
    assert "storage_backend" not in payload


def test_update_settings_rejects_unknown_fields(monkeypatch):
    monkeypatch.setattr(api, "RagService", DummyRagService)
    _reset_runtime()

    req = api.SettingsPatchRequest(changes={"database_url": "***"})

    try:
        api.update_settings(req)
        assert False, "Expected HTTPException"
    except api.HTTPException as exc:
        assert exc.status_code == 422
        assert "database_url" in str(exc.detail)


def test_update_settings_updates_only_allowed_fields(monkeypatch):
    monkeypatch.setattr(api, "RagService", DummyRagService)
    _reset_runtime()

    old_db_url = api.settings.database_url
    req = api.SettingsPatchRequest(changes={"top_k": "7", "llm_timeout_sec": "123.5"})

    result = api.update_settings(req)

    assert result["top_k"] == 7
    assert result["llm_timeout_sec"] == 123.5
    assert api.settings.top_k == 7
    assert api.settings.llm_timeout_sec == 123.5
    assert api.settings.database_url == old_db_url
