"""
Unit & Integration Tests for API Security, OWASP Headers,
Rate Limiting, and Payload Protection.
"""

import pytest
from fastapi.testclient import TestClient
from LivestockVisionAI.api.main import app
from LivestockVisionAI.api.security import RateLimiter, verify_api_key
from fastapi import HTTPException, Request


@pytest.fixture
def client():
    return TestClient(app)


def test_owasp_security_headers(client):
    """Verifies that all responses include defensive OWASP headers."""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.headers.get("X-Content-Type-Options") == "nosniff"
    assert response.headers.get("X-Frame-Options") == "DENY"
    assert response.headers.get("X-XSS-Protection") == "1; mode=block"
    assert "max-age" in response.headers.get("Strict-Transport-Security", "")


def test_payload_limit_enforcement(client):
    """Verifies that requests exceeding 15 MB are rejected with HTTP 413."""
    oversized_headers = {"content-length": str(16 * 1024 * 1024)}  # 16 MB
    response = client.post(
        "/api/v1/quality/check",
        headers=oversized_headers,
        content=b"x" * 100,
    )
    assert response.status_code == 413
    assert "Payload too large" in response.json()["detail"]


def test_rate_limiter_logic():
    """Verifies sliding-window rate limiting thresholds."""
    limiter = RateLimiter(requests_per_minute=5)
    ip = "192.168.1.100"

    # First 5 should pass
    for _ in range(5):
        assert limiter.is_rate_limited(ip) is False

    # 6th should be blocked
    assert limiter.is_rate_limited(ip) is True


def test_api_key_auth_verification(monkeypatch):
    """Verifies API key enforcement when LIVESTOCK_API_KEY is configured."""
    monkeypatch.setenv("LIVESTOCK_API_KEY", "secret_key_audit_2026")

    # Mock request without key -> should raise 401
    class DummyRequest:
        headers = {}

    with pytest.raises(HTTPException) as exc_info:
        verify_api_key(DummyRequest(), api_key_header=None)
    assert exc_info.value.status_code == 401

    # Mock request with valid key -> should pass
    assert verify_api_key(DummyRequest(), api_key_header="secret_key_audit_2026") is True


def test_endpoint_auth_enforced_when_enabled(client, monkeypatch):
    """Verifies that protected POST endpoints require authentication when enabled."""
    test_key = "audit_prod_key_9876"
    monkeypatch.setenv("LIVESTOCK_API_KEY", test_key)

    # 1. Unauthenticated request to protected endpoint -> 401
    res_no_auth = client.post(
        "/api/v1/predict/age",
        data={"species": "COW", "dentition_incisor_pairs": 2},
    )
    assert res_no_auth.status_code == 401
    assert "Invalid or missing API key" in res_no_auth.json()["detail"]

    # 2. Invalid API key -> 401
    res_bad_auth = client.post(
        "/api/v1/predict/age",
        data={"species": "COW", "dentition_incisor_pairs": 2},
        headers={"X-API-Key": "wrong_key"},
    )
    assert res_bad_auth.status_code == 401

    # 3. Valid API key -> 200
    res_good_auth = client.post(
        "/api/v1/predict/age",
        data={"species": "COW", "dentition_incisor_pairs": 2},
        headers={"X-API-Key": test_key},
    )
    assert res_good_auth.status_code == 200

    # 4. Bearer token format also accepted -> 200
    res_bearer = client.post(
        "/api/v1/predict/age",
        data={"species": "COW", "dentition_incisor_pairs": 2},
        headers={"Authorization": f"Bearer {test_key}"},
    )
    assert res_bearer.status_code == 200

    # 5. Animals sync endpoint also protected
    res_sync_no_auth = client.post(
        "/api/v1/animals/sync/queue",
        json={"captures": []},
    )
    assert res_sync_no_auth.status_code == 401

    res_sync_good_auth = client.post(
        "/api/v1/animals/sync/queue",
        json={"captures": []},
        headers={"X-API-Key": test_key},
    )
    assert res_sync_good_auth.status_code == 200

    # 6. Public endpoints remain accessible without auth
    res_health = client.get("/health")
    assert res_health.status_code == 200


def test_endpoint_auth_disabled_by_default(client, monkeypatch):
    """Verifies that endpoints remain accessible when auth is not configured (dev/test default)."""
    monkeypatch.delenv("LIVESTOCK_API_KEY", raising=False)

    res = client.post(
        "/api/v1/predict/age",
        data={"species": "COW", "dentition_incisor_pairs": 2},
    )
    assert res.status_code == 200

