"""Integration tests for application root and health endpoints."""

from fastapi.testclient import TestClient


def test_root_endpoint(client: TestClient) -> None:
    """Test GET / returns basic application information."""

    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "running"
    assert "application" in data
    assert "version" in data


def test_main_health_endpoint(client: TestClient) -> None:
    """Test GET /health returns application health status."""

    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"


def test_api_v1_health_endpoint(client: TestClient) -> None:
    """Test GET /api/v1/health returns structured HealthResponse metadata."""

    response = client.get("/api/v1/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "application" in data
    assert "version" in data
    assert "environment" in data
    assert "timestamp" in data
