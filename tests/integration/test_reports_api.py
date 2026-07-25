"""Integration tests for GET & DELETE /api/v1/reports endpoints."""

from uuid import uuid4

from fastapi.testclient import TestClient

from backend.dependencies import get_report_service


def test_list_reports_empty(client: TestClient) -> None:
    """Test GET /api/v1/reports when no reports exist."""

    response = client.get("/api/v1/reports")

    assert response.status_code == 200
    data = response.json()
    assert data["reports"] == []
    assert data["total"] == 0
    assert data["status"] == "ready"


def test_reports_lifecycle(client: TestClient) -> None:
    """Test listing, retrieving, and deleting a report via API."""

    report_service = get_report_service()
    upload_id = uuid4()
    report_metadata = report_service.store_report(
        upload_id,
        content={"path": "/outputs/reports/rep.pdf", "summary": "Sample"},
    )

    # 1. List reports
    list_res = client.get("/api/v1/reports")
    assert list_res.status_code == 200
    list_data = list_res.json()
    assert list_data["total"] == 1
    assert list_data["reports"][0]["report_id"] == str(report_metadata.report_id)

    # 2. Get single report
    get_res = client.get(f"/api/v1/reports/{upload_id}")
    assert get_res.status_code == 200
    get_data = get_res.json()
    assert get_data["report_id"] == str(report_metadata.report_id)
    assert get_data["path"] == "/outputs/reports/rep.pdf"

    # 3. Delete report
    del_res = client.delete(f"/api/v1/reports/{upload_id}")
    assert del_res.status_code == 204

    # 4. Verify 404 after deletion
    get_res_after = client.get(f"/api/v1/reports/{upload_id}")
    assert get_res_after.status_code == 404


def test_get_report_not_found(client: TestClient) -> None:
    """Test GET /api/v1/reports/{id} for non-existent upload ID returns 404."""

    response = client.get(f"/api/v1/reports/{uuid4()}")

    assert response.status_code == 404
    data = response.json()
    assert data["detail"] == "Report was not found."


def test_delete_report_not_found(client: TestClient) -> None:
    """Test DELETE /api/v1/reports/{id} for non-existent upload ID returns 404."""

    response = client.delete(f"/api/v1/reports/{uuid4()}")

    assert response.status_code == 404
    data = response.json()
    assert data["detail"] == "Report was not found."
