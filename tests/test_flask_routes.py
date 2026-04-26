def test_home_page(client):

    response = client.get("/")

    assert response.status_code == 200


def test_dashboard_page(client):

    response = client.get("/dashboard")

    assert response.status_code == 200


def test_forecast_page(client):

    response = client.get("/forecast")

    assert response.status_code == 200


def test_anomaly_page(client):

    response = client.get("/anomaly")

    assert response.status_code == 200


def test_home_contains_title(client):

    response = client.get("/")

    assert b"Climate Trend Analyzer" in response.data


def test_dashboard_contains_text(client):

    response = client.get("/dashboard")

    assert b"Model Performance Dashboard" in response.data


def test_chart_image_loads(client):

    response = client.get(
        "/outputs/future_forecast.png"
    )

    assert response.status_code == 200