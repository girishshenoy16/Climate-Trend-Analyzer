import os
import sys
import pytest

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
APP_DIR = os.path.join(BASE_DIR, "app")

sys.path.append(APP_DIR)

from app import app


@pytest.fixture
def client():
    app.config["TESTING"] = True

    with app.test_client() as client:
        yield client