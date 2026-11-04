"""Tests for the data tools (no LLM or internet needed).  Run:  pytest -q"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import generate_data  # noqa: E402
import tools  # noqa: E402


@pytest.fixture(scope="module", autouse=True)
def data(tmp_path_factory):
    path = tmp_path_factory.mktemp("d") / "logs.csv"
    generate_data.generate(path)
    tools.DATA_PATH = path
    tools._load.cache_clear()
    yield
    tools._load.cache_clear()


def test_overview_lists_services():
    out = tools.get_data_overview()
    assert "payment-api" in out["services"]
    assert out["first_timestamp"].startswith("2026-11-01")


def test_spike_found_in_payment_api():
    out = tools.detect_spike("payment-api")
    assert len(out["spike_windows"]) == 1
    w = out["spike_windows"][0]
    assert w["start"] == "2026-11-02 14:00"
    assert w["end_exclusive"] == "2026-11-02 18:00"
    assert w["avg_error_rate_pct"] > 15


@pytest.mark.parametrize("service", ["search-service", "notification-service"])
def test_no_false_alarm_on_healthy_services(service):
    assert tools.detect_spike(service)["spike_windows"] == []


def test_auth_red_herring_is_flagged_but_much_smaller_than_payment_incident():
    auth = tools.detect_spike("auth-service")["spike_windows"]
    pay = tools.detect_spike("payment-api")["spike_windows"][0]
    assert len(auth) == 1 and auth[0]["start"] == "2026-11-02 03:00" and auth[0]["hours"] == 1
    assert auth[0]["total_errors"] * 5 < pay["total_errors"]


def test_top_errors_points_to_db_timeouts():
    out = tools.top_errors("payment-api", "2026-11-02 14:00", "2026-11-02 18:00")
    assert next(iter(out["error_types"])) == "DB_CONNECTION_TIMEOUT"
    assert out["deploy_versions_running"] == {"v2.4.1": out["total_requests"]}
    assert out["avg_latency_ms_errors"] > out["avg_latency_ms_success"]


def test_deploy_history_shows_new_version_before_spike():
    versions = {v["version"]: v for v in tools.deploy_history("payment-api")["versions"]}
    assert versions["v2.4.1"]["first_seen"] < "2026-11-02 14:00"


def test_bad_inputs_return_error_not_exception():
    assert "error" in tools.detect_spike("nope-service")
    assert "error" in tools.top_errors("payment-api", "garbage", "garbage")
    assert "error" in tools.error_rate_by_hour("payment-api", date="2030-01-01")
