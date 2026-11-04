"""Generate synthetic API request logs for the demo.

The data covers 2 days (2026-11-01 and 2026-11-02) for 4 services.

Hidden story the agent has to discover:
  * payment-api is deployed as v2.4.1 at 13:45 on 2026-11-02.
  * From 14:00 to 18:00 its error rate jumps from ~1.5% to ~28%,
    mostly DB_CONNECTION_TIMEOUT (HTTP 503).
  * auth-service has a short 1-hour bump at 03:00 (a red herring: RATE_LIMITED
    errors in the middle of the night, far smaller than the payment-api incident).
  * Other services stay normal.

Run:  python generate_data.py
"""
from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

OUT_PATH = Path(__file__).parent / "data" / "logs.csv"
START = datetime(2026, 11, 1, 0, 0, 0)
HOURS = 48

# error_type -> HTTP status code
STATUS_FOR = {
    "VALIDATION_ERROR": 400,
    "AUTH_FAILED": 401,
    "RATE_LIMITED": 429,
    "INTERNAL_ERROR": 500,
    "DB_CONNECTION_TIMEOUT": 503,
    "UPSTREAM_TIMEOUT": 504,
}

# service -> (requests per hour at peak, normal error rate, normal error mix)
SERVICES = {
    "payment-api": (140, 0.015, {"VALIDATION_ERROR": 0.6, "INTERNAL_ERROR": 0.4}),
    "auth-service": (160, 0.02, {"AUTH_FAILED": 0.8, "VALIDATION_ERROR": 0.2}),
    "search-service": (200, 0.01, {"VALIDATION_ERROR": 0.7, "INTERNAL_ERROR": 0.3}),
    "notification-service": (90, 0.03, {"UPSTREAM_TIMEOUT": 0.5, "INTERNAL_ERROR": 0.5}),
}

DEPLOY_TIME = datetime(2026, 11, 2, 13, 45)


def _pick(rng: np.random.Generator, mix: dict[str, float]) -> str:
    names = list(mix)
    return str(rng.choice(names, p=np.array(list(mix.values())) / sum(mix.values())))


def _version(service: str, ts: datetime) -> str:
    if service == "payment-api":
        return "v2.4.1" if ts >= DEPLOY_TIME else "v2.4.0"
    return {"auth-service": "v5.1.2", "search-service": "v3.0.7"}.get(service, "v1.9.4")


def generate(path: Path = OUT_PATH, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for h in range(HOURS):
        hour_start = START + timedelta(hours=h)
        # more traffic during the day, less at night
        daytime = 0.6 + 0.4 * np.sin((hour_start.hour - 6) / 24 * 2 * np.pi) ** 2
        for service, (peak, base_err, base_mix) in SERVICES.items():
            n = int(peak * daytime * rng.uniform(0.9, 1.1))
            err_rate, mix, err_latency_x = base_err, base_mix, 1.5

            # --- the incident: payment-api, 14:00-18:00 on Nov 2 ---
            if (
                service == "payment-api"
                and hour_start.date() == datetime(2026, 11, 2).date()
                and 14 <= hour_start.hour < 18
            ):
                err_rate = 0.28
                mix = {"DB_CONNECTION_TIMEOUT": 0.72, "UPSTREAM_TIMEOUT": 0.2, "INTERNAL_ERROR": 0.08}
                err_latency_x = 4.0
            # --- red herring: small auth-service bump at 03:00 on Nov 2 ---
            if service == "auth-service" and hour_start == datetime(2026, 11, 2, 3, 0):
                err_rate = 0.12
                mix = {"RATE_LIMITED": 0.9, "AUTH_FAILED": 0.1}

            for _ in range(n):
                ts = hour_start + timedelta(seconds=int(rng.integers(0, 3600)))
                is_err = rng.random() < err_rate
                err_type = _pick(rng, mix) if is_err else ""
                latency = float(rng.lognormal(mean=4.8, sigma=0.35))  # ~120 ms typical
                if is_err:
                    latency *= err_latency_x
                rows.append(
                    {
                        "timestamp": ts.strftime("%Y-%m-%d %H:%M:%S"),
                        "service": service,
                        "status_code": STATUS_FOR[err_type] if is_err else 200,
                        "error_type": err_type,
                        "latency_ms": round(latency, 1),
                        "deploy_version": _version(service, ts),
                    }
                )

    df = pd.DataFrame(rows).sort_values("timestamp").reset_index(drop=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return df


if __name__ == "__main__":
    df = generate()
    print(f"Wrote {len(df):,} log rows to {OUT_PATH}")
