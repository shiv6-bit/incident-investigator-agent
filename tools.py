"""Tools the agent can call.

Each tool is a plain Python function. PydanticAI reads the function's name,
type hints and docstring and shows them to the LLM, so the docstrings are
written for the model: they say WHEN to use the tool and what it returns.

Tools never crash on bad input; they return {"error": "..."} so the model
can read the message and fix its next call.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

DATA_PATH = Path(__file__).parent / "data" / "logs.csv"


@lru_cache(maxsize=1)
def _load() -> pd.DataFrame:
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"{DATA_PATH} not found. Run `python generate_data.py` first.")
    df = pd.read_csv(DATA_PATH, parse_dates=["timestamp"])
    df["error_type"] = df["error_type"].fillna("")
    df["is_error"] = df["status_code"] >= 400  # an "error" = any HTTP status >= 400
    df["hour"] = df["timestamp"].dt.floor("h")
    return df


def _service_df(service: str) -> tuple[pd.DataFrame | None, dict | None]:
    df = _load()
    sub = df[df["service"] == service]
    if sub.empty:
        return None, {"error": f"Unknown service '{service}'. Known services: {sorted(df['service'].unique())}"}
    return sub, None


def _fmt(ts: pd.Timestamp) -> str:
    return ts.strftime("%Y-%m-%d %H:%M")


# --------------------------------------------------------------------------
# Tool 1: orient yourself
# --------------------------------------------------------------------------
def get_data_overview() -> dict:
    """Describe the log data that is available: time range, services and row count.

    Call this FIRST to learn which services exist and what dates the logs cover.
    """
    df = _load()
    return {
        "first_timestamp": _fmt(df["timestamp"].min()),
        "last_timestamp": _fmt(df["timestamp"].max()),
        "services": sorted(df["service"].unique().tolist()),
        "total_rows": int(len(df)),
        "columns": df.columns.drop(["is_error", "hour"]).tolist(),
        "error_definition": "A request is an error when status_code >= 400.",
    }


# --------------------------------------------------------------------------
# Tool 2: statistics
# --------------------------------------------------------------------------
def error_rate_by_hour(service: str, date: str | None = None) -> dict:
    """Hourly request count and error rate (%) for one service.

    Args:
        service: Service name, e.g. 'payment-api'.
        date: Optional day filter in YYYY-MM-DD format. Omit for all days.
    """
    sub, err = _service_df(service)
    if err:
        return err
    if date:
        sub = sub[sub["timestamp"].dt.strftime("%Y-%m-%d") == date]
        if sub.empty:
            return {"error": f"No rows for {service} on {date}."}
    g = sub.groupby("hour").agg(requests=("is_error", "size"), errors=("is_error", "sum"))
    g["error_rate_pct"] = (g["errors"] / g["requests"] * 100).round(2)
    hourly = [
        {"hour": _fmt(h), "requests": int(r.requests), "errors": int(r.errors), "error_rate_pct": float(r.error_rate_pct)}
        for h, r in g.iterrows()
    ]
    return {"service": service, "hourly": hourly}


def detect_spike(service: str) -> dict:
    """Find hours where a service's error rate is abnormally high versus its own normal level.

    The baseline is the median hourly error rate (a big spike cannot distort a median).
    An hour is flagged only when BOTH are true:
      1. its error count is far above what random chance would give at that traffic level
         (binomial z-score > 5), and
      2. its error rate is at least 3 percentage points above the baseline.
    Returns contiguous spike windows with start, end (exclusive) and peak error rate.
    Use this to find WHEN an incident happened before digging into WHAT happened.
    Several windows can be returned; compare their size and duration to decide which matters most.

    Args:
        service: Service name, e.g. 'payment-api'.
    """
    sub, err = _service_df(service)
    if err:
        return err
    hourly = sub.groupby("hour")["is_error"].agg(n="size", k="sum")
    g = hourly["k"] / hourly["n"] * 100  # error rate % per hour
    baseline = float(g.median())
    p0 = max(baseline / 100, 0.005)  # baseline as a fraction (floored to avoid dividing by ~0)
    # how many standard deviations above "chance alone" is each hour?
    z = (hourly["k"] - hourly["n"] * p0) / np.sqrt(hourly["n"] * p0 * (1 - p0))
    flagged = (z > 5) & (g > baseline + 3)

    windows, current = [], []
    for hour, is_flag in flagged.items():
        if is_flag:
            current.append(hour)
        elif current:
            windows.append(current)
            current = []
    if current:
        windows.append(current)

    out = []
    for w in windows:
        peak_hour = g.loc[w].idxmax()
        out.append(
            {
                "start": _fmt(w[0]),
                "end_exclusive": _fmt(w[-1] + pd.Timedelta(hours=1)),
                "hours": len(w),
                "peak_hour": _fmt(peak_hour),
                "peak_error_rate_pct": round(float(g.loc[peak_hour]), 2),
                "avg_error_rate_pct": round(float(g.loc[w].mean()), 2),
                "total_errors": int(hourly.loc[w, "k"].sum()),
            }
        )
    return {
        "service": service,
        "baseline_error_rate_pct": round(baseline, 2),
        "spike_windows": out,
        "note": "No abnormal spike found." if not out else f"{len(out)} spike window(s) found.",
    }


# --------------------------------------------------------------------------
# Tool 3: drill down
# --------------------------------------------------------------------------
def top_errors(service: str, start: str, end: str) -> dict:
    """Break down the errors of one service inside a time window.

    Returns error counts by type and status code, which deploy versions were running,
    and average latency for failed vs successful requests. Use this on a spike window
    to see WHAT is failing.

    Args:
        service: Service name, e.g. 'payment-api'.
        start: Window start, e.g. '2026-11-02 14:00' (inclusive).
        end: Window end, e.g. '2026-11-02 18:00' (exclusive).
    """
    sub, err = _service_df(service)
    if err:
        return err
    try:
        t0, t1 = pd.Timestamp(start), pd.Timestamp(end)
    except (ValueError, TypeError):
        return {"error": "start/end must look like '2026-11-02 14:00'."}
    w = sub[(sub["timestamp"] >= t0) & (sub["timestamp"] < t1)]
    if w.empty:
        return {"error": f"No rows for {service} between {start} and {end}."}
    errs, oks = w[w["is_error"]], w[~w["is_error"]]
    return {
        "service": service,
        "window": f"{start} to {end}",
        "total_requests": int(len(w)),
        "error_count": int(len(errs)),
        "error_rate_pct": round(len(errs) / len(w) * 100, 2),
        "error_types": errs["error_type"].value_counts().to_dict(),
        "status_codes": {str(k): int(v) for k, v in errs["status_code"].value_counts().items()},
        "deploy_versions_running": w["deploy_version"].value_counts().to_dict(),
        "avg_latency_ms_errors": round(float(errs["latency_ms"].mean()), 1) if len(errs) else None,
        "avg_latency_ms_success": round(float(oks["latency_ms"].mean()), 1) if len(oks) else None,
    }


def deploy_history(service: str) -> dict:
    """When each deploy version of a service was first and last seen in the logs.

    Use this to check whether an incident started right after a new version went live.

    Args:
        service: Service name, e.g. 'payment-api'.
    """
    sub, err = _service_df(service)
    if err:
        return err
    g = sub.groupby("deploy_version")["timestamp"].agg(["min", "max"]).sort_values("min")
    return {
        "service": service,
        "versions": [
            {"version": v, "first_seen": _fmt(r["min"]), "last_seen": _fmt(r["max"])} for v, r in g.iterrows()
        ],
    }


# --------------------------------------------------------------------------
# Tool 4: outside knowledge
# --------------------------------------------------------------------------
def web_search(query: str, max_results: int = 3) -> dict:
    """Search the web (DuckDuckGo) for known causes, fixes or documentation.

    Use this AFTER you know the error pattern, e.g. 'database connection pool exhausted
    after deploy 503 timeout'. Do not search for the user's private data; it is not on the web.

    Args:
        query: A short, specific search query.
        max_results: How many results to return (1-5).
    """
    try:
        from ddgs import DDGS  # imported here so the other tools work without it
    except ImportError:
        return {"error": "ddgs is not installed. Run: pip install ddgs"}
    try:
        hits = DDGS().text(query, max_results=max(1, min(int(max_results), 5)))
    except Exception as exc:  # network blocked, rate limited, etc.
        return {"error": f"Web search failed: {exc}. Continue using the log evidence only."}
    return {
        "query": query,
        "results": [
            {"title": h.get("title", ""), "url": h.get("href", ""), "snippet": (h.get("body", "") or "")[:300]}
            for h in hits
        ],
    }


ALL_TOOLS = [get_data_overview, error_rate_by_hour, detect_spike, top_errors, deploy_history, web_search]
