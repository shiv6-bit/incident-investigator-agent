#Incident Investigator Agent

An autonomous agent that investigates a production incident from a plain-English question and returns a structured, validated report: probable root cause, evidence, what was ruled out, confidence, and next steps.

> "Why did our payment API errors spike yesterday afternoon?" -> the agent finds the spike, drills into the errors, checks deploys, searches the web for known causes, and writes the report.

Built with PydanticAI (agent loop + typed output) and the Groq API using a Qwen model. The project uses synthetic logs and free-tier API access where available.

#The problem

When something breaks in production, an on-call engineer spends the first 20-30 minutes on repetitive detective work: when did it start? which service? what kind of errors? did we just deploy? has anyone seen this before?* This agent automates that first pass so a human starts with evidence instead of a blank page.

## How it works

```
 question ──► Agent (LLM + PydanticAI) ◄──── system prompt: investigation playbook + rules
                 │   ▲
     picks a tool│   │tool result (JSON)          loop until enough evidence
                 ▼   │
   ┌────────────────────────────────────────┐
   │ get_data_overview   what data exists   │
   │ error_rate_by_hour  hourly stats       │   logs.csv (synthetic)
   │ detect_spike        WHEN was it bad    │◄──────────────────────
   │ top_errors          WHAT was failing   │
   │ deploy_history      did a deploy land? │
   │ web_search          known causes/fixes │◄── DuckDuckGo (free)
   └────────────────────────────────────────┘
                 │
                 ▼
        IncidentReport (Pydantic model)  ── validated; the model is re-asked if invalid
```

| Tool | Type | Why it exists |
|---|---|---|
| `get_data_overview` | data | lets the agent orient itself (services, dates) instead of guessing |
| `error_rate_by_hour` | statistics | hourly error rate for a service |
| `detect_spike` | statistics | finds *when* a service was abnormal (binomial z-score + minimum lift) |
| `top_errors` | drill-down | error types, status codes, latency, deploy versions in a window |
| `deploy_history` | drill-down | tests the "did a deploy cause it?" hypothesis |
| `web_search` | external | outside knowledge on known causes and fixes |

## Design decisions

- The LLM decides the order of tool calls; the code decides what the tools compute. Numbers come from deterministic pandas code, never from the LLM, so it cannot hallucinate statistics.
- Typed output with Pydantic. The report must match `IncidentReport`; `confidence` can only be `low | medium | high`. Downstream systems (tickets, Slack) can rely on the shape.
- Robust spike detection. An hourly error rate is noisy when traffic is low. `detect_spike` compares each hour to what *random chance* would produce at that traffic level (binomial z-score > 5) and also requires a lift of at least 3 percentage points. My first version (median/MAD) flagged a random 1-hour blip, which is why I changed it.
- Guardrails.A 15-request cap per question stops runaway loops; tools return `{"error": ...}` instead of crashing so the model can self-correct; the prompt forbids invented numbers and warns that correlation is not proof.
- Synthetic data with a hidden story (including a red herring) so the agent's reasoning can be checked against a known answer.

## Setup

Requires Python 3.10+.

```bash
git clone https://github.com/shiv6-bit/incident-investigator-agent.git
cd incident-investigator-agent

python -m venv .venv
# Windows:      .venv\Scripts\activate
# Mac / Linux:  source .venv/bin/activate

pip install -r requirements.txt
```
Get a Groq API key from https://console.groq.com/keys.

Create your `.env` file:

Windows:
copy .env.example .env

Mac/Linux:
cp .env.example .env

Open `.env` and add:

GROQ_API_KEY=your_groq_api_key
MODEL=groq:qwen/qwen3.8-27b
```

#Run

```bash
python generate_data.py       # creates data/logs.csv (already included; deterministic)
python agent.py               # runs the default question
python agent.py "Is auth-service healthy? Anything unusual on Nov 2?"
python agent.py --json "..."  # raw JSON report
pytest -q                     # tests for the tools (no API key or internet needed)
```

The output shows each tool call the agent made, then the final report.
## Example run

Question: Why did our payment API errors spike November 2 afternoon, and what should we do?

The agent investigated the logs and found an issue with the payment-api service on 2 November 2026, between 14:00 and 18:00.

The error rate increased to around 26–31%, with a peak of 30.69% at 15:00.

The main findings were:
- DB_CONNECTION_TIMEOUT: 68 errors
- UPSTREAM_TIMEOUT: 33 errors
- v2.4.1 was deployed at 13:45, shortly before the errors started.
- The auth-service spike at 03:00 was unrelated to this incident.
- Other services did not show any abnormal spike.

The agent assigned medium confidence because the deployment timing suggests a possible connection, but there is not enough evidence to prove that the deployment caused the issue.

It recommended rolling back to v2.4.0, checking database connection settings and monitoring database health before deploying again.

The investigation used data overview, spike detection, error analysis, deployment history and web search tools.

#The synthetic scenario (so you can check the agent's answer)

- `payment-api` deploys **v2.4.1 at 13:45** on 2026-11-02.
- From **14:00 to 18:00** its error rate jumps from ~1.5% to ~26%, mostly `DB_CONNECTION_TIMEOUT` (HTTP 503), with latency about 4x higher.
- `auth-service` has a short 1-hour `RATE_LIMITED` bump at 03:00 (a red herring).
- `search-service` and `notification-service` are normal.

A good answer: the payment-api incident starts right after the v2.4.1 deploy, and the error type points at database connection exhaustion. The auth bump is dismissed as minor. Confidence is stated honestly (the logs show correlation and a matching error type, not proof).

## Limitations and next steps

**Main limitation:** it only sees one log table with fixed columns. It cannot read real infrastructure metrics (DB connection counts, CPU), so "database connection pool exhaustion" remains a well-supported hypothesis rather than a verified root cause, and the agent can only say so.

**What I would improve next:**
1. Connect real sources (log store, metrics, deploy API) as additional tools.
2. Add a second "reviewer" step that challenges the first conclusion before reporting (explicit state machine in LangGraph).
3. Build an evaluation set of several seeded incidents and measure how often the agent finds the right cause.
4. Cross-service reasoning (does an upstream failure explain downstream errors?).

## Project layout

```
agent.py          agent definition, output schema, CLI
tools.py          the six tools (plain Python functions)
generate_data.py  creates the synthetic logs
data/logs.csv     generated logs (~22k rows)
tests/            unit tests for the tools
```
