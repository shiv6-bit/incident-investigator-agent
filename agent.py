"""Incident Investigator Agent.

Ask a plain-English question about an incident. The agent decides which tools to
call (log stats, drill-down, deploy history, web search), loops until it has
enough evidence, and returns a structured, validated report.

Usage:
    python agent.py                       # runs the default demo question
    python agent.py "Why did checkout fail yesterday afternoon?"
    python agent.py --json "..."          # print the raw JSON report
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from typing import Literal

from dotenv import load_dotenv
from pydantic import BaseModel, Field
from pydantic_ai import Agent
from pydantic_ai.exceptions import ModelHTTPError, UsageLimitExceeded
from pydantic_ai.messages import ToolCallPart
from pydantic_ai.usage import UsageLimits

import tools

load_dotenv()

# Free Gemini model. Change in .env (MODEL=...) if you hit rate limits or the name changes.
MODEL = os.getenv("MODEL", "google:gemini-3.7-flash")
# PydanticAI's Google provider reads GEMINI_API_KEY (older) or GOOGLE_API_KEY (newer).
if os.getenv("GEMINI_API_KEY") and not os.getenv("GOOGLE_API_KEY"):
    os.environ["GOOGLE_API_KEY"] = os.environ["GEMINI_API_KEY"]

DEFAULT_QUESTION = "Why did our payment API errors spike yesterday afternoon, and what should we do?"


# ---------------------------------------------------------------------------
# 1. The structured output. PydanticAI forces the LLM's answer into this shape
#    and re-asks the model if it does not validate.
# ---------------------------------------------------------------------------
class IncidentReport(BaseModel):
    affected_service: str
    incident_window: str = Field(description="Start and end time of the incident, from the logs")
    summary: str = Field(description="2-3 sentence plain-English summary")
    likely_root_cause: str
    evidence: list[str] = Field(description="Specific numbers from the tool results that support the cause")
    ruled_out: list[str] = Field(description="Other anomalies or hypotheses checked and why they were dismissed")
    confidence: Literal["low", "medium", "high"]
    confidence_reason: str = Field(description="Why this confidence level; mention what is correlation vs proven")
    next_steps: list[str] = Field(description="Concrete actions for the on-call engineer, most urgent first")


# ---------------------------------------------------------------------------
# 2. The agent: model + instructions + output type + tools.
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = """You are an on-call incident investigator for an engineering team.
You answer questions about outages using ONLY evidence from your tools.

Work like this:
1. Call get_data_overview first to learn the services and date range.
2. Use detect_spike to find WHEN error rates were abnormal. Check the service mentioned in the
   question, and check the other services too so you can rule them out.
3. Use top_errors on the spike window to find WHAT was failing.
4. Use deploy_history to see whether a new version went live just before the spike.
5. Use web_search ONCE, after you know the error pattern, to look for known causes and fixes.
6. Then give your final report.

Rules:
- Never invent numbers. Every item in `evidence` must come from a tool result.
- If several anomalies exist, say which one matters most and why; list the rest in `ruled_out`.
- Correlation is not proof. If you only have timing correlation (e.g. a deploy right before the spike),
  say so in confidence_reason and do not claim "high" confidence unless the error type itself points to the cause.
- Dates in questions like "yesterday" are relative to the LAST timestamp in the data, not today's real date.
"""

agent = Agent(MODEL, output_type=IncidentReport, system_prompt=SYSTEM_PROMPT, retries=2)
for fn in tools.ALL_TOOLS:
    agent.tool_plain(fn)  # register each function as a tool the LLM may call


# ---------------------------------------------------------------------------
# 3. Run it and show the agent's tool calls (great for the live demo).
# ---------------------------------------------------------------------------
def investigate(question: str, max_attempts: int = 3):
    for attempt in range(1, max_attempts + 1):
        try:
            # request_limit stops runaway loops: at most 15 LLM calls per question
            return agent.run_sync(question, usage_limits=UsageLimits(request_limit=15))
        except ModelHTTPError as exc:
            if exc.status_code == 429 and attempt < max_attempts:  # free-tier rate limit
                wait = 30 * attempt
                print(f"[rate limited by the free tier; retrying in {wait}s ...]", file=sys.stderr)
                time.sleep(wait)
                continue
            raise
        except UsageLimitExceeded:
            sys.exit("The agent used too many steps without finishing. Try a more specific question.")


def print_trace(result) -> None:
    print("\n=== Agent steps (tool calls) ===")
    step = 0
    for message in result.all_messages():
        for part in message.parts:
            if isinstance(part, ToolCallPart) and not part.tool_name.startswith("final_result"):
                step += 1
                print(f"{step}. {part.tool_name}({part.args})")
    print(f"\nTotal LLM requests: {result.usage().requests}")


def print_report(r: IncidentReport) -> None:
    print("\n=== Incident Report ===")
    print(f"Service:    {r.affected_service}")
    print(f"Window:     {r.incident_window}")
    print(f"Confidence: {r.confidence.upper()} - {r.confidence_reason}\n")
    print(f"Summary:\n  {r.summary}\n")
    print(f"Likely root cause:\n  {r.likely_root_cause}\n")
    for title, items in (("Evidence", r.evidence), ("Ruled out", r.ruled_out), ("Next steps", r.next_steps)):
        print(f"{title}:")
        for i, item in enumerate(items, 1):
            print(f"  {i}. {item}")
        print()


def main() -> None:
    parser = argparse.ArgumentParser(description="Incident Investigator Agent")
    parser.add_argument("question", nargs="?", default=DEFAULT_QUESTION)
    parser.add_argument("--json", action="store_true", help="print the report as JSON only")
    args = parser.parse_args()

    if not os.getenv("GROQ_API_KEY"):
        sys.exit("Missing GROQ_API_KEY. Add your Groq API key to the .env file.")

    result = investigate(args.question)
    if args.json:
        print(result.output.model_dump_json(indent=2))
    else:
        print(f"Question: {args.question}")
        print_trace(result)
        print_report(result.output)


if __name__ == "__main__":
    main()
