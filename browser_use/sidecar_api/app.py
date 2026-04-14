from __future__ import annotations

import json
import time
from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, ConfigDict
from pydantic.alias_generators import to_camel

try:
    from uuid_extensions import uuid7str
except ImportError:
    import uuid
    def uuid7str() -> str:
        return str(uuid.uuid4())

app = FastAPI(title="Browser Use Sidecar", version="0.1.0")

# ---------------------------------------------------------------------------
# In-memory stores (reset per test via fixture, cleared on process restart)
# ---------------------------------------------------------------------------
_training_sessions: dict[str, dict] = {}
_compiled_skills: dict[str, dict] = {}


# ---------------------------------------------------------------------------
# Shared model config: camelCase aliases for .NET interop
# ---------------------------------------------------------------------------
class _CamelModel(BaseModel):
    model_config = ConfigDict(
        alias_generator=to_camel,
        populate_by_name=True,
    )


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------
class TrainStartRequest(_CamelModel):
    goal: str
    skill_id: str
    start_url: str


class TrainStartResponse(_CamelModel):
    training_session_id: str
    status: str = "recording"


class RawEvent(_CamelModel):
    type: str
    url: str | None = None
    selector: str | None = None
    field: str | None = None


class TrainCompleteRequest(_CamelModel):
    training_session_id: str
    raw_events: list[RawEvent] = Field(default_factory=list)


class CompiledStep(_CamelModel):
    action: str
    selector: str | None = None
    field: str | None = None
    url: str | None = None
    is_terminal_write: bool = False


class CompiledImplementation(_CamelModel):
    skill_id: str
    goal: str
    start_url: str
    steps: list[CompiledStep]


class TrainCompleteResponse(_CamelModel):
    training_session_id: str
    raw_trace_json: str
    compiled_implementation_json: str


class TaskRequest(_CamelModel):
    compiled_implementation_json: str
    mode: str = "dry_run"  # "dry_run" | "execute"
    input_payload_json: str  # JSON array of dicts


class ItemResult(_CamelModel):
    index: int
    status: str  # Created | Updated | Deleted | Skipped | Failed
    detail: str = ""


class TaskStats(_CamelModel):
    total: int
    duration_seconds: float
    mode: str
    counts: dict[str, int]


class TaskResponse(_CamelModel):
    status: str
    summary: str
    stats: TaskStats
    item_results: list[ItemResult]


# ---------------------------------------------------------------------------
# Pluggable seams — replace these to wire a real BrowserSession
# ---------------------------------------------------------------------------
def _record_trace(session: dict, raw_events: list[RawEvent]) -> list[dict]:
    """Convert raw events list to a serialisable trace."""
    return [e.model_dump(by_alias=False) for e in raw_events]


def _compile_implementation(session: dict, trace: list[dict]) -> CompiledImplementation:
    """Turn a raw trace into a reusable compiled implementation."""
    steps: list[CompiledStep] = []
    for event in trace:
        t = event.get("type", "")
        if t == "navigate":
            steps.append(CompiledStep(action="navigate", url=event.get("url"), is_terminal_write=False))
        elif t == "input":
            steps.append(CompiledStep(
                action="fill",
                selector=event.get("selector"),
                field=event.get("field"),
                is_terminal_write=False,
            ))
        elif t in ("save", "submit"):
            steps.append(CompiledStep(
                action="click",
                selector=event.get("selector"),
                is_terminal_write=True,
            ))
        elif t == "delete":
            steps.append(CompiledStep(
                action="click",
                selector=event.get("selector"),
                is_terminal_write=True,
            ))
    return CompiledImplementation(
        skill_id=session["skill_id"],
        goal=session["goal"],
        start_url=session["start_url"],
        steps=steps,
    )


def _execute_step(
    step: CompiledStep,
    item: dict,
    mode: str,
    existing: bool,
) -> str:
    """
    Stub: execute a single compiled step against a real browser.
    Returns a status string: Created | Updated | Deleted | Skipped.

    Wire this to a real BrowserSession (Playwright / browser-use) when ready.
    Until then it short-circuits terminal writes in dry_run mode and returns
    synthetic statuses in execute mode.
    """
    if step.is_terminal_write and mode == "dry_run":
        return "Skipped"
    if step.is_terminal_write:
        # Determine terminal outcome
        if step.action == "click" and step.selector and "delete" in (step.selector or "").lower():
            return "Deleted"
        return "Updated" if existing else "Created"
    return "Skipped"


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------
@app.post("/api/browser/train-start", response_model=TrainStartResponse)
def train_start(req: TrainStartRequest):
    session_id = uuid7str()
    _training_sessions[session_id] = {
        "skill_id": req.skill_id,
        "goal": req.goal,
        "start_url": req.start_url,
    }
    return TrainStartResponse(training_session_id=session_id)


@app.post("/api/browser/train-complete", response_model=TrainCompleteResponse)
def train_complete(req: TrainCompleteRequest):
    session = _training_sessions.get(req.training_session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Training session not found")

    trace = _record_trace(session, req.raw_events)
    impl = _compile_implementation(session, trace)
    impl_json = impl.model_dump_json(by_alias=True)

    _compiled_skills[impl.skill_id] = json.loads(impl_json)

    return TrainCompleteResponse(
        training_session_id=req.training_session_id,
        raw_trace_json=json.dumps(trace),
        compiled_implementation_json=impl_json,
    )


@app.post("/api/browser/tasks", response_model=TaskResponse)
def run_tasks(req: TaskRequest):
    impl = CompiledImplementation.model_validate_json(req.compiled_implementation_json)
    items: list[dict] = json.loads(req.input_payload_json)

    start = time.monotonic()
    results: list[ItemResult] = []
    counts: dict[str, int] = {}

    for idx, item in enumerate(items):
        try:
            last_status = "Skipped"
            existing = bool(item.get("_existing", False))
            for step in impl.steps:
                # Substitute placeholders in selector/url
                resolved_step = step.model_copy(update={
                    "selector": step.selector.format(**item) if step.selector else step.selector,
                    "url": step.url.format(**item) if step.url else step.url,
                })
                s = _execute_step(resolved_step, item, req.mode, existing)
                if s not in ("Skipped",):
                    last_status = s
            if last_status == "Skipped":
                last_status = "Skipped"
        except Exception as exc:
            last_status = "Failed"
            results.append(ItemResult(index=idx, status="Failed", detail=str(exc)))
            counts["Failed"] = counts.get("Failed", 0) + 1
            continue

        results.append(ItemResult(index=idx, status=last_status))
        counts[last_status] = counts.get(last_status, 0) + 1

    duration = round(time.monotonic() - start, 3)
    total = len(items)
    summary = f"Processed {total} items in {duration}s ({req.mode}): " + ", ".join(
        f"{v} {k}" for k, v in counts.items()
    )

    return TaskResponse(
        status="completed",
        summary=summary,
        stats=TaskStats(total=total, duration_seconds=duration, mode=req.mode, counts=counts),
        item_results=results,
    )
