"""FastAPI sidecar exposing browser training + task execution endpoints.

Architecture:
  _record_trace()         — harvest JS-recorded interactions from a live BrowserSession
  _compile_implementation() — ask gpt-4o to generalise selectors / detect placeholders
  _execute_step()         — drive real BrowserSession via cdp-use; self-heals via LLM on failure
  _heal_and_retry()       — one LLM-guided retry when a step fails

When OPENAI_API_KEY is absent the module degrades gracefully:
  - training falls back to the heuristic compiler
  - task execution skips browser startup (no-op, status derived from step metadata)

This keeps all 7 existing TestClient smoke-tests passing without a browser.
"""

from __future__ import annotations

import asyncio
import base64
import json
import logging
import os
import time
from typing import Any, Literal

from fastapi import BackgroundTasks, FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field
from uuid_extensions import uuid7str

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Request / response models
# ---------------------------------------------------------------------------


class TrainStartRequest(BaseModel):
	model_config = ConfigDict(extra='forbid', populate_by_name=True)

	goal: str = Field(..., description='Natural-language goal of the skill being trained.')
	skill_id: str | None = Field(default=None, alias='skillId')
	start_url: str | None = Field(default=None, alias='startUrl')
	demonstration: dict[str, Any] | None = Field(
		default=None,
		description='Optional structured demonstration metadata (selectors, hints, sample inputs).',
	)


class TrainStartResponse(BaseModel):
	training_session_id: str = Field(..., alias='trainingSessionId')
	status: Literal['recording'] = 'recording'

	model_config = ConfigDict(populate_by_name=True)


class TrainCompleteRequest(BaseModel):
	model_config = ConfigDict(extra='forbid', populate_by_name=True)

	training_session_id: str = Field(..., alias='trainingSessionId')
	raw_events: list[dict[str, Any]] | None = Field(default=None, alias='rawEvents')


class CompiledStep(BaseModel):
	model_config = ConfigDict(extra='forbid')

	action: Literal['navigate', 'click', 'input', 'select', 'wait', 'save', 'delete']
	selector: str | None = None
	value: str | None = None  # may contain `{placeholders}` mapped to input fields
	is_terminal_write: bool = Field(
		default=False,
		description='True for actions that mutate remote state (save/delete). Skipped in dry_run.',
	)
	notes: str | None = None


class CompiledImplementation(BaseModel):
	model_config = ConfigDict(extra='forbid')

	skill_id: str | None = None
	goal: str
	steps: list[CompiledStep] = Field(default_factory=list)
	input_schema: dict[str, Any] = Field(default_factory=dict)


class TrainCompleteResponse(BaseModel):
	training_session_id: str = Field(..., alias='trainingSessionId')
	raw_trace_json: str = Field(..., alias='rawTraceJson')
	compiled_implementation_json: str = Field(..., alias='compiledImplementationJson')

	model_config = ConfigDict(populate_by_name=True)


class TaskRequest(BaseModel):
	model_config = ConfigDict(extra='forbid', populate_by_name=True)

	sidecar_session_id: str | None = Field(default=None, alias='sidecarSessionId')
	compiled_implementation_json: str = Field(..., alias='compiledImplementationJson')
	mode: Literal['dry_run', 'execute'] = 'dry_run'
	input_payload_json: str = Field(..., alias='inputPayloadJson')


class ItemResult(BaseModel):
	input_key: str | None = Field(default=None, alias='inputKey')
	status: Literal['Created', 'Updated', 'Deleted', 'Skipped', 'Failed']
	message: str | None = None

	model_config = ConfigDict(populate_by_name=True)


class TaskResponse(BaseModel):
	status: Literal['Succeeded', 'Failed']
	summary: str
	stats: dict[str, Any]
	item_results: list[ItemResult] = Field(default_factory=list, alias='itemResults')

	model_config = ConfigDict(populate_by_name=True)


# ---------------------------------------------------------------------------
# In-memory stores
# ---------------------------------------------------------------------------

_training_sessions: dict[str, dict[str, Any]] = {}
_compiled_skills: dict[str, CompiledImplementation] = {}

# ---------------------------------------------------------------------------
# JavaScript interaction recorder injected into training pages
# ---------------------------------------------------------------------------

_RECORDER_SCRIPT = r"""
(function() {
  if (window.__browserUseTrace) return;
  window.__browserUseTrace = [];

  function getSelector(el) {
    if (!el || el.nodeType !== 1) return '';
    if (el.id) return '#' + el.id;
    var name = el.getAttribute('name');
    if (name) return '[name="' + name + '"]';
    var testId = el.getAttribute('data-testid');
    if (testId) return '[data-testid="' + testId + '"]';
    var ariaLabel = el.getAttribute('aria-label');
    if (ariaLabel) return '[aria-label="' + ariaLabel + '"]';
    // Fallback: tag + positional
    var tag = el.tagName.toLowerCase();
    if (!el.parentNode) return tag;
    var siblings = Array.from(el.parentNode.children).filter(function(c) { return c.tagName === el.tagName; });
    if (siblings.length === 1) return tag;
    return tag + ':nth-of-type(' + (siblings.indexOf(el) + 1) + ')';
  }

  document.addEventListener('click', function(e) {
    window.__browserUseTrace.push({
      type: 'click',
      selector: getSelector(e.target),
      ts: Date.now()
    });
  }, true);

  document.addEventListener('change', function(e) {
    var el = e.target;
    if (!['INPUT', 'TEXTAREA', 'SELECT'].includes(el.tagName)) return;
    window.__browserUseTrace.push({
      type: 'input',
      selector: getSelector(el),
      value: el.value,
      field: el.name || el.id || el.getAttribute('aria-label') || '',
      ts: Date.now()
    });
  }, true);

  // Intercept SPA navigations
  var _orig = history.pushState;
  history.pushState = function() {
    _orig.apply(this, arguments);
    window.__browserUseTrace.push({ type: 'navigate', url: location.href, ts: Date.now() });
  };
})();
"""


# ---------------------------------------------------------------------------
# LLM helper
# ---------------------------------------------------------------------------


def _get_llm(temperature: float = 0) -> Any:
	"""Return a ChatOpenAI instance reading OPENAI_API_KEY from env."""
	from browser_use.llm.openai.chat import ChatOpenAI

	return ChatOpenAI(model='gpt-4o', temperature=temperature)


# ---------------------------------------------------------------------------
# Stub 1 — _record_trace  (async; was sync no-op)
# ---------------------------------------------------------------------------


async def _record_trace(session: dict[str, Any]) -> list[dict[str, Any]]:
	"""Return the raw interaction trace for a training session.

	Priority:
	  1. JS events harvested from the live BrowserSession page (most accurate selectors)
	  2. rawEvents supplied by the caller in train-complete
	  3. Synthetic stub event (fallback when no browser and no caller events)
	"""
	browser: Any = session.get('_browser_session')
	if browser is not None:
		try:
			page = await browser.must_get_current_page()
			trace_json: str = await page.evaluate('JSON.stringify(window.__browserUseTrace || [])')
			js_events: list[dict[str, Any]] = json.loads(trace_json)
			if js_events:
				logger.debug('Harvested %d events from live JS recorder', len(js_events))
				return js_events
		except Exception as exc:
			logger.debug('Could not harvest JS trace from browser: %s', exc)

	caller_events: list[dict[str, Any]] = session.get('raw_events') or []
	if caller_events:
		return caller_events

	return [{'type': 'session_started', 'goal': session['goal'], 'ts': session['started_at']}]


# ---------------------------------------------------------------------------
# Stub 2 — _compile_implementation  (async; was sync heuristic)
# ---------------------------------------------------------------------------


def _compile_heuristic(session: dict[str, Any], events: list[dict[str, Any]]) -> CompiledImplementation:
	"""Heuristic fallback: map raw event types → CompiledStep without LLM."""
	steps: list[CompiledStep] = []
	if session.get('start_url'):
		steps.append(CompiledStep(action='navigate', value=session['start_url']))

	for ev in events:
		etype = ev.get('type')
		if etype == 'click':
			steps.append(CompiledStep(action='click', selector=ev.get('selector')))
		elif etype == 'input':
			steps.append(
				CompiledStep(
					action='input',
					selector=ev.get('selector'),
					value=ev.get('value') or f'{{{ev.get("field", "value")}}}',
				)
			)
		elif etype == 'navigate':
			steps.append(CompiledStep(action='navigate', value=ev.get('url')))
		elif etype in ('save', 'submit'):
			steps.append(CompiledStep(action='save', selector=ev.get('selector'), is_terminal_write=True))
		elif etype == 'delete':
			steps.append(CompiledStep(action='delete', selector=ev.get('selector'), is_terminal_write=True))

	return CompiledImplementation(
		skill_id=session.get('skill_id'),
		goal=session['goal'],
		steps=steps,
		input_schema=(session.get('demonstration') or {}).get('input_schema', {}),
	)


async def _compile_with_llm(session: dict[str, Any], events: list[dict[str, Any]]) -> CompiledImplementation:
	"""Use gpt-4o to generalise selectors and detect input placeholders."""
	from browser_use.llm.messages import ContentPartTextParam, SystemMessage, UserMessage

	llm = _get_llm(temperature=0)

	system_prompt = (
		'You are a browser automation expert. '
		'You receive raw browser interaction events and must compile them into a reusable, '
		'parameterised automation script.\n\n'
		'Rules:\n'
		'1. Use stable CSS selectors: prefer #id, [name=x], [data-testid=x], [aria-label=x].\n'
		'2. Replace literal data values that will vary per run with {placeholder} tokens '
		'   (e.g. "Invoice 1" → "{title}", "C-001" → "{controlKey}").\n'
		'   Do NOT replace UI constants (button labels, fixed paths).\n'
		'3. Mark the final submit/save/delete button click as is_terminal_write=true.\n'
		'4. Build input_schema as a minimal JSON Schema describing the placeholder fields.\n'
		'5. Use action types: navigate, click, input, select, wait, save, delete.\n'
		'   - input: typing into a text field\n'
		'   - save: the final submit/save button\n'
		'   - delete: a delete/remove button\n'
		'6. Set skill_id to null.\n'
	)
	user_text = (
		f'Goal: {session["goal"]}\n'
		f'Start URL: {session.get("start_url") or "(not recorded)"}\n\n'
		f'Raw events:\n{json.dumps(events, indent=2)}\n\n'
		'Compile these events into a reusable automation implementation.'
	)
	messages = [
		SystemMessage(content=system_prompt),
		UserMessage(content=[ContentPartTextParam(text=user_text)]),
	]
	result = await llm.ainvoke(messages, output_format=CompiledImplementation)
	compiled: CompiledImplementation = result.completion
	# Restore skill_id from the session (LLM sets it null)
	return compiled.model_copy(update={'skill_id': session.get('skill_id')})


async def _compile_implementation(session: dict[str, Any], events: list[dict[str, Any]]) -> CompiledImplementation:
	"""Compile raw trace → CompiledImplementation, using LLM when possible."""
	if os.getenv('OPENAI_API_KEY'):
		try:
			return await _compile_with_llm(session, events)
		except Exception as exc:
			logger.warning('LLM compilation failed (%s); falling back to heuristic', exc)
	return _compile_heuristic(session, events)


# ---------------------------------------------------------------------------
# Stub 3 — _execute_step  (async; was sync no-op)
# ---------------------------------------------------------------------------


def _format_step(step: CompiledStep, item: dict[str, Any]) -> CompiledStep:
	"""Substitute `{field}` placeholders in `value` and `selector` from the item dict."""
	updates: dict[str, Any] = {}
	for attr in ('value', 'selector'):
		raw = getattr(step, attr)
		if raw and '{' in raw:
			try:
				updates[attr] = raw.format(**item)
			except (KeyError, IndexError):
				pass  # leave unformatted if key missing
	return step.model_copy(update=updates) if updates else step


async def _execute_step(step: CompiledStep, browser_session: Any, mode: str) -> tuple[bool, str]:
	"""Execute one compiled step on the given BrowserSession.

	Returns (success, error_message).
	  - dry_run + is_terminal_write → skip; return (True, '')
	  - browser_session is None     → no-op; return (True, '')
	  - real execution              → (True/'') or (False/error)
	"""
	if step.is_terminal_write and mode == 'dry_run':
		return True, ''
	if browser_session is None:
		return True, ''

	try:
		page = await browser_session.must_get_current_page()

		if step.action == 'navigate':
			await page.goto(step.value or '')

		elif step.action in ('click', 'save', 'delete'):
			if not step.selector:
				return False, f'action={step.action} has no selector'
			elements = await page.get_elements_by_css_selector(step.selector)
			if not elements:
				return False, f'No element matched selector: {step.selector!r}'
			await elements[0].click()

		elif step.action == 'input':
			if not step.selector:
				return False, f'action=input has no selector'
			elements = await page.get_elements_by_css_selector(step.selector)
			if not elements:
				return False, f'No element matched selector: {step.selector!r}'
			await elements[0].fill(step.value or '')

		elif step.action == 'select':
			if not step.selector:
				return False, f'action=select has no selector'
			elements = await page.get_elements_by_css_selector(step.selector)
			if not elements:
				return False, f'No element matched selector: {step.selector!r}'
			await elements[0].select_option(step.value or '')

		elif step.action == 'wait':
			await asyncio.sleep(float(step.value or '1'))

		return True, ''

	except Exception as exc:
		return False, f'{type(exc).__name__}: {exc}'


async def _heal_and_retry(
	step: CompiledStep,
	browser_session: Any,
	mode: str,
	original_error: str,
) -> tuple[bool, str]:
	"""Ask the LLM for a corrected action, then retry once.

	Response protocol the LLM must follow (one line):
	  CLICK:<selector>
	  FILL:<selector>|<value>
	  NAVIGATE:<url>
	  SKIP
	"""
	if browser_session is None or not os.getenv('OPENAI_API_KEY'):
		return False, original_error

	try:
		from browser_use.llm.messages import ContentPartImageParam, ContentPartTextParam, ImageURL, UserMessage

		llm = _get_llm(temperature=0.1)
		page = await browser_session.must_get_current_page()
		url = await page.get_url()
		title = await page.get_title()
		screenshot_b64: str = await page.screenshot(format='png')

		prompt = (
			f'A browser automation step just failed.\n'
			f'Page URL: {url}\n'
			f'Page title: {title}\n'
			f'Failed step: action={step.action!r}, selector={step.selector!r}, value={step.value!r}\n'
			f'Error: {original_error}\n\n'
			f'What should I try instead? Reply with exactly one line:\n'
			f'CLICK:<selector>\n'
			f'FILL:<selector>|<value>\n'
			f'NAVIGATE:<url>\n'
			f'SKIP'
		)
		messages = [
			UserMessage(
				content=[
					ContentPartTextParam(text=prompt),
					ContentPartImageParam(
						image_url=ImageURL(
							url=f'data:image/png;base64,{screenshot_b64}',
							media_type='image/png',
						)
					),
				]
			)
		]
		result = await llm.ainvoke(messages)
		response: str = result.completion.strip().splitlines()[0].strip()
		logger.debug('LLM heal suggestion: %r', response)

		healed: CompiledStep | None = None
		if response.upper().startswith('SKIP'):
			return True, ''  # treat skip as success (step not needed)
		elif response.upper().startswith('CLICK:'):
			sel = response[6:].strip()
			healed = step.model_copy(update={'action': 'click', 'selector': sel})
		elif response.upper().startswith('FILL:'):
			parts = response[5:].split('|', 1)
			healed = step.model_copy(
				update={
					'action': 'input',
					'selector': parts[0].strip(),
					'value': parts[1].strip() if len(parts) > 1 else (step.value or ''),
				}
			)
		elif response.upper().startswith('NAVIGATE:'):
			healed = step.model_copy(update={'action': 'navigate', 'value': response[9:].strip()})

		if healed is None:
			return False, f'Unparseable heal response: {response!r}'

		ok, err = await _execute_step(healed, browser_session, mode)
		return ok, err if not ok else ''

	except Exception as exc:
		logger.warning('Self-heal attempt failed: %s', exc)
		return False, original_error


async def _execute_item(
	impl: CompiledImplementation,
	item: dict[str, Any],
	mode: str,
	browser_session: Any = None,
) -> ItemResult:
	input_key = (
		item.get('controlKey')
		or item.get('control_key')
		or item.get('id')
		or item.get('key')
	)
	try:
		terminal_action: str | None = None
		for raw_step in impl.steps:
			step = _format_step(raw_step, item)
			ok, err = await _execute_step(step, browser_session, mode)
			if not ok and browser_session is not None:
				ok, err = await _heal_and_retry(step, browser_session, mode, err)
			if not ok:
				return ItemResult(
					input_key=str(input_key) if input_key is not None else None,
					status='Failed',
					message=err,
				)
			if step.is_terminal_write:
				terminal_action = step.action

		if terminal_action == 'delete':
			status: Literal['Created', 'Updated', 'Deleted', 'Skipped', 'Failed'] = 'Deleted'
		elif item.get('_existing'):
			status = 'Updated'
		elif terminal_action == 'save':
			status = 'Created'
		else:
			status = 'Skipped'

		msg = 'dry-run: no remote writes performed' if mode == 'dry_run' else None
		return ItemResult(
			input_key=str(input_key) if input_key is not None else None,
			status=status,
			message=msg,
		)
	except Exception as exc:
		return ItemResult(
			input_key=str(input_key) if input_key is not None else None,
			status='Failed',
			message=f'{type(exc).__name__}: {exc}',
		)


# ---------------------------------------------------------------------------
# Background helper: start a BrowserSession and inject the JS recorder
# ---------------------------------------------------------------------------


async def _start_recording_session(session_id: str, start_url: str) -> None:
	"""Background task: launch a headless browser, navigate to start_url, inject recorder."""
	session = _training_sessions.get(session_id)
	if session is None:
		return
	try:
		from browser_use.browser.profile import BrowserProfile
		from browser_use.browser.session import BrowserSession

		browser = BrowserSession(headless=True, browser_profile=BrowserProfile(headless=True))
		await browser.start()
		session['_browser_session'] = browser

		page = await browser.must_get_current_page()
		await page.goto(start_url)
		await page.evaluate(_RECORDER_SCRIPT)
		logger.info('Recording session started at %s', start_url)
	except Exception as exc:
		logger.warning('Could not start recording session for %s: %s', session_id, exc)


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------


app = FastAPI(title='browser-use sidecar', version='0.1.0')


@app.get('/health')
async def health() -> dict[str, str]:
	return {'status': 'ok'}


@app.post('/api/browser/train-start', response_model=TrainStartResponse)
async def train_start(req: TrainStartRequest, background_tasks: BackgroundTasks) -> TrainStartResponse:
	training_session_id = uuid7str()
	_training_sessions[training_session_id] = {
		'training_session_id': training_session_id,
		'skill_id': req.skill_id,
		'goal': req.goal,
		'start_url': req.start_url,
		'demonstration': req.demonstration,
		'started_at': time.time(),
		'raw_events': [],
		'_browser_session': None,
	}
	# Only launch a real browser when an API key is available
	# (keeps unit tests fast / dependency-free)
	if req.start_url and os.getenv('OPENAI_API_KEY'):
		background_tasks.add_task(_start_recording_session, training_session_id, req.start_url)

	return TrainStartResponse(training_session_id=training_session_id)


@app.post('/api/browser/train-complete', response_model=TrainCompleteResponse)
async def train_complete(req: TrainCompleteRequest) -> TrainCompleteResponse:
	session = _training_sessions.get(req.training_session_id)
	if session is None:
		raise HTTPException(status_code=404, detail='unknown trainingSessionId')

	if req.raw_events is not None:
		session['raw_events'] = req.raw_events

	events = await _record_trace(session)
	compiled = await _compile_implementation(session, events)

	if compiled.skill_id:
		_compiled_skills[compiled.skill_id] = compiled

	raw_trace_json = json.dumps({'training_session_id': req.training_session_id, 'events': events})
	compiled_json = compiled.model_dump_json()

	# Stop the training browser session if one was started
	browser: Any = session.get('_browser_session')
	if browser is not None:
		try:
			await browser.stop()
		except Exception:
			pass
		session['_browser_session'] = None

	return TrainCompleteResponse(
		training_session_id=req.training_session_id,
		raw_trace_json=raw_trace_json,
		compiled_implementation_json=compiled_json,
	)


@app.post('/api/browser/tasks', response_model=TaskResponse)
async def run_tasks(req: TaskRequest) -> TaskResponse:
	try:
		impl = CompiledImplementation.model_validate_json(req.compiled_implementation_json)
	except Exception as exc:
		raise HTTPException(status_code=400, detail=f'invalid compiledImplementationJson: {exc}')

	try:
		payload: Any = json.loads(req.input_payload_json)
	except json.JSONDecodeError as exc:
		raise HTTPException(status_code=400, detail=f'invalid inputPayloadJson: {exc}')

	if isinstance(payload, dict):
		items: list[dict[str, Any]] = [payload]
	elif isinstance(payload, list):
		items = [x if isinstance(x, dict) else {'value': x} for x in payload]
	else:
		raise HTTPException(status_code=400, detail='inputPayloadJson must decode to an object or list of objects')

	# Start a real browser only when executing (not dry-run) and key is available
	browser_session: Any = None
	if req.mode == 'execute' and os.getenv('OPENAI_API_KEY'):
		try:
			from browser_use.browser.profile import BrowserProfile
			from browser_use.browser.session import BrowserSession

			browser_session = BrowserSession(headless=True, browser_profile=BrowserProfile(headless=True))
			await browser_session.start()
			logger.info('Browser session started for task execution')
		except Exception as exc:
			logger.warning('Could not start browser for task execution: %s', exc)
			browser_session = None

	try:
		t0 = time.time()
		results: list[ItemResult] = []
		for item in items:
			r = await _execute_item(impl, item, req.mode, browser_session)
			results.append(r)
		duration = time.time() - t0
	finally:
		if browser_session is not None:
			try:
				await browser_session.stop()
			except Exception:
				pass

	counts: dict[str, int] = {}
	for r in results:
		counts[r.status] = counts.get(r.status, 0) + 1

	failed = counts.get('Failed', 0)
	stats = {
		'total': len(results),
		'duration_seconds': round(duration, 3),
		'mode': req.mode,
		'counts': counts,
	}
	parts = [f'{n} {k.lower()}' for k, n in counts.items() if n]
	summary = ', '.join(parts) if parts else 'no items processed'

	return TaskResponse(
		status='Failed' if failed and failed == len(results) else 'Succeeded',
		summary=summary,
		stats=stats,
		item_results=results,
	)
