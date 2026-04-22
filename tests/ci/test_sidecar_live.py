"""Live integration test for the browser sidecar.

Skipped automatically when ANTHROPIC_API_KEY is not set.

Tests the full pipeline:
  1. POST /api/browser/train-start  → creates a session
  2. POST /api/browser/train-complete (with rawEvents) → LLM compiles the trace
  3. POST /api/browser/tasks (execute mode) → real browser drives a local form

Uses pytest-httpserver so no external network access is needed.
"""

from __future__ import annotations

import json
import os

import httpx
import pytest

# Skip the entire module when no API key is available
pytestmark = pytest.mark.skipif(
	not os.getenv('ANTHROPIC_API_KEY'),
	reason='ANTHROPIC_API_KEY not set — skipping live sidecar tests',
)


@pytest.fixture
async def async_client():
	"""Async ASGI test client for the sidecar app."""
	from browser_use.sidecar_api.app import app, _compiled_skills, _training_sessions

	_training_sessions.clear()
	_compiled_skills.clear()
	transport = httpx.ASGITransport(app=app)
	async with httpx.AsyncClient(transport=transport, base_url='http://test') as client:
		yield client


async def test_live_train_and_compile(httpserver, async_client):
	"""LLM compilation turns raw events into sensible CompiledImplementation."""
	# Simple form that accepts a POST and returns 200
	httpserver.expect_request('/form').respond_with_data(
		'<html><body>'
		'<form action="/submit" method="post">'
		'<input id="title" name="title" type="text" placeholder="Title"/>'
		'<input id="controlKey" name="controlKey" type="text" placeholder="Control Key"/>'
		'<button id="save" type="submit">Save</button>'
		'</form></body></html>',
		content_type='text/html',
	)
	httpserver.expect_request('/submit', method='POST').respond_with_data('OK', status=200)

	start_url = httpserver.url_for('/form')

	# --- train-start ---
	r = await async_client.post(
		'/api/browser/train-start',
		json={'goal': 'fill and submit a control form', 'skillId': 'live-skill', 'startUrl': start_url},
	)
	assert r.status_code == 200
	session_id = r.json()['trainingSessionId']
	assert session_id

	# --- train-complete with representative raw events ---
	r = await async_client.post(
		'/api/browser/train-complete',
		json={
			'trainingSessionId': session_id,
			'rawEvents': [
				{'type': 'navigate', 'url': start_url},
				{'type': 'input', 'selector': '#title', 'field': 'title', 'value': 'Sample Title'},
				{'type': 'input', 'selector': '#controlKey', 'field': 'controlKey', 'value': 'C-001'},
				{'type': 'save', 'selector': '#save'},
			],
		},
	)
	assert r.status_code == 200
	body = r.json()
	assert body['trainingSessionId'] == session_id

	raw = json.loads(body['rawTraceJson'])
	assert len(raw['events']) == 4

	compiled = json.loads(body['compiledImplementationJson'])
	actions = [s['action'] for s in compiled['steps']]
	assert len(actions) >= 2, 'Expected at least navigate + input/save steps'

	# LLM should have detected the save step as terminal
	terminal_steps = [s for s in compiled['steps'] if s.get('is_terminal_write')]
	assert terminal_steps, 'LLM should have flagged a terminal write step'

	# LLM should have templated at least one placeholder
	all_values = ' '.join(str(s.get('value') or '') for s in compiled['steps'])
	assert '{' in all_values, f'Expected placeholder tokens in step values; got: {all_values}'

	return body['compiledImplementationJson']


async def test_live_task_execute(httpserver, async_client):
	"""Full execute cycle: browser fills a form and submits it."""
	submitted: list[dict] = []

	def handle_submit(request):
		from werkzeug.wrappers import Response

		data = dict(request.form)
		submitted.append(data)
		return Response('OK', status=200, content_type='text/plain')

	httpserver.expect_request('/form').respond_with_data(
		'<html><body>'
		'<form action="/submit" method="post">'
		'<input id="title" name="title" type="text"/>'
		'<input id="controlKey" name="controlKey" type="text"/>'
		'<button id="save" type="submit">Save</button>'
		'</form></body></html>',
		content_type='text/html',
	)
	httpserver.expect_request('/submit', method='POST').respond_with_handler(handle_submit)

	start_url = httpserver.url_for('/form')

	# A hand-crafted compiled implementation (skip LLM for this test)
	compiled = {
		'goal': 'fill and submit control form',
		'skill_id': None,
		'steps': [
			{'action': 'navigate', 'value': start_url, 'is_terminal_write': False},
			{'action': 'input', 'selector': '#title', 'value': '{title}', 'is_terminal_write': False},
			{'action': 'input', 'selector': '#controlKey', 'value': '{controlKey}', 'is_terminal_write': False},
			{'action': 'save', 'selector': '#save', 'is_terminal_write': True},
		],
		'input_schema': {
			'type': 'object',
			'properties': {'title': {'type': 'string'}, 'controlKey': {'type': 'string'}},
			'required': ['title', 'controlKey'],
		},
	}

	items = [
		{'title': 'Widget A', 'controlKey': 'C-100'},
		{'title': 'Widget B', 'controlKey': 'C-101'},
	]

	r = await async_client.post(
		'/api/browser/tasks',
		json={
			'compiledImplementationJson': json.dumps(compiled),
			'mode': 'execute',
			'inputPayloadJson': json.dumps(items),
		},
	)
	assert r.status_code == 200
	body = r.json()

	assert body['status'] == 'Succeeded', f'Unexpected status: {body}'
	assert body['stats']['total'] == 2
	assert body['stats']['mode'] == 'execute'

	keys = {ir['inputKey'] for ir in body['itemResults']}
	assert keys == {'C-100', 'C-101'}

	for ir in body['itemResults']:
		assert ir['status'] in ('Created', 'Updated', 'Skipped'), f'Unexpected item status: {ir}'


async def test_live_task_dry_run_does_not_submit(httpserver, async_client):
	"""dry_run mode must NOT trigger the terminal save action."""
	submit_calls: list[bool] = []

	def handle_submit(request):
		from werkzeug.wrappers import Response

		submit_calls.append(True)
		return Response('OK', status=200, content_type='text/plain')

	httpserver.expect_request('/form').respond_with_data(
		'<html><body>'
		'<form action="/submit" method="post">'
		'<input id="val" name="val" type="text"/>'
		'<button id="save" type="submit">Save</button>'
		'</form></body></html>',
		content_type='text/html',
	)
	httpserver.expect_request('/submit', method='POST').respond_with_handler(handle_submit)

	start_url = httpserver.url_for('/form')
	compiled = {
		'goal': 'test dry run',
		'skill_id': None,
		'steps': [
			{'action': 'navigate', 'value': start_url, 'is_terminal_write': False},
			{'action': 'input', 'selector': '#val', 'value': '{val}', 'is_terminal_write': False},
			{'action': 'save', 'selector': '#save', 'is_terminal_write': True},
		],
		'input_schema': {},
	}

	r = await async_client.post(
		'/api/browser/tasks',
		json={
			'compiledImplementationJson': json.dumps(compiled),
			'mode': 'dry_run',
			'inputPayloadJson': json.dumps({'val': 'test', 'key': 'K-DRY'}),
		},
	)
	assert r.status_code == 200
	body = r.json()
	assert body['status'] == 'Succeeded'
	# In dry_run, the terminal save must NOT have fired
	assert not submit_calls, 'dry_run should not POST to the form'
	# Message should mention dry-run
	for ir in body['itemResults']:
		assert 'dry-run' in (ir.get('message') or '')
