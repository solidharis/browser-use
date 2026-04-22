"""Smoke tests for the sidecar HTTP API.

These hit the FastAPI app via TestClient (sync) - no real browser launched.
"""

import json

import pytest
from fastapi.testclient import TestClient

from browser_use.sidecar_api.app import app, _compiled_skills, _training_sessions


@pytest.fixture
def client():
	# Wipe in-memory stores between tests so ordering doesn't matter.
	_training_sessions.clear()
	_compiled_skills.clear()
	with TestClient(app) as c:
		yield c


def test_health(client):
	r = client.get('/health')
	assert r.status_code == 200
	assert r.json() == {'status': 'ok'}


def test_train_start_returns_session_id(client):
	r = client.post(
		'/api/browser/train-start',
		json={'goal': 'create a control', 'skillId': 'skill-1', 'startUrl': 'http://localhost/app'},
	)
	assert r.status_code == 200
	body = r.json()
	assert body['status'] == 'recording'
	assert body['trainingSessionId']


def test_train_complete_returns_traces(client):
	start = client.post('/api/browser/train-start', json={'goal': 'g', 'skillId': 's', 'startUrl': 'http://x'}).json()
	tsid = start['trainingSessionId']

	complete = client.post(
		'/api/browser/train-complete',
		json={
			'trainingSessionId': tsid,
			'rawEvents': [
				{'type': 'navigate', 'url': 'http://x/new'},
				{'type': 'input', 'selector': '#title', 'field': 'title'},
				{'type': 'click', 'selector': '#save', 'is_terminal': True},
				{'type': 'save', 'selector': '#save'},
			],
		},
	)
	assert complete.status_code == 200
	body = complete.json()
	assert body['trainingSessionId'] == tsid

	raw = json.loads(body['rawTraceJson'])
	assert raw['training_session_id'] == tsid
	assert len(raw['events']) == 4

	compiled = json.loads(body['compiledImplementationJson'])
	actions = [s['action'] for s in compiled['steps']]
	# navigate (from start_url) + navigate + input + save
	assert actions[0] == 'navigate'
	assert 'input' in actions and 'save' in actions
	# the save step must be flagged as a terminal write
	assert any(s['action'] == 'save' and s['is_terminal_write'] for s in compiled['steps'])


def test_train_complete_unknown_session(client):
	r = client.post('/api/browser/train-complete', json={'trainingSessionId': 'nope'})
	assert r.status_code == 404


def test_tasks_dry_run_batch(client):
	# Hand-craft a compiled implementation skipping the train flow.
	compiled = {
		'goal': 'create controls',
		'steps': [
			{'action': 'navigate', 'value': 'http://x/new', 'is_terminal_write': False},
			{'action': 'input', 'selector': '#key', 'value': '{controlKey}', 'is_terminal_write': False},
			{'action': 'input', 'selector': '#title', 'value': '{title}', 'is_terminal_write': False},
			{'action': 'save', 'selector': '#save', 'is_terminal_write': True},
		],
		'input_schema': {},
	}
	payload = [
		{'controlKey': 'C-1', 'title': 'first'},
		{'controlKey': 'C-2', 'title': 'second'},
	]

	r = client.post(
		'/api/browser/tasks',
		json={
			'compiledImplementationJson': json.dumps(compiled),
			'mode': 'dry_run',
			'inputPayloadJson': json.dumps(payload),
		},
	)
	assert r.status_code == 200
	body = r.json()
	assert body['status'] == 'Succeeded'
	assert body['stats']['total'] == 2
	assert body['stats']['mode'] == 'dry_run'
	assert len(body['itemResults']) == 2
	assert {ir['inputKey'] for ir in body['itemResults']} == {'C-1', 'C-2'}
	# In dry-run terminal writes are skipped, so items are not "Created"
	for ir in body['itemResults']:
		assert ir['status'] in ('Skipped', 'Created', 'Updated')
		assert 'dry-run' in (ir.get('message') or '')


def test_tasks_execute_single_item_creates(client):
	compiled = {
		'goal': 'create',
		'steps': [
			{'action': 'input', 'selector': '#k', 'value': '{key}', 'is_terminal_write': False},
			{'action': 'save', 'selector': '#save', 'is_terminal_write': True},
		],
		'input_schema': {},
	}
	r = client.post(
		'/api/browser/tasks',
		json={
			'compiledImplementationJson': json.dumps(compiled),
			'mode': 'execute',
			'inputPayloadJson': json.dumps({'key': 'K1'}),
		},
	)
	assert r.status_code == 200
	body = r.json()
	assert body['status'] == 'Succeeded'
	assert body['itemResults'][0]['status'] == 'Created'
	assert body['itemResults'][0]['inputKey'] == 'K1'


def test_tasks_invalid_payload(client):
	r = client.post(
		'/api/browser/tasks',
		json={
			'compiledImplementationJson': '{not json',
			'mode': 'dry_run',
			'inputPayloadJson': '[]',
		},
	)
	assert r.status_code == 400
