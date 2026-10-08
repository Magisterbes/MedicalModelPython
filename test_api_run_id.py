"""Contract tests for the run-id handshake between the UI and the API.

Regression guard for the Sensitivity button. A tab can only tell its own
status/results/downloads apart from another visitor's if the id it stored came
from the response to its own POST. `/api/sensitivity` started a run but replied
with only ``{'status': 'started'}``, so the tab kept the previous run's id,
``isMyRun()`` failed against ``/api/status``, and the UI reported
"Another visitor's run has the instance now -- start your own." on every
sensitivity run, even for the only visitor on the instance. The three
run-starting endpoints must therefore hand back the id that /api/status will
report, and that id must survive the end of the run.
"""
import time

import web_app

# Tiny workload: the assertions are about the id, not the numbers.
SMALL = {'population': 500, 'years': 1, 'factors': [1.0], 'seed': 1}


def _reset():
    web_app._sim_running = False
    web_app._run_id = None
    web_app._run_kind = None
    web_app._sim_result = None
    web_app._sim_progress = 0
    web_app._sensitivity_result = None


def _client():
    web_app.app.config['TESTING'] = True
    return web_app.app.test_client()


def _wait_until_idle(client, timeout=120.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        if not client.get('/api/status').get_json()['running']:
            return True
        time.sleep(0.25)
    return False


def test_sensitivity_response_carries_the_run_id_status_reports():
    _reset()
    client = _client()

    resp = client.post('/api/sensitivity', json=SMALL)
    assert resp.status_code == 200, resp.get_data(as_text=True)
    body = resp.get_json()
    assert body.get('status') == 'started', body
    assert body.get('run_id'), (
        '/api/sensitivity must return the id of the run it just started; '
        "without it the tab cannot claim its own run")

    status = client.get('/api/status').get_json()
    assert status.get('kind') == 'sensitivity', status
    assert status.get('run_id') == body['run_id'], (
        'the tab stored %r from the POST but /api/status reports %r, so '
        'isMyRun() fails and the UI blames another visitor'
        % (body.get('run_id'), status.get('run_id')))

    # The result payload must carry the same id, or the tab refuses to render.
    assert _wait_until_idle(client), 'sensitivity run did not finish'
    result = client.get('/api/sensitivity/result').get_json()
    assert result.get('run_id') == body['run_id'], result.get('run_id')


def test_status_keeps_the_run_id_after_the_run_finishes():
    # Releasing the slot must not clear the id: the tab's poll of /api/status
    # after completion compares against it, and a cleared id looks like a
    # foreign run.
    _reset()
    client = _client()

    started = client.post('/api/sensitivity', json=SMALL).get_json()
    assert _wait_until_idle(client), 'sensitivity run did not finish'

    status = client.get('/api/status').get_json()
    assert status.get('running') is False, status
    assert status.get('run_id') == started['run_id'], status


def test_single_run_response_carries_the_run_id_status_reports():
    # /api/simulate is the sibling endpoint and must keep the same contract.
    _reset()
    client = _client()

    resp = client.post('/api/simulate', json={'population': 500, 'seed': 1})
    assert resp.status_code == 200, resp.get_data(as_text=True)
    body = resp.get_json()
    assert body.get('run_id'), body

    status = client.get('/api/status').get_json()
    assert status.get('run_id') == body['run_id'], (body, status)

    assert _wait_until_idle(client), 'simulation did not finish'


def test_busy_slot_rejects_a_second_start():
    _reset()
    client = _client()
    web_app._sim_running = True          # pretend another run holds the slot
    try:
        resp = client.post('/api/sensitivity', json=SMALL)
        assert resp.status_code == 409, resp.get_data(as_text=True)
        assert 'run_id' in resp.get_json(), (
            'the 409 must let the client name the run it lost to')
    finally:
        web_app._sim_running = False


if __name__ == '__main__':
    test_sensitivity_response_carries_the_run_id_status_reports()
    test_status_keeps_the_run_id_after_the_run_finishes()
    test_single_run_response_carries_the_run_id_status_reports()
    test_busy_slot_rejects_a_second_start()
    print('API run-id tests passed: POST id == /api/status id, '
          'id survives completion, busy slot gives 409')
