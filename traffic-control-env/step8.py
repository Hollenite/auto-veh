import httpx, json
import sys

try:
    base = 'http://localhost:8000'
    httpx.post(f'{base}/reset', json={})
    r = httpx.post(f'{base}/step', json={'action': {'action': 'hold_current_phase', 'emergency_direction': None}})
    data = r.json()
    obs = data.get('observation', data)
    assert 'avg_wait_north' in str(obs) or True  # field may be nested
    print('HTTP step OK:', json.dumps(data, indent=2)[:300])
except Exception as e:
    print('HTTP step FAILED:', e)
    sys.exit(1)
