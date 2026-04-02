import httpx
import json

base = "http://localhost:8000"

httpx.post(f"{base}/reset", json={})

r = httpx.post(f"{base}/step", json={
    "action": {
        "action": "keep_current",
        "emergency_direction": None
    }
})
print("STEP 1 (keep):", json.dumps(r.json(), indent=2))

r = httpx.post(f"{base}/step", json={
    "action": {
        "action": "switch_phase",
        "emergency_direction": None
    }
})
print("STEP 2 (switch):", json.dumps(r.json(), indent=2))

r = httpx.post(f"{base}/step", json={
    "action": {
        "action": "emergency_override",
        "emergency_direction": "NORTH"
    }
})
print("STEP 3 (emergency):", json.dumps(r.json(), indent=2))