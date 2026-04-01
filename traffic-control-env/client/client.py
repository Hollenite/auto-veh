"""
client.py — TrafficEnv client for the Traffic Control Environment.

Provides an EnvClient subclass that communicates with the FastAPI server
over HTTP, handling serialization/deserialization of TrafficAction and
TrafficObservation models.

Classes:
    TrafficEnv: EnvClient subclass with reset(), step(), and state() methods.
                Connects to the environment server at a configurable base_url.

Usage:
    from client.client import TrafficEnv

    env = TrafficEnv(base_url="http://localhost:8000", task_id="easy")
    obs = env.reset()
    obs = env.step({"action": "SWITCH_PHASE"})
"""
