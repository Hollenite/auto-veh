"""
app.py — FastAPI application entry point.

Creates the OpenEnv-compliant FastAPI app using the openenv-core
create_app() factory, wiring together:
- TrafficEnvironment (environment class)
- TrafficAction (action model)
- TrafficObservation (observation model)

The generated app exposes:
- POST /reset   — Reset the environment and return initial observation
- POST /step    — Submit an action and receive the next observation
- GET  /state   — Retrieve the current environment state

Usage:
    uvicorn server.app:app --host 0.0.0.0 --port 8000
"""
