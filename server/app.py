"""
app.py — FastAPI application entry point.

Creates the OpenEnv-compliant FastAPI app using the ``create_app()``
factory, wiring together the TrafficEnvironment, TrafficAction, and
TrafficObservation types.

The generated app exposes:
    - WebSocket ``/ws``   — Persistent session for reset/step/state
    - GET ``/health``     — Health check endpoint

Usage:
    uvicorn server.app:app --host 0.0.0.0 --port 7860
    python -m server.app
"""

from __future__ import annotations

import logging
import sys

import uvicorn
from fastapi import FastAPI
from fastapi.responses import JSONResponse

from openenv.core.env_server import create_app

from models import TrafficAction, TrafficObservation
from server.environment import TrafficEnvironment

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logger = logging.getLogger("traffic-control-env")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
)

# ---------------------------------------------------------------------------
# Startup Logic
# ---------------------------------------------------------------------------


def _create_environment() -> TrafficEnvironment:
    """Factory function for creating TrafficEnvironment instances.

    Called by the OpenEnv framework for each new session.
    Defaults to the 'easy' task; task selection is handled
    at the session/client level.
    """
    return TrafficEnvironment(task_id="easy")


# Create the OpenEnv FastAPI application
app: FastAPI = create_app(
    env=_create_environment,
    action_cls=TrafficAction,
    observation_cls=TrafficObservation,
    env_name="traffic-control-env",
)


# ---------------------------------------------------------------------------
# Health Check
# ---------------------------------------------------------------------------

@app.get("/health", tags=["system"])
async def health_check() -> JSONResponse:
    """Health check endpoint for monitoring and container orchestration.

    Returns:
        JSON with status and environment name.
    """
    return JSONResponse(
        content={
            "status": "ok",
            "environment": "traffic-control-env",
        }
    )


# ---------------------------------------------------------------------------
# Startup Event
# ---------------------------------------------------------------------------

@app.on_event("startup")
async def on_startup() -> None:
    """Log environment info when the server starts."""
    env = TrafficEnvironment(task_id="easy")
    meta = env.get_metadata()
    logger.info("=" * 60)
    logger.info("  %s v%s", meta.name, meta.version)
    logger.info("  %s", meta.description)
    logger.info("  Tasks: easy, medium, hard")
    logger.info("  Endpoints: /ws (WebSocket), /health (GET)")
    logger.info("=" * 60)


# ---------------------------------------------------------------------------
# Main Entry Point
# ---------------------------------------------------------------------------


def main() -> None:
    """Main entry point for running the server."""
    uvicorn.run(
        "server.app:app",
        host="0.0.0.0",
        port=7860,
        log_level="info",
    )


if __name__ == "__main__":
    main()
