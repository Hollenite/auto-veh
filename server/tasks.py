"""
tasks.py — Task definitions for the Traffic Control Environment.

Defines three tasks of increasing difficulty, each as a configuration
dictionary consumed by ``IntersectionSimulation``.

Task Difficulty Progression:
    EASY   → Balanced traffic, no emergencies, predictable arrivals.
    MEDIUM → Asymmetric rush-hour load, occasional emergency vehicles.
    HARD   → High-volume surges on all approaches, frequent simultaneous emergencies.

Usage:
    from server.tasks import ALL_TASKS
    config = ALL_TASKS["medium"]
    sim = IntersectionSimulation(config)
"""

from __future__ import annotations


# ---------------------------------------------------------------------------
# Task 1: EASY — Steady State Control
# ---------------------------------------------------------------------------

EASY_TASK: dict = {
    "task_id": "easy",
    "description": (
        "Balanced 4-way intersection with steady predictable traffic. "
        "No emergencies."
    ),
    "max_steps": 50,
    "arrival_rates": {
        "NORTH": 1.5,
        "SOUTH": 1.5,
        "EAST": 1.5,
        "WEST": 1.5,
    },
    "arrival_noise_std": 0.3,       # Low noise — predictable flow
    "emergency_probability": 0.0,   # No emergency vehicles
    "max_simultaneous_emergencies": 0,
    "success_threshold": 0.6,       # Normalized score to "pass"
}


# ---------------------------------------------------------------------------
# Task 2: MEDIUM — Rush Hour + Emergencies
# ---------------------------------------------------------------------------

MEDIUM_TASK: dict = {
    "task_id": "medium",
    "description": (
        "Rush hour with heavy north-south load and occasional "
        "emergency vehicles."
    ),
    "max_steps": 50,
    "arrival_rates": {
        "NORTH": 3.0,   # Heavy north-south corridor
        "SOUTH": 2.8,
        "EAST": 0.8,    # Light east-west cross traffic
        "WEST": 0.9,
    },
    "arrival_noise_std": 0.8,       # Moderate noise — some unpredictability
    "emergency_probability": 0.08,  # ~4 emergencies per 50-step episode
    "max_simultaneous_emergencies": 1,
    "success_threshold": 0.55,
}


# ---------------------------------------------------------------------------
# Task 3: HARD — Crisis Management
# ---------------------------------------------------------------------------

HARD_TASK: dict = {
    "task_id": "hard",
    "description": (
        "Traffic surge with high volume and multiple simultaneous "
        "emergencies. Triage required."
    ),
    "max_steps": 60,
    "arrival_rates": {
        "NORTH": 4.0,   # Heavy on all approaches
        "SOUTH": 4.0,
        "EAST": 3.5,
        "WEST": 3.5,
    },
    "arrival_noise_std": 1.5,       # High noise — traffic surges
    "emergency_probability": 0.15,  # Frequent emergencies
    "max_simultaneous_emergencies": 2,
    "success_threshold": 0.45,
}


# ---------------------------------------------------------------------------
# Unified lookup
# ---------------------------------------------------------------------------

ALL_TASKS: dict[str, dict] = {
    "easy": EASY_TASK,
    "medium": MEDIUM_TASK,
    "hard": HARD_TASK,
}
"""Map of ``task_id`` → task configuration dictionary."""
