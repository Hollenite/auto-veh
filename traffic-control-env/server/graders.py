"""
graders.py — Episode grading functions for the Traffic Control Environment.

Each grader receives the full ``episode_history`` — a list of state dicts
(one per step, as returned by ``IntersectionSimulation._build_state_dict``)
— and returns a normalized score in ``[0.0, 1.0]``.

Grading Components:
    **throughput_score**:    Fraction of maximum possible vehicle clearance.
    **emergency_score**:     How quickly emergency vehicles were handled.
    **queue_balance_score**: Evenness of final queue lengths (hard only).

Functions:
    grade_easy    — Throughput only.
    grade_medium  — 60% throughput + 40% emergency response.
    grade_hard    — 40% throughput + 40% emergency + 20% queue balance.
    grade_episode — Router that dispatches to the correct grader.
"""

from __future__ import annotations

import numpy as np

from server.simulation import EMERGENCY_TIMEOUT, MAX_QUEUE_SIZE


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _clamp(value: float) -> float:
    """Clamp a value to the [0.0, 1.0] range."""
    return max(0.0, min(1.0, value))


def _compute_throughput_score(episode_history: list[dict], max_possible: int) -> float:
    """Compute throughput as a fraction of the theoretical maximum clearance.

    Args:
        episode_history: List of per-step state dicts.
        max_possible: Theoretical maximum vehicles that could be cleared
                      (``max_steps * 4`` — 2 directions × 2 vehicles each).

    Returns:
        Throughput score in [0.0, 1.0].
    """
    if not episode_history:
        return 0.0
    total_cleared = episode_history[-1]["total_vehicles_cleared"]
    return _clamp(total_cleared / max_possible)


def _compute_emergency_score(episode_history: list[dict]) -> float:
    """Score how quickly the agent handled emergency vehicles.

    For each contiguous stretch of steps where an emergency was present,
    count how many steps it waited (``steps_waiting`` at resolution or
    expiry). The score is the mean of
    ``1.0 - (steps_waited / EMERGENCY_TIMEOUT)`` across all emergencies.

    If no emergencies occurred during the episode, returns 1.0 (perfect).

    Args:
        episode_history: List of per-step state dicts.

    Returns:
        Emergency response score in [0.0, 1.0].
    """
    if not episode_history:
        return 1.0

    # Track emergency waiting durations by detecting transitions.
    # An emergency "event" starts when emergency_present goes from
    # False → True and ends when it goes from True → False.
    emergency_wait_durations: list[int] = []
    current_wait: int = 0
    was_present: bool = False

    for step_state in episode_history:
        is_present = step_state["emergency_present"]

        if is_present:
            current_wait += 1
            was_present = True
        elif was_present and not is_present:
            # Emergency just resolved (or expired)
            emergency_wait_durations.append(current_wait)
            current_wait = 0
            was_present = False

    # Handle emergency still active at episode end
    if was_present and current_wait > 0:
        emergency_wait_durations.append(current_wait)

    if not emergency_wait_durations:
        return 1.0

    # Score each emergency event
    scores = [
        _clamp(1.0 - (wait / EMERGENCY_TIMEOUT))
        for wait in emergency_wait_durations
    ]
    return float(np.mean(scores))


def _compute_queue_balance_score(episode_history: list[dict]) -> float:
    """Score the evenness of queue lengths at the end of the episode.

    A perfectly balanced intersection has equal queue lengths across all
    four approaches. The score penalises high standard deviation.

    ``score = 1.0 - (std_dev_of_final_queues / MAX_QUEUE_SIZE)``

    Args:
        episode_history: List of per-step state dicts.

    Returns:
        Queue balance score in [0.0, 1.0].
    """
    if not episode_history:
        return 1.0

    final_queues = episode_history[-1]["queues"]
    queue_values = list(final_queues.values())
    std_dev = float(np.std(queue_values))
    return _clamp(1.0 - (std_dev / MAX_QUEUE_SIZE))


def _compute_wait_score(episode_history: list[dict]) -> float:
    """Score based on average wait time across the episode.
    
    Lower average wait = higher score.
    Acceptable threshold = half the episode length.
    
    Returns:
        Wait quality score in [0.0, 1.0].
    """
    if not episode_history:
        return 1.0
    # Use total_wait_time from last step divided by steps to get avg
    last = episode_history[-1]
    steps = len(episode_history)
    avg_wait = last.get("total_wait_time", 0) / max(steps, 1)
    acceptable = steps / 2.0
    return _clamp(1.0 - (avg_wait / max(acceptable, 1.0)))


def _compute_stability_score(episode_history: list[dict]) -> float:
    """Score based on how often the agent switched signal phases.
    
    Fewer switches relative to episode length = higher score.
    
    Returns:
        Stability score in [0.0, 1.0].
    """
    if not episode_history:
        return 1.0
    # Count phase changes by detecting when current_phase changes between steps
    switches = 0
    for i in range(1, len(episode_history)):
        if episode_history[i]["current_phase"] != episode_history[i - 1]["current_phase"]:
            switches += 1
    max_switches = len(episode_history) - 1
    return _clamp(1.0 - (switches / max(max_switches, 1)))


def _compute_fairness_score(episode_history: list[dict]) -> float:
    """Score how evenly the agent served all four directions.
    
    Uses avg_wait per direction from the last step to compute fairness.
    A perfectly fair controller has equal avg_wait across all directions.
    
    Returns:
        Fairness score in [0.0, 1.0]. 1.0 = perfectly fair.
    """
    if not episode_history:
        return 1.0
    
    last = episode_history[-1]
    avg_wait = last.get("avg_wait", {})
    
    if not avg_wait:
        return 1.0
    
    values = list(avg_wait.values())
    if not values or max(values) == 0:
        return 1.0
    
    # Fairness = 1 - (range / max) — penalizes starving one direction
    return _clamp(1.0 - (max(values) - min(values)) / max(max(values), 1.0))


# ---------------------------------------------------------------------------
# Per-Task Graders
# ---------------------------------------------------------------------------

def grade_easy(episode_history: list[dict]) -> float:
    """Grade an easy-task episode on throughput only.

    ``score = total_vehicles_cleared / 200``

    Max possible clearance = 50 steps × 4 vehicles/step = 200.

    Args:
        episode_history: List of per-step state dicts from the episode.

    Returns:
        Score in [0.0, 1.0].
    """
    max_possible = 50 * 4  # 200
    return _compute_throughput_score(episode_history, max_possible)


def grade_medium(episode_history: list[dict]) -> float:
    """Grade medium task: throughput + emergency + wait quality + stability.
    
    Weights:
        throughput    35%
        emergency     30%
        wait quality  25%
        stability     10%
    
    Max possible clearance = 50 steps × 4 vehicles = 200.
    """
    max_possible = 50 * 4
    throughput = _compute_throughput_score(episode_history, max_possible)
    emergency  = _compute_emergency_score(episode_history)
    wait       = _compute_wait_score(episode_history)
    stability  = _compute_stability_score(episode_history)
    return _clamp(
        0.35 * throughput
        + 0.30 * emergency
        + 0.25 * wait
        + 0.10 * stability
    )


def grade_hard(episode_history: list[dict]) -> float:
    """Grade hard task: throughput + emergency + wait + fairness + stability.
    
    Weights:
        emergency     35%
        throughput    25%
        wait quality  20%
        fairness      15%
        stability      5%
    
    Max possible clearance = 60 steps × 4 vehicles = 240.
    """
    max_possible = 60 * 4
    throughput = _compute_throughput_score(episode_history, max_possible)
    emergency  = _compute_emergency_score(episode_history)
    wait       = _compute_wait_score(episode_history)
    fairness   = _compute_fairness_score(episode_history)
    stability  = _compute_stability_score(episode_history)
    return _clamp(
        0.35 * emergency
        + 0.25 * throughput
        + 0.20 * wait
        + 0.15 * fairness
        + 0.05 * stability
    )


# ---------------------------------------------------------------------------
# Router
# ---------------------------------------------------------------------------

_GRADERS: dict[str, callable] = {
    "easy": grade_easy,
    "medium": grade_medium,
    "hard": grade_hard,
}


def grade_episode(task_id: str, episode_history: list[dict]) -> float:
    """Route to the correct grader and return the episode score.

    Args:
        task_id: One of ``"easy"``, ``"medium"``, ``"hard"``.
        episode_history: List of per-step state dicts from the episode.

    Returns:
        Clamped float score in [0.0, 1.0].

    Raises:
        ValueError: If ``task_id`` is not recognised.
    """
    if task_id not in _GRADERS:
        raise ValueError(
            f"Unknown task_id '{task_id}'. Must be one of: {list(_GRADERS.keys())}"
        )
    score = _GRADERS[task_id](episode_history)
    return _clamp(score)
