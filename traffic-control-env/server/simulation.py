"""
simulation.py — IntersectionSimulation logic engine.

This module contains the core simulation for a 4-way intersection,
handling vehicle queues, signal phases, emergency vehicles, and
reward computation.

The simulation models a real-world traffic controller's decision loop:
each ``step()`` call represents ~10 real-world seconds. Vehicles arrive
stochastically, queues build up, and the active signal phase determines
which approaches can discharge vehicles.

Classes:
    IntersectionSimulation: Core physics/logic engine for the intersection.
"""

from __future__ import annotations

import numpy as np


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DISCHARGE_RATE_PER_DIRECTION: int = 2
"""Vehicles that can clear per green direction per simulation step."""

MAX_QUEUE_SIZE: int = 20
"""Maximum vehicles allowed in any single approach queue."""

MIN_PHASE_DURATION: int = 2
"""Minimum steps a phase must remain active before switching is allowed."""

EMERGENCY_TIMEOUT: int = 8
"""Steps before emergency vehicle penalty escalates severely."""

PHASE_ROTATION: list[str] = ["NS_GREEN", "ALL_RED", "EW_GREEN", "ALL_RED"]
"""Cyclic signal phase rotation order."""

# Mapping: which directions get green in each phase.
_GREEN_DIRECTIONS: dict[str, list[str]] = {
    "NS_GREEN": ["NORTH", "SOUTH"],
    "EW_GREEN": ["EAST", "WEST"],
    "ALL_RED": [],
}

# Reverse lookup: for a given direction, which phase grants green.
_PHASE_FOR_DIRECTION: dict[str, str] = {
    "NORTH": "NS_GREEN",
    "SOUTH": "NS_GREEN",
    "EAST": "EW_GREEN",
    "WEST": "EW_GREEN",
}

ALL_DIRECTIONS: list[str] = ["NORTH", "SOUTH", "EAST", "WEST"]


# ---------------------------------------------------------------------------
# Simulation Engine
# ---------------------------------------------------------------------------

class IntersectionSimulation:
    """Core physics/logic engine for a 4-way intersection.

    Manages vehicle queues, signal phase transitions, emergency vehicle
    spawning/tracking, and per-step reward calculation.

    Args:
        task_config: Dictionary containing task-specific parameters:
            - ``arrival_rates``           dict[str, float] — vehicles/step per direction
            - ``arrival_noise_std``       float — Gaussian noise std on arrivals
            - ``max_steps``               int — episode length
            - ``emergency_probability``   float — per-step probability of spawning
            - ``max_simultaneous_emergencies`` int (optional, default 1)
    """

    def __init__(self, task_config: dict) -> None:
        self.task_config = task_config

        # --- Mutable state (all reset in reset()) ---
        self.queues: dict[str, int] = {}
        self.current_phase: str = ""
        self.phase_duration: int = 0
        self.phase_index: int = 0
        self.emergency: dict | None = None
        self.total_vehicles_cleared: int = 0
        self.total_wait_time: int = 0
        self.step_count: int = 0

        # Reproducible RNG
        self.rng: np.random.Generator = np.random.default_rng(seed=42)

        # Initialise to a clean starting state
        self.reset()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def reset(self) -> dict:
        """Reset all simulation state and return the initial state dict.

        Queues are zeroed, the phase is set to NS_GREEN (index 0),
        emergency is cleared, and all counters are reset.

        Returns:
            State dictionary compatible with ``TrafficObservation`` fields.
        """
        self.queues = {d: 0 for d in ALL_DIRECTIONS}
        self.current_phase = PHASE_ROTATION[0]  # NS_GREEN
        self.phase_duration = 0
        self.phase_index = 0
        self.emergency = None
        self.total_vehicles_cleared = 0
        self.total_wait_time = 0
        self.step_count = 0

        # Re-seed RNG for reproducible episodes
        self.rng = np.random.default_rng(seed=42)

        return self._build_state_dict(reward=0.0)

    def step(self, action: str) -> dict:
        """Advance the simulation by one tick.

        Execution order:
            1. Process the agent's action (phase transition logic).
            2. Discharge vehicles from queues that have a green signal.
            3. Arrive new vehicles based on configured arrival rates.
            4. Update / spawn emergency vehicles.
            5. Calculate per-step reward.
            6. Increment phase duration and accumulate wait time.

        Args:
            action: One of ``"keep_current"``, ``"switch_phase"``,
                    or ``"emergency_override"``.

        Returns:
            State dictionary with all fields needed by ``TrafficObservation``.
        """
        self.step_count += 1

        # 1. Phase transition
        self._process_action(action)

        # 2. Discharge vehicles on green approaches
        vehicles_cleared = self._discharge_vehicles()

        # 3. Stochastic vehicle arrivals
        self._arrive_vehicles()

        # 4. Emergency vehicle lifecycle
        self._update_emergency()

        # 5. Reward computation
        reward = self._calculate_reward(vehicles_cleared, action)

        # 6. Bookkeeping
        self.phase_duration += 1
        self.total_wait_time += sum(self.queues.values())

        return self._build_state_dict(reward=reward)

    # ------------------------------------------------------------------
    # Action Processing
    # ------------------------------------------------------------------

    def _process_action(self, action: str) -> None:
        """Update the signal phase based on the agent's action.

        Rules:
            - ``KEEP_CURRENT``: No phase change.
            - ``SWITCH_PHASE``: Advance to the next phase in rotation,
              **unless** the current phase has been active for fewer than
              ``MIN_PHASE_DURATION`` steps (action is silently ignored).
            - ``EMERGENCY_OVERRIDE``: Force the phase that gives green to
              the emergency vehicle's direction. This is always allowed
              regardless of ``MIN_PHASE_DURATION`` to model the real-world
              priority of emergency response.

        Args:
            action: The agent's chosen action string.
        """
        if action == "keep_current":
            return

        if action == "switch_phase":
            # Enforce minimum phase duration — prevent signal thrashing
            if self.phase_duration < MIN_PHASE_DURATION:
                return
            self._advance_phase()

        elif action == "emergency_override":
            if self.emergency is not None:
                target_phase = _PHASE_FOR_DIRECTION[self.emergency["direction"]]
                if self.current_phase != target_phase:
                    # Jump directly to the target phase
                    self.phase_index = PHASE_ROTATION.index(target_phase)
                    self.current_phase = target_phase
                    self.phase_duration = 0

    def _advance_phase(self) -> None:
        """Move to the next phase in the rotation cycle."""
        self.phase_index = (self.phase_index + 1) % len(PHASE_ROTATION)
        self.current_phase = PHASE_ROTATION[self.phase_index]
        self.phase_duration = 0

    # ------------------------------------------------------------------
    # Vehicle Discharge
    # ------------------------------------------------------------------

    def _discharge_vehicles(self) -> int:
        """Clear vehicles from approaches that currently have a green signal.

        Each green direction can discharge up to ``DISCHARGE_RATE_PER_DIRECTION``
        vehicles per step. During ``ALL_RED``, no vehicles are discharged.

        Returns:
            Total number of vehicles cleared this step.
        """
        green_dirs = _GREEN_DIRECTIONS.get(self.current_phase, [])
        total_cleared = 0

        for direction in green_dirs:
            can_clear = min(self.queues[direction], DISCHARGE_RATE_PER_DIRECTION)
            self.queues[direction] -= can_clear
            total_cleared += can_clear

        self.total_vehicles_cleared += total_cleared

        # If emergency vehicle's direction got green, resolve the emergency
        if self.emergency is not None:
            if self.emergency["direction"] in green_dirs:
                self.emergency = None  # Emergency vehicle has passed through

        return total_cleared

    # ------------------------------------------------------------------
    # Vehicle Arrivals
    # ------------------------------------------------------------------

    def _arrive_vehicles(self) -> None:
        """Add new vehicles to each approach queue.

        Arrival count per direction =
            ``max(0, round(arrival_rate + N(0, noise_std)))``

        Queues are capped at ``MAX_QUEUE_SIZE``.
        """
        arrival_rates: dict[str, float] = self.task_config["arrival_rates"]
        noise_std: float = self.task_config["arrival_noise_std"]

        for direction in ALL_DIRECTIONS:
            rate = arrival_rates[direction]
            noise = float(self.rng.normal(0, noise_std))
            arrivals = max(0, round(rate + noise))
            self.queues[direction] = min(
                self.queues[direction] + arrivals,
                MAX_QUEUE_SIZE,
            )

    # ------------------------------------------------------------------
    # Emergency Vehicle Lifecycle
    # ------------------------------------------------------------------

    def _update_emergency(self) -> None:
        """Spawn, age, or expire emergency vehicles.

        Spawning:
            If no emergency is active, one may spawn with probability
            ``task_config["emergency_probability"]`` at a random direction.

        Aging:
            Each step the emergency exists, ``steps_waiting`` is incremented
            and urgency is escalated:
                - steps 0–2  → LOW
                - steps 3–5  → HIGH
                - steps 6+   → CRITICAL

        Expiry:
            If the emergency has waited longer than ``EMERGENCY_TIMEOUT * 2``
            steps, it is cleared (the vehicle gave up / rerouted).
        """
        if self.emergency is None:
            # Possibly spawn a new emergency
            prob = self.task_config.get("emergency_probability", 0.0)
            if prob > 0.0 and float(self.rng.random()) < prob:
                direction = str(self.rng.choice(ALL_DIRECTIONS))
                self.emergency = {
                    "direction": direction,
                    "urgency": "LOW",
                    "steps_waiting": 0,
                }
        else:
            # Age the existing emergency
            self.emergency["steps_waiting"] += 1
            self.emergency["urgency"] = self._urgency_for_steps(
                self.emergency["steps_waiting"]
            )

            # Expiry: vehicle gives up after extended wait
            if self.emergency["steps_waiting"] > EMERGENCY_TIMEOUT * 2:
                self.emergency = None

    @staticmethod
    def _urgency_for_steps(steps_waiting: int) -> str:
        """Map wait duration to urgency level.

        Args:
            steps_waiting: How many steps the emergency vehicle has waited.

        Returns:
            ``"LOW"`` (0–2), ``"HIGH"`` (3–5), or ``"CRITICAL"`` (6+).
        """
        if steps_waiting <= 2:
            return "LOW"
        elif steps_waiting <= 5:
            return "HIGH"
        else:
            return "CRITICAL"

    # ------------------------------------------------------------------
    # Reward Calculation
    # ------------------------------------------------------------------

    def _calculate_reward(self, vehicles_cleared: int, action: str) -> float:
        """Compute the per-step reward signal.

        Components:
            **throughput_reward**: ``+0.1`` per vehicle cleared this step.
            **wait_penalty**: ``-0.01`` per vehicle currently waiting.
            **emergency_handling**: Large bonus for correct prioritisation,
                escalating penalty for delays, severe penalty beyond timeout.
            **switch_penalty**: ``-0.3`` if the agent attempted to switch
                phases before ``MIN_PHASE_DURATION`` elapsed.

        Args:
            vehicles_cleared: Vehicles discharged this step.
            action: The action string the agent submitted.

        Returns:
            Combined reward for this simulation step.
        """
        # --- Throughput ---
        throughput_reward: float = vehicles_cleared * 0.1

        # --- Wait penalty ---
        wait_penalty: float = -0.01 * sum(self.queues.values())

        # --- Emergency handling ---
        emergency_reward: float = 0.0
        if self.emergency is not None:
            direction = self.emergency["direction"]
            green_dirs = _GREEN_DIRECTIONS.get(self.current_phase, [])
            direction_gets_green = direction in green_dirs

            if direction_gets_green:
                emergency_reward = 1.0
            else:
                emergency_reward = -0.2 * self.emergency["steps_waiting"]

            if self.emergency["steps_waiting"] > EMERGENCY_TIMEOUT:
                emergency_reward -= 5.0

        # --- Switch penalty (thrashing deterrent) ---
        switch_penalty: float = 0.0
        if action == "switch_phase" and self.phase_duration < MIN_PHASE_DURATION:
            switch_penalty = -0.3

        return throughput_reward + wait_penalty + emergency_reward + switch_penalty

    # ------------------------------------------------------------------
    # State Serialisation
    # ------------------------------------------------------------------

    def _build_state_dict(self, reward: float = 0.0) -> dict:
        """Build the state dictionary consumed by ``TrafficObservation``.

        Returns:
            dict with keys: ``queues``, ``current_phase``, ``phase_duration``,
            ``emergency_present``, ``emergency_direction``, ``emergency_urgency``,
            ``total_vehicles_cleared``, ``total_wait_time``, ``reward``, ``message``.
        """
        emergency_present = self.emergency is not None
        emergency_direction = self.emergency["direction"] if emergency_present else None
        emergency_urgency = self.emergency["urgency"] if emergency_present else None

        # Contextual message for the agent
        if emergency_present:
            ew = self.emergency["steps_waiting"]  # type: ignore[index]
            message = (
                f"Emergency vehicle at {emergency_direction} "
                f"(urgency={emergency_urgency}, waiting={ew} steps)"
            )
        else:
            message = f"Phase {self.current_phase} active for {self.phase_duration} steps"

        return {
            "queues": dict(self.queues),
            "current_phase": self.current_phase,
            "phase_duration": self.phase_duration,
            "emergency_present": emergency_present,
            "emergency_direction": emergency_direction,
            "emergency_urgency": emergency_urgency,
            "total_vehicles_cleared": self.total_vehicles_cleared,
            "total_wait_time": self.total_wait_time,
            "reward": reward,
            "message": message,
        }
