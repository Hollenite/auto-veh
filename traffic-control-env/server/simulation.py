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
from models import VehicleRecord, VehicleType, Direction


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
        self.queues: dict[str, list] = {}
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

    def reset(self, seed: int | None = None) -> dict:
        """Reset all simulation state and return the initial state dict.

        Queues are zeroed, the phase is set to NS_GREEN (index 0),
        emergency is cleared, and all counters are reset.

        Returns:
            State dictionary compatible with ``TrafficObservation`` fields.
        """
        self.queues = {d: [] for d in ALL_DIRECTIONS}
        self.current_phase = PHASE_ROTATION[0]  # NS_GREEN
        self.phase_duration = 0
        self.phase_index = 0
        self.emergency = None
        self.total_vehicles_cleared = 0
        self.total_wait_time = 0
        self.step_count = 0

        # Re-seed RNG for reproducible episodes if seed is provided, else random
        if seed is not None:
            self.rng = np.random.default_rng(seed=seed)
        else:
            self.rng = np.random.default_rng()

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

        # Track if emergency gets green before discharge clears it
        emergency_handled = False
        if self.emergency is not None:
            if self.emergency["direction"] in _GREEN_DIRECTIONS.get(self.current_phase, []):
                emergency_handled = True

        # 2. Discharge vehicles on green approaches
        cleared_counts = self._discharge_vehicles()

        # 3. Stochastic vehicle arrivals
        self._arrive_vehicles()
        self._increment_wait_times()

        # 4. Emergency vehicle lifecycle
        self._update_emergency()

        # 5. Reward computation
        reward = self._calculate_reward(cleared_counts, action, emergency_handled)

        # 6. Bookkeeping
        self.phase_duration += 1
        self.total_wait_time += sum(len(q) for q in self.queues.values())

        return self._build_state_dict(reward=reward)

    # ------------------------------------------------------------------
    # Action Processing
    # ------------------------------------------------------------------

    def _process_action(self, action: str) -> None:
        """Update signal phase based on agent action.
        
        Actions:
            set_ns_green:       Force NS_GREEN immediately.
            set_ew_green:       Force EW_GREEN immediately.
            set_all_red:        Force ALL_RED immediately.
            hold_current_phase: No change.
        
        MIN_PHASE_DURATION is enforced for set_ns_green and set_ew_green
        unless an emergency vehicle is present (emergency always overrides).
        """
        if action == "hold_current_phase":
            return

        emergency_active = self.emergency is not None
        
        if action == "set_ns_green":
            if self.current_phase != "NS_GREEN":
                if self.phase_duration >= MIN_PHASE_DURATION or emergency_active:
                    self.current_phase = "NS_GREEN"
                    self.phase_index = PHASE_ROTATION.index("NS_GREEN")
                    self.phase_duration = 0
        
        elif action == "set_ew_green":
            if self.current_phase != "EW_GREEN":
                if self.phase_duration >= MIN_PHASE_DURATION or emergency_active:
                    self.current_phase = "EW_GREEN"
                    self.phase_index = PHASE_ROTATION.index("EW_GREEN")
                    self.phase_duration = 0
        
        elif action == "set_all_red":
            if self.current_phase != "ALL_RED":
                self.current_phase = "ALL_RED"
                self.phase_index = PHASE_ROTATION.index("ALL_RED")
                self.phase_duration = 0

        # Legacy support: old action names map to new ones
        elif action == "keep_current":
            return
        elif action == "switch_phase":
            if self.phase_duration >= MIN_PHASE_DURATION:
                self._advance_phase()
        elif action == "emergency_override":
            if self.emergency is not None:
                target = _PHASE_FOR_DIRECTION[self.emergency["direction"]]
                if self.current_phase != target:
                    self.phase_index = PHASE_ROTATION.index(target)
                    self.current_phase = target
                    self.phase_duration = 0

    def _advance_phase(self) -> None:
        """Move to the next phase in the rotation cycle."""
        self.phase_index = (self.phase_index + 1) % len(PHASE_ROTATION)
        self.current_phase = PHASE_ROTATION[self.phase_index]
        self.phase_duration = 0

    # ------------------------------------------------------------------
    # Vehicle Discharge
    # ------------------------------------------------------------------

    def _discharge_vehicles(self) -> dict:
        """Clear vehicles from approaches with green signal.
        
        Emergency vehicles are prioritized in their queue (popped first).
        Returns dict with keys "normal" and "emergency" counting cleared vehicles.
        """
        green_dirs = _GREEN_DIRECTIONS.get(self.current_phase, [])
        cleared = {"normal": 0, "emergency": 0}

        for direction in green_dirs:
            queue = self.queues[direction]
            for _ in range(DISCHARGE_RATE_PER_DIRECTION):
                if not queue:
                    break
                # Prioritize emergency vehicles
                emerg_idx = next(
                    (i for i, v in enumerate(queue) if v.vehicle_type == VehicleType.EMERGENCY),
                    None
                )
                if emerg_idx is not None:
                    vehicle = queue.pop(emerg_idx)
                else:
                    vehicle = queue.pop(0)
                cleared[vehicle.vehicle_type.value] += 1
                self.total_vehicles_cleared += 1

        # Resolve emergency if its direction got green
        if self.emergency is not None:
            if self.emergency["direction"] in green_dirs:
                self.emergency = None

        return cleared

    # ------------------------------------------------------------------
    # Vehicle Arrivals
    # ------------------------------------------------------------------

    def _arrive_vehicles(self) -> None:
        """Add new VehicleRecord objects to each approach queue.
        
        Arrival count = max(0, round(rate + N(0, noise_std))).
        Queues are capped at MAX_QUEUE_SIZE.
        """
        arrival_rates = self.task_config["arrival_rates"]
        noise_std = self.task_config["arrival_noise_std"]

        for direction in ALL_DIRECTIONS:
            rate = arrival_rates[direction]
            noise = float(self.rng.normal(0, noise_std))
            arrivals = max(0, round(rate + noise))
            
            for i in range(arrivals):
                if len(self.queues[direction]) >= MAX_QUEUE_SIZE:
                    break
                self.queues[direction].append(VehicleRecord(
                    vehicle_id=f"{direction}-{self.step_count}-{i}",
                    direction=Direction(direction),
                    vehicle_type=VehicleType.NORMAL,
                    wait_time=0,
                    arrival_step=self.step_count,
                ))

    def _increment_wait_times(self) -> None:
        """Increment wait_time for every vehicle currently in a queue."""
        for queue in self.queues.values():
            for vehicle in queue:
                vehicle.wait_time += 1

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

    def _calculate_reward(
        self,
        cleared_counts: dict,
        action: str,
        emergency_handled: bool
    ) -> float:
        """Compute per-step reward.
        
        Components:
            throughput:        +1.0 per normal vehicle, +3.0 per emergency vehicle cleared.
            queue_penalty:     -0.15 per vehicle currently waiting.
            wait_penalty:      -0.10 * average wait time across all queues.
            emergency_penalty: Escalating penalty when emergency waits; severe past timeout.
            switch_penalty:    -0.25 for switching before MIN_PHASE_DURATION.
            invalid_penalty:   -2.0 for unrecognized action strings.
        """
        # Throughput — emergency vehicles worth 3x
        throughput = (
            cleared_counts.get("normal", 0) * 1.0
            + cleared_counts.get("emergency", 0) * 3.0
        )

        # Queue metrics from VehicleRecord lists
        all_vehicles = [v for q in self.queues.values() for v in q]
        total_waiting = len(all_vehicles)
        avg_wait = (
            sum(v.wait_time for v in all_vehicles) / total_waiting
            if total_waiting > 0 else 0.0
        )

        queue_penalty = total_waiting * 0.15
        wait_penalty = avg_wait * 0.10

        # Emergency handling
        emergency_reward = 0.0
        if emergency_handled:
            emergency_reward = 0.0  # Vehicle already cleared — captured in throughput
        elif self.emergency is not None:
            steps_waited = self.emergency["steps_waiting"]
            urgency_multiplier = {"LOW": 1.0, "HIGH": 1.5, "CRITICAL": 2.0}.get(
                self.emergency.get("urgency", "LOW"), 1.0
            )
            emergency_reward = -0.2 * steps_waited * urgency_multiplier
            if steps_waited > EMERGENCY_TIMEOUT:
                emergency_reward -= 5.0

        # Switch penalty — only when switching before min duration
        switch_penalty = 0.0
        premature_switch = (
            action in ("set_ns_green", "set_ew_green", "set_all_red", "switch_phase")
            and self.phase_duration < MIN_PHASE_DURATION
            and self.emergency is None  # no penalty when emergency forces switch
        )
        if premature_switch:
            switch_penalty = -0.25

        # Invalid action penalty
        valid_actions = {
            "set_ns_green", "set_ew_green", "hold_current_phase", "set_all_red",
            "keep_current", "switch_phase", "emergency_override"
        }
        invalid_penalty = -2.0 if action not in valid_actions else 0.0

        return throughput - queue_penalty - wait_penalty + emergency_reward - switch_penalty + invalid_penalty

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

        queue_counts = {d: len(q) for d, q in self.queues.items()}
        avg_waits = {}
        for d, q in self.queues.items():
            avg_waits[d] = (sum(v.wait_time for v in q) / len(q)) if q else 0.0

        return {
            "queues": queue_counts,
            "avg_wait": avg_waits,
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
