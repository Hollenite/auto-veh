"""
environment.py — Main TrafficEnvironment class.

Implements the OpenEnv ``Environment`` interface for the autonomous traffic
signal control problem, wiring together the simulation engine, task configs,
and grading system.

Responsibilities:
    - Wraps ``IntersectionSimulation`` with OpenEnv-compliant
      ``reset()`` / ``step()`` / ``state`` interface.
    - Manages episode lifecycle (episode IDs, step counting, done detection).
    - Converts raw simulation state dicts into typed ``TrafficObservation`` models.
    - Records episode history for post-episode grading.

Classes:
    TrafficEnvironment: OpenEnv ``Environment`` subclass.
"""

from __future__ import annotations

from typing import Any, Optional
from uuid import uuid4

from openenv.core.env_server.interfaces import Environment, EnvironmentMetadata

from models import TrafficAction, TrafficObservation, TrafficState
from server.graders import grade_episode
from server.simulation import IntersectionSimulation
from server.tasks import ALL_TASKS


class TrafficEnvironment(Environment[TrafficAction, TrafficObservation, TrafficState]):
    """Autonomous Traffic Control Environment.

    An AI agent controls a 4-way intersection traffic signal to maximize
    vehicle throughput and handle emergency vehicle prioritization.

    Observation space:
        Queue lengths (4 approaches), signal phase, phase duration,
        emergency vehicle info, cumulative performance metrics.

    Action space:
        ``{KEEP_CURRENT, SWITCH_PHASE, EMERGENCY_OVERRIDE}``

    Reward:
        Per-step signal combining throughput bonus, wait penalty,
        emergency handling reward/penalty, and switch penalty.

    Args:
        task_id: Difficulty level — ``"easy"``, ``"medium"``, or ``"hard"``.

    Raises:
        ValueError: If ``task_id`` is not one of the defined tasks.
    """

    # Each instance has fully isolated state — safe for concurrent sessions.
    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    def __init__(self, task_id: str = "easy", **kwargs: Any) -> None:
        super().__init__(**kwargs)

        if task_id not in ALL_TASKS:
            raise ValueError(
                f"Unknown task_id '{task_id}'. "
                f"Must be one of: {list(ALL_TASKS.keys())}"
            )

        self.task_id: str = task_id
        self.task_config: dict = ALL_TASKS[task_id]
        self.sim: IntersectionSimulation = IntersectionSimulation(self.task_config)
        self._episode_history: list[dict] = []
        self._final_score: float | None = None

        # Initialise internal state tracking
        self._state: TrafficState = TrafficState(
            episode_id=str(uuid4()),
            step_count=0,
            task_id=self.task_id,
            current_phase="NS_GREEN",
            emergency_active=False,
            total_cleared=0,
            total_wait=0,
        )

    # ------------------------------------------------------------------
    # OpenEnv Interface — reset
    # ------------------------------------------------------------------

    def reset(
        self,
        seed: Optional[int] = None,
        episode_id: Optional[str] = None,
        **kwargs: Any,
    ) -> TrafficObservation:
        """Reset the simulation and return the initial observation.

        A new episode is started: the simulation is reset, episode history
        is cleared, and a fresh ``episode_id`` is generated.

        Args:
            seed: Optional RNG seed (reserved for future use).
            episode_id: Optional custom episode ID; auto-generated if omitted.

        Returns:
            Initial ``TrafficObservation`` with zeroed queues and default phase.
        """
        # If a task_id is passed during reset, update environment difficulty
        task_id = kwargs.get("task_id")
        if task_id and task_id in ALL_TASKS and task_id != self.task_id:
            self.task_id = task_id
            self.task_config = ALL_TASKS[task_id]
            self.sim = IntersectionSimulation(self.task_config)

        # Reset the simulation engine
        sim_state = self.sim.reset(seed=seed)

        # Clear episode tracking
        self._episode_history = []
        self._final_score = None

        # Build fresh internal state
        new_episode_id = episode_id or str(uuid4())
        self._state = TrafficState(
            episode_id=new_episode_id,
            step_count=0,
            task_id=self.task_id,
            current_phase=sim_state["current_phase"],
            emergency_active=False,
            total_cleared=0,
            total_wait=0,
        )

        return self._build_observation(sim_state, done=False)

    # ------------------------------------------------------------------
    # OpenEnv Interface — step
    # ------------------------------------------------------------------

    def step(
        self,
        action: TrafficAction,
        timeout_s: Optional[float] = None,
        **kwargs: Any,
    ) -> TrafficObservation:
        """Execute one simulation tick and return the resulting observation.

        The action is forwarded to the simulation engine, step counters are
        updated, and the step's state dict is appended to episode history
        for grading.

        When the episode ends (``step_count >= max_steps``), the final
        grader score is computed and included in the observation message.

        Args:
            action: ``TrafficAction`` specifying the signal control decision.
            timeout_s: Optional timeout (unused — simulation is instant).

        Returns:
            ``TrafficObservation`` with updated queues, metrics, and reward.
        """
        # Advance simulation by one tick
        sim_state = self.sim.step(action.action.value)

        # Record step for grading
        self._episode_history.append(sim_state)

        # Update internal state
        self._state.step_count += 1
        self._state.current_phase = sim_state["current_phase"]
        self._state.emergency_active = sim_state["emergency_present"]
        self._state.total_cleared = sim_state["total_vehicles_cleared"]
        self._state.total_wait = sim_state["total_wait_time"]

        # Check termination
        done = self._state.step_count >= self.task_config["max_steps"]

        # Compute final grade on episode end
        if done:
            self._final_score = grade_episode(self.task_id, self._episode_history)
            sim_state["message"] = (
                f"Episode complete. "
                f"Steps: {self._state.step_count}, "
                f"Cleared: {self._state.total_cleared}, "
                f"Score: {self._final_score:.4f}"
            )

        return self._build_observation(sim_state, done=done)

    # ------------------------------------------------------------------
    # OpenEnv Interface — state property
    # ------------------------------------------------------------------

    @property
    def state(self) -> TrafficState:
        """Return the current internal environment state.

        This is a compact summary used for monitoring and debugging,
        not for agent decision-making (use the observation for that).
        """
        return self._state

    # ------------------------------------------------------------------
    # Metadata
    # ------------------------------------------------------------------

    def get_metadata(self) -> EnvironmentMetadata:
        """Return rich metadata about the environment.

        Provides environment name, description, version, and author
        for the OpenEnv dashboard and documentation.
        """
        return EnvironmentMetadata(
            name="Traffic Control Environment",
            description=(
                "Autonomous traffic signal control for a 4-way intersection. "
                "The agent must maximize vehicle throughput while prioritizing "
                "emergency vehicles. Features 3 difficulty levels: "
                "easy (steady traffic), medium (rush hour + emergencies), "
                "and hard (traffic surge + multiple simultaneous emergencies)."
            ),
            version="0.1.0",
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _build_observation(
        self, sim_state: dict, done: bool
    ) -> TrafficObservation:
        """Convert a raw simulation state dict into a typed observation.

        Args:
            sim_state: Dictionary returned by ``IntersectionSimulation.step()``
                       or ``IntersectionSimulation.reset()``.
            done: Whether the episode has terminated.

        Returns:
            Fully populated ``TrafficObservation``.
        """
        return TrafficObservation(
            queue_north=sim_state["queues"]["NORTH"],
            queue_south=sim_state["queues"]["SOUTH"],
            queue_east=sim_state["queues"]["EAST"],
            queue_west=sim_state["queues"]["WEST"],
            current_phase=sim_state["current_phase"],
            phase_duration=sim_state["phase_duration"],
            emergency_present=sim_state["emergency_present"],
            emergency_direction=sim_state.get("emergency_direction"),
            emergency_urgency=sim_state.get("emergency_urgency"),
            total_vehicles_cleared=sim_state["total_vehicles_cleared"],
            total_wait_time=sim_state["total_wait_time"],
            current_step=self._state.step_count,
            reward=sim_state["reward"],
            done=done,
            success=done,
            message=sim_state.get("message", ""),
        )

    @property
    def final_score(self) -> float | None:
        """The grader score from the last completed episode, or ``None``."""
        return self._final_score

    def __repr__(self) -> str:
        return (
            f"TrafficEnvironment("
            f"task_id='{self.task_id}', "
            f"step={self._state.step_count}, "
            f"phase='{self._state.current_phase}', "
            f"cleared={self._state.total_cleared}"
            f")"
        )
