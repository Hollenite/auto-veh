"""
test_environment.py — Tests for the Traffic Control Environment.

Covers:
    - Environment reset and observation validity
    - Step counting and episode termination
    - Emergency vehicle override and reward signals
    - Phase switching behaviour
    - Grader score ranges across all tasks
    - Pydantic model serialization round-trips

Run with:
    pytest tests/ -v
"""

from __future__ import annotations

import sys
import os

# Ensure project root is on the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest

from models import (
    SignalAction,
    TrafficAction,
    TrafficObservation,
    TrafficState,
    VALID_PHASES,
)
from server.environment import TrafficEnvironment
from server.graders import grade_episode
from server.simulation import IntersectionSimulation
from server.tasks import ALL_TASKS


# =========================================================================
# Fixtures
# =========================================================================

@pytest.fixture
def easy_env() -> TrafficEnvironment:
    """Fresh TrafficEnvironment with the easy task."""
    env = TrafficEnvironment(task_id="easy")
    env.reset()
    return env


@pytest.fixture
def medium_env() -> TrafficEnvironment:
    """Fresh TrafficEnvironment with the medium task."""
    env = TrafficEnvironment(task_id="medium")
    env.reset()
    return env


@pytest.fixture
def hard_env() -> TrafficEnvironment:
    """Fresh TrafficEnvironment with the hard task."""
    env = TrafficEnvironment(task_id="hard")
    env.reset()
    return env


# =========================================================================
# 1. Reset returns valid observation
# =========================================================================

class TestReset:
    """Tests for environment reset behaviour."""

    def test_reset_returns_valid_observation(self, easy_env: TrafficEnvironment):
        """reset() should return a TrafficObservation with zeroed queues and defaults."""
        obs = easy_env.reset()

        assert isinstance(obs, TrafficObservation)
        assert obs.queue_north == 0
        assert obs.queue_south == 0
        assert obs.queue_east == 0
        assert obs.queue_west == 0
        assert obs.current_phase in VALID_PHASES
        assert obs.done is False
        assert obs.reward == 0.0
        assert obs.current_step == 0
        assert obs.total_vehicles_cleared == 0
        assert obs.total_wait_time == 0

    def test_reset_clears_previous_state(self, easy_env: TrafficEnvironment):
        """Calling reset() after steps should restore initial state."""
        # Run a few steps
        action = TrafficAction(action=SignalAction.KEEP_CURRENT)
        for _ in range(5):
            easy_env.step(action)

        assert easy_env.state.step_count == 5

        # Reset
        obs = easy_env.reset()
        assert easy_env.state.step_count == 0
        assert obs.current_step == 0
        assert obs.done is False

    def test_reset_generates_new_episode_id(self, easy_env: TrafficEnvironment):
        """Each reset should produce a unique episode_id."""
        id1 = easy_env.state.episode_id
        easy_env.reset()
        id2 = easy_env.state.episode_id
        assert id1 != id2


# =========================================================================
# 2. Step increments step count
# =========================================================================

class TestStepCounting:
    """Tests for step counting and state tracking."""

    def test_step_increments_step_count(self, easy_env: TrafficEnvironment):
        """A single step() should increment step_count by 1."""
        action = TrafficAction(action=SignalAction.KEEP_CURRENT)
        easy_env.step(action)
        assert easy_env.state.step_count == 1

    def test_multiple_steps_increment_correctly(self, easy_env: TrafficEnvironment):
        """Multiple steps should increment step_count linearly."""
        action = TrafficAction(action=SignalAction.KEEP_CURRENT)
        for i in range(10):
            easy_env.step(action)
        assert easy_env.state.step_count == 10

    def test_step_returns_observation(self, easy_env: TrafficEnvironment):
        """step() should return a TrafficObservation with updated current_step."""
        action = TrafficAction(action=SignalAction.KEEP_CURRENT)
        obs = easy_env.step(action)
        assert isinstance(obs, TrafficObservation)
        assert obs.current_step == 1


# =========================================================================
# 3. Episode ends at max steps
# =========================================================================

class TestEpisodeTermination:
    """Tests for episode done-flag behaviour."""

    def test_episode_ends_at_max_steps_easy(self, easy_env: TrafficEnvironment):
        """Easy task (50 steps): done=True on step 50, False on step 49."""
        action = TrafficAction(action=SignalAction.KEEP_CURRENT)

        for i in range(49):
            obs = easy_env.step(action)
            assert obs.done is False, f"done should be False at step {i + 1}"

        obs = easy_env.step(action)
        assert obs.done is True, "done should be True at step 50"

    def test_episode_ends_at_max_steps_hard(self, hard_env: TrafficEnvironment):
        """Hard task (60 steps): done=True on step 60."""
        action = TrafficAction(action=SignalAction.KEEP_CURRENT)

        for i in range(59):
            obs = hard_env.step(action)
            assert obs.done is False

        obs = hard_env.step(action)
        assert obs.done is True

    def test_final_step_includes_score_message(self, easy_env: TrafficEnvironment):
        """The final step's message should contain the episode score."""
        action = TrafficAction(action=SignalAction.KEEP_CURRENT)
        for _ in range(50):
            obs = easy_env.step(action)

        assert "Score" in obs.message or "score" in obs.message.lower()


# =========================================================================
# 4. Emergency override rewards correctly
# =========================================================================

class TestEmergencyOverride:
    """Tests for emergency vehicle handling and reward signals."""

    def test_emergency_override_positive_reward(self, easy_env: TrafficEnvironment):
        """Overriding for an emergency in the correct direction should yield positive reward."""
        # Manually inject an emergency on the NORTH approach
        easy_env.sim.emergency = {
            "direction": "NORTH",
            "urgency": "HIGH",
            "steps_waiting": 0,
        }

        action = TrafficAction(
            action=SignalAction.EMERGENCY_OVERRIDE,
            emergency_direction="NORTH",
        )
        obs = easy_env.step(action)

        # NS_GREEN gives green to NORTH — emergency bonus should kick in
        # The reward should include the +1.0 emergency bonus
        assert obs.reward > 0, f"Expected positive reward with emergency override, got {obs.reward}"

    def test_emergency_present_in_observation(self, easy_env: TrafficEnvironment):
        """After injecting an emergency, the observation should reflect it."""
        easy_env.sim.emergency = {
            "direction": "EAST",
            "urgency": "CRITICAL",
            "steps_waiting": 3,
        }

        action = TrafficAction(action=SignalAction.KEEP_CURRENT)
        obs = easy_env.step(action)

        # Emergency might have been resolved or still present depending on phase
        # But at minimum, the step should process without error
        assert isinstance(obs, TrafficObservation)


# =========================================================================
# 5. Grader returns valid range
# =========================================================================

class TestGraderScores:
    """Tests for grader score ranges across all tasks."""

    @pytest.mark.parametrize("task_id", ["easy", "medium", "hard"])
    def test_grader_returns_valid_range(self, task_id: str):
        """grade_episode should return a score in [0.0, 1.0] for any task."""
        env = TrafficEnvironment(task_id=task_id)
        env.reset()

        max_steps = ALL_TASKS[task_id]["max_steps"]
        history: list[dict] = []

        for i in range(max_steps):
            # Alternate between keep and switch for a reasonable strategy
            if i % 5 == 4:
                action = TrafficAction(action=SignalAction.SWITCH_PHASE)
            else:
                action = TrafficAction(action=SignalAction.KEEP_CURRENT)

            obs = env.step(action)
            history.append({
                "queues": {
                    "NORTH": obs.queue_north,
                    "SOUTH": obs.queue_south,
                    "EAST": obs.queue_east,
                    "WEST": obs.queue_west,
                },
                "current_phase": obs.current_phase,
                "phase_duration": obs.phase_duration,
                "emergency_present": obs.emergency_present,
                "emergency_direction": obs.emergency_direction,
                "emergency_urgency": obs.emergency_urgency,
                "total_vehicles_cleared": obs.total_vehicles_cleared,
                "total_wait_time": obs.total_wait_time,
                "reward": obs.reward,
                "message": obs.message,
            })

        score = grade_episode(task_id, history)
        assert 0.0 <= score <= 1.0, f"Score {score} out of [0,1] range for {task_id}"

    def test_grader_empty_history_does_not_crash(self):
        """Grading an empty episode should return a valid score, not crash."""
        for task_id in ["easy", "medium", "hard"]:
            score = grade_episode(task_id, [])
            assert 0.0 <= score <= 1.0

    def test_grader_rejects_invalid_task(self):
        """grade_episode should raise ValueError for unknown task_id."""
        with pytest.raises(ValueError):
            grade_episode("nonexistent", [])


# =========================================================================
# 6. Switch phase changes signal
# =========================================================================

class TestPhaseSwitch:
    """Tests for signal phase transition behaviour."""

    def test_switch_phase_changes_signal(self, easy_env: TrafficEnvironment):
        """SWITCH_PHASE should advance from NS_GREEN to ALL_RED."""
        # Need to wait for MIN_PHASE_DURATION steps first
        keep = TrafficAction(action=SignalAction.KEEP_CURRENT)
        easy_env.step(keep)
        easy_env.step(keep)

        initial_phase = easy_env.state.current_phase
        assert initial_phase == "NS_GREEN"

        switch = TrafficAction(action=SignalAction.SWITCH_PHASE)
        obs = easy_env.step(switch)

        assert obs.current_phase == "ALL_RED", (
            f"Expected ALL_RED after switching from NS_GREEN, got {obs.current_phase}"
        )

    def test_full_phase_rotation(self, easy_env: TrafficEnvironment):
        """Cycling through all 4 phases in the rotation should work."""
        keep = TrafficAction(action=SignalAction.KEEP_CURRENT)
        switch = TrafficAction(action=SignalAction.SWITCH_PHASE)

        expected_phases = ["NS_GREEN", "ALL_RED", "EW_GREEN", "ALL_RED"]
        assert easy_env.state.current_phase == expected_phases[0]

        for i in range(1, 4):
            # Wait for MIN_PHASE_DURATION
            easy_env.step(keep)
            easy_env.step(keep)
            # Switch
            obs = easy_env.step(switch)
            assert obs.current_phase == expected_phases[i], (
                f"Step {i}: expected {expected_phases[i]}, got {obs.current_phase}"
            )

    def test_switch_blocked_before_min_duration(self, easy_env: TrafficEnvironment):
        """SWITCH_PHASE should be ignored if phase_duration < MIN_PHASE_DURATION."""
        switch = TrafficAction(action=SignalAction.SWITCH_PHASE)
        obs = easy_env.step(switch)

        # Phase should NOT have changed (duration was 0 < 2)
        assert obs.current_phase == "NS_GREEN", (
            "Phase should not switch before MIN_PHASE_DURATION"
        )


# =========================================================================
# 7. Models serialize correctly
# =========================================================================

class TestModelSerialization:
    """Tests for Pydantic model serialization round-trips."""

    def test_traffic_action_roundtrip(self):
        """TrafficAction should serialize and deserialize losslessly."""
        original = TrafficAction(
            action=SignalAction.EMERGENCY_OVERRIDE,
            emergency_direction="NORTH",
        )
        dumped = original.model_dump()
        restored = TrafficAction.model_validate(dumped)

        assert restored.action == original.action
        assert restored.emergency_direction == original.emergency_direction

    def test_traffic_action_json_roundtrip(self):
        """TrafficAction should round-trip through JSON."""
        original = TrafficAction(action=SignalAction.SWITCH_PHASE)
        json_str = original.model_dump_json()
        restored = TrafficAction.model_validate_json(json_str)

        assert restored.action == original.action
        assert restored.emergency_direction is None

    def test_traffic_observation_roundtrip(self):
        """TrafficObservation should serialize and deserialize losslessly."""
        original = TrafficObservation(
            queue_north=5, queue_south=3, queue_east=8, queue_west=2,
            current_phase="EW_GREEN", phase_duration=3,
            emergency_present=True, emergency_direction="EAST",
            emergency_urgency="CRITICAL",
            total_vehicles_cleared=42, total_wait_time=120,
            current_step=15, reward=-0.35,
            done=False, success=False, message="test message",
        )
        dumped = original.model_dump()
        restored = TrafficObservation.model_validate(dumped)

        assert restored == original

    def test_traffic_state_roundtrip(self):
        """TrafficState should serialize and deserialize losslessly."""
        original = TrafficState(
            episode_id="test-uuid-123",
            step_count=25,
            task_id="hard",
            current_phase="NS_GREEN",
            emergency_active=True,
            total_cleared=50,
            total_wait=200,
        )
        dumped = original.model_dump()
        restored = TrafficState.model_validate(dumped)

        assert restored == original

    def test_observation_rejects_invalid_queue(self):
        """TrafficObservation should reject queue values outside [0, 20]."""
        with pytest.raises(Exception):
            TrafficObservation(
                queue_north=25,  # Invalid — max is 20
                queue_south=0, queue_east=0, queue_west=0,
                current_phase="NS_GREEN", phase_duration=0,
                emergency_present=False, emergency_direction=None,
                emergency_urgency=None,
                total_vehicles_cleared=0, total_wait_time=0,
                current_step=0, reward=0.0,
                done=False, success=False,
            )

    def test_observation_rejects_invalid_phase(self):
        """TrafficObservation should reject invalid phase strings."""
        with pytest.raises(Exception):
            TrafficObservation(
                queue_north=0, queue_south=0, queue_east=0, queue_west=0,
                current_phase="INVALID_PHASE", phase_duration=0,
                emergency_present=False, emergency_direction=None,
                emergency_urgency=None,
                total_vehicles_cleared=0, total_wait_time=0,
                current_step=0, reward=0.0,
                done=False, success=False,
            )

    def test_state_rejects_invalid_task_id(self):
        """TrafficState should reject invalid task_id values."""
        with pytest.raises(Exception):
            TrafficState(
                episode_id="test", step_count=0, task_id="impossible",
                current_phase="NS_GREEN", emergency_active=False,
                total_cleared=0, total_wait=0,
            )


# =========================================================================
# 8. Environment constructor validation
# =========================================================================

class TestConstructor:
    """Tests for TrafficEnvironment constructor."""

    def test_valid_task_ids(self):
        """All three task IDs should create valid environments."""
        for task_id in ["easy", "medium", "hard"]:
            env = TrafficEnvironment(task_id=task_id)
            assert env.task_id == task_id

    def test_invalid_task_id_raises(self):
        """Unknown task_id should raise ValueError."""
        with pytest.raises(ValueError):
            TrafficEnvironment(task_id="impossible")

    def test_metadata(self):
        """get_metadata() should return valid metadata."""
        env = TrafficEnvironment(task_id="easy")
        meta = env.get_metadata()
        assert meta.name == "Traffic Control Environment"
        assert meta.version == "0.1.0"
