"""
models.py — Pydantic data models for the Traffic Control Environment.

Defines the typed request/response models used across the server and client
for the OpenEnv-compliant traffic signal control environment.

Models:
    SignalCommand  — Enum of valid signal control commands (aliased as SignalAction).
    VehicleType    — Enum for vehicle classification (normal / emergency).
    Direction      — Enum for approach directions at the intersection.
    VehicleRecord  — Tracks an individual vehicle in a queue.
    TrafficAction  — Agent action submitted each step.
    TrafficObservation — Full observation returned after each step.
    TrafficState   — Internal environment state exposed via /state endpoint.
"""

from __future__ import annotations

from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field, field_validator


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class SignalCommand(str, Enum):
    """Discrete commands the agent can issue to control the intersection signal.

    Members:
        SET_NS_GREEN:        Switch signal to north-south green (east-west red).
        SET_EW_GREEN:        Switch signal to east-west green (north-south red).
        HOLD_CURRENT_PHASE:  Maintain the current signal phase unchanged.
        SET_ALL_RED:         Force all approaches to red (safety clearance).
    """

    SET_NS_GREEN = "set_ns_green"
    SET_EW_GREEN = "set_ew_green"
    HOLD_CURRENT_PHASE = "hold_current_phase"
    SET_ALL_RED = "set_all_red"


# Backward-compatible alias so existing imports keep working.
SignalAction = SignalCommand


class VehicleType(str, Enum):
    """Type of vehicle in the intersection queue."""

    NORMAL = "normal"
    EMERGENCY = "emergency"


class Direction(str, Enum):
    """Approach direction at the intersection."""

    NORTH = "NORTH"
    SOUTH = "SOUTH"
    EAST = "EAST"
    WEST = "WEST"


# Valid compass directions used for emergency vehicle positions.
VALID_DIRECTIONS = {"NORTH", "SOUTH", "EAST", "WEST"}

# Valid signal phases for the intersection (case-insensitive support).
VALID_PHASES = {"NS_GREEN", "EW_GREEN", "ALL_RED", "ns_green", "ew_green", "all_red"}

# Valid emergency urgency levels.
VALID_URGENCIES = {"LOW", "HIGH", "CRITICAL"}


# ---------------------------------------------------------------------------
# Vehicle Record Model
# ---------------------------------------------------------------------------

class VehicleRecord(BaseModel):
    """Tracks an individual vehicle in a queue.

    Attributes:
        vehicle_id:    Unique identifier string.
        direction:     Which approach the vehicle is on.
        vehicle_type:  NORMAL or EMERGENCY.
        wait_time:     Steps this vehicle has been waiting.
        arrival_step:  Simulation step when vehicle arrived.
    """

    vehicle_id: str
    direction: Direction
    vehicle_type: VehicleType = VehicleType.NORMAL
    wait_time: int = Field(0, ge=0)
    arrival_step: int = Field(0, ge=0)


# ---------------------------------------------------------------------------
# Action Model
# ---------------------------------------------------------------------------

class TrafficAction(BaseModel):
    """Action submitted by the agent at each simulation step.

    Attributes:
        action:              The signal control command to execute.
        emergency_direction: When using an emergency-related action, the direction
                             the agent believes has the emergency vehicle.
                             Must be one of NORTH, SOUTH, EAST, WEST or None.
    """

    action: SignalCommand = Field(
        ...,
        description="Signal control command to execute this step.",
    )
    emergency_direction: Optional[str] = Field(
        default=None,
        description=(
            "Direction of the emergency vehicle when using EMERGENCY_OVERRIDE. "
            "Must be one of: NORTH, SOUTH, EAST, WEST, or null."
        ),
    )

    @field_validator("emergency_direction")
    @classmethod
    def validate_emergency_direction(cls, value: Optional[str]) -> Optional[str]:
        """Ensure emergency_direction is a valid compass direction or None."""
        if value is not None and value not in VALID_DIRECTIONS:
            raise ValueError(
                f"emergency_direction must be one of {VALID_DIRECTIONS} or None, "
                f"got '{value}'"
            )
        return value


# ---------------------------------------------------------------------------
# Observation Model
# ---------------------------------------------------------------------------

class TrafficObservation(BaseModel):
    """Full observation returned by the environment after each step.

    Contains queue states, signal info, emergency status, cumulative
    performance metrics, per-direction average wait times, remaining steps,
    and per-step reward/episode signals.

    Attributes:
        queue_north:            Vehicles queued on the NORTH approach (0–20).
        queue_south:            Vehicles queued on the SOUTH approach (0–20).
        queue_east:             Vehicles queued on the EAST approach (0–20).
        queue_west:             Vehicles queued on the WEST approach (0–20).
        current_phase:          Active signal phase (NS_GREEN | EW_GREEN | ALL_RED).
        phase_duration:         Steps the current phase has been active.
        emergency_present:      Whether an emergency vehicle is at the intersection.
        emergency_direction:    Direction of the emergency vehicle, if present.
        emergency_urgency:      Urgency level of the emergency (LOW | HIGH | CRITICAL).
        total_vehicles_cleared: Cumulative vehicles discharged this episode.
        total_wait_time:        Cumulative vehicle-steps of waiting this episode.
        avg_wait_north:         Average wait time (steps) for vehicles on NORTH approach.
        avg_wait_south:         Average wait time (steps) for vehicles on SOUTH approach.
        avg_wait_east:          Average wait time (steps) for vehicles on EAST approach.
        avg_wait_west:          Average wait time (steps) for vehicles on WEST approach.
        steps_remaining:        Steps remaining until episode ends.
        current_step:           Current simulation step within the episode.
        reward:                 Reward earned on this step.
        done:                   Whether the episode has ended.
        success:                Whether the episode ended successfully.
        message:                Optional human-readable status message.
    """

    # --- Queue lengths (one per approach) ---
    queue_north: int = Field(
        ..., ge=0, le=20, description="Vehicles queued on the NORTH approach (0–20)."
    )
    queue_south: int = Field(
        ..., ge=0, le=20, description="Vehicles queued on the SOUTH approach (0–20)."
    )
    queue_east: int = Field(
        ..., ge=0, le=20, description="Vehicles queued on the EAST approach (0–20)."
    )
    queue_west: int = Field(
        ..., ge=0, le=20, description="Vehicles queued on the WEST approach (0–20)."
    )

    # --- Signal state ---
    current_phase: str = Field(
        ..., description="Active signal phase: NS_GREEN, EW_GREEN, or ALL_RED."
    )
    phase_duration: int = Field(
        ..., ge=0, description="Number of steps the current phase has been active."
    )

    # --- Emergency vehicle info ---
    emergency_present: bool = Field(
        ..., description="True if an emergency vehicle is at the intersection."
    )
    emergency_direction: Optional[str] = Field(
        default=None,
        description="Direction of the emergency vehicle (NORTH/SOUTH/EAST/WEST), if present.",
    )
    emergency_urgency: Optional[str] = Field(
        default=None,
        description="Urgency level of the emergency: LOW, HIGH, or CRITICAL.",
    )

    # --- Cumulative performance metrics ---
    total_vehicles_cleared: int = Field(
        ..., ge=0, description="Total vehicles discharged so far this episode."
    )
    total_wait_time: int = Field(
        ..., ge=0, description="Cumulative vehicle-steps of waiting this episode."
    )

    # --- Per-direction average wait times ---
    avg_wait_north: float = Field(
        0.0, ge=0.0,
        description="Average wait time (steps) for vehicles on NORTH approach.",
    )
    avg_wait_south: float = Field(
        0.0, ge=0.0,
        description="Average wait time (steps) for vehicles on SOUTH approach.",
    )
    avg_wait_east: float = Field(
        0.0, ge=0.0,
        description="Average wait time (steps) for vehicles on EAST approach.",
    )
    avg_wait_west: float = Field(
        0.0, ge=0.0,
        description="Average wait time (steps) for vehicles on WEST approach.",
    )

    # --- Episode countdown ---
    steps_remaining: int = Field(
        0, ge=0,
        description="Steps remaining until episode ends.",
    )

    current_step: int = Field(
        ..., ge=0, description="Current simulation step within the episode."
    )

    # --- Per-step reward and episode signals ---
    reward: float = Field(
        default=0.0, description="Reward earned on this simulation step."
    )
    done: bool = Field(
        default=False, description="Whether the episode has terminated."
    )
    success: bool = Field(
        default=False, description="Whether the episode ended successfully."
    )
    message: str = Field(
        default="", description="Optional human-readable status message."
    )

    # --- Validators ---

    @field_validator("current_phase")
    @classmethod
    def validate_current_phase(cls, value: str) -> str:
        """Ensure current_phase is a recognized signal phase, normalized to uppercase."""
        normalized = value.upper()
        valid = {"NS_GREEN", "EW_GREEN", "ALL_RED"}
        if normalized not in valid:
            raise ValueError(
                f"current_phase must be one of {valid}, got '{value}'"
            )
        return normalized

    @field_validator("emergency_direction")
    @classmethod
    def validate_emergency_direction(cls, value: Optional[str]) -> Optional[str]:
        """Ensure emergency_direction is a valid compass direction or None."""
        if value is not None and value not in VALID_DIRECTIONS:
            raise ValueError(
                f"emergency_direction must be one of {VALID_DIRECTIONS} or None, "
                f"got '{value}'"
            )
        return value

    @field_validator("emergency_urgency")
    @classmethod
    def validate_emergency_urgency(cls, value: Optional[str]) -> Optional[str]:
        """Ensure emergency_urgency is a valid urgency level or None."""
        if value is not None and value not in VALID_URGENCIES:
            raise ValueError(
                f"emergency_urgency must be one of {VALID_URGENCIES} or None, "
                f"got '{value}'"
            )
        return value


# ---------------------------------------------------------------------------
# State Model
# ---------------------------------------------------------------------------

class TrafficState(BaseModel):
    """Internal environment state exposed via the ``/state`` endpoint.

    This is a compact summary of the simulation state, used for debugging
    and monitoring rather than agent decision-making.

    Attributes:
        episode_id:       Unique identifier for the current episode (UUID).
        step_count:       Number of steps taken so far in this episode.
        task_id:          Active task identifier (easy | medium | hard).
        current_phase:    Active signal phase (NS_GREEN | EW_GREEN | ALL_RED).
        emergency_active: Whether an emergency vehicle is currently present.
        total_cleared:    Cumulative vehicles discharged this episode.
        total_wait:       Cumulative vehicle-steps of waiting this episode.
    """

    episode_id: str = Field(
        ..., description="Unique identifier (UUID) for the current episode."
    )
    step_count: int = Field(
        ..., ge=0, description="Steps taken so far in this episode."
    )
    task_id: str = Field(
        ..., description="Active task: 'easy', 'medium', or 'hard'."
    )
    current_phase: str = Field(
        ..., description="Active signal phase: NS_GREEN, EW_GREEN, or ALL_RED."
    )
    emergency_active: bool = Field(
        ..., description="Whether an emergency vehicle is currently present."
    )
    total_cleared: int = Field(
        ..., ge=0, description="Cumulative vehicles discharged this episode."
    )
    total_wait: int = Field(
        ..., ge=0, description="Cumulative vehicle-steps of waiting this episode."
    )

    @field_validator("current_phase")
    @classmethod
    def validate_current_phase(cls, value: str) -> str:
        """Ensure current_phase is a recognized signal phase."""
        if value not in VALID_PHASES:
            raise ValueError(
                f"current_phase must be one of {VALID_PHASES}, got '{value}'"
            )
        return value

    @field_validator("task_id")
    @classmethod
    def validate_task_id(cls, value: str) -> str:
        """Ensure task_id is one of the defined difficulty levels."""
        valid_tasks = {"easy", "medium", "hard"}
        if value not in valid_tasks:
            raise ValueError(
                f"task_id must be one of {valid_tasks}, got '{value}'"
            )
        return value
