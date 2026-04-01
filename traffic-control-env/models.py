"""
models.py — Pydantic data models for the Traffic Control Environment.

Defines the typed models used across the server and client:

Enums:
    SignalAction: KEEP_CURRENT, SWITCH_PHASE, EMERGENCY_OVERRIDE

Models:
    TrafficAction:
        - action: SignalAction
        - emergency_direction: Optional[str] (NORTH/SOUTH/EAST/WEST)

    TrafficObservation:
        - Queue lengths (queue_north, queue_south, queue_east, queue_west)
        - Signal state (current_phase, phase_duration)
        - Emergency info (emergency_present, emergency_direction, emergency_urgency)
        - Performance metrics (total_vehicles_cleared, total_wait_time, current_step)
        - Episode info (reward, done, success, message)

    TrafficState:
        - episode_id, step_count, task_id
        - current_phase, emergency_active
        - total_cleared, total_wait
"""
