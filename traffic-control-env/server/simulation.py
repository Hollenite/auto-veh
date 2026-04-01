"""
simulation.py — IntersectionSimulation logic engine.

This module contains the core simulation for a 4-way intersection,
handling vehicle queues, signal phases, emergency vehicles, and
reward computation.

Responsibilities:
- Maintain queue state for 4 approaches (NORTH, SOUTH, EAST, WEST)
- Process signal phase transitions (NS_GREEN, EW_GREEN, ALL_RED)
- Discharge vehicles from queues based on active green phase
- Arrive new vehicles based on configurable arrival rates + noise
- Spawn and track emergency vehicles with urgency levels
- Calculate per-step reward (throughput, wait penalty, emergency handling, switch penalty)

Key Constants:
    DISCHARGE_RATE_PER_DIRECTION = 2   # Vehicles cleared per green direction per step
    MAX_QUEUE_SIZE = 20                # Maximum vehicles in any single queue
    MIN_PHASE_DURATION = 2             # Minimum steps before phase switch allowed
    PHASE_SWITCH_PENALTY = 0.05        # Reward penalty for unnecessary switching
    EMERGENCY_TIMEOUT = 8              # Steps before emergency penalty escalates severely

Classes:
    IntersectionSimulation: Core physics/logic engine for the intersection.
"""
