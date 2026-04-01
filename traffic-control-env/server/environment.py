"""
environment.py — Main TrafficEnvironment class.

This module implements the OpenEnv Environment interface for the
autonomous traffic signal control problem.

Responsibilities:
- Wraps IntersectionSimulation with OpenEnv-compliant reset()/step()/state() interface
- Manages episode lifecycle (episode IDs, step counting, done detection)
- Converts raw simulation state dicts into typed TrafficObservation models
- Routes to the correct task configuration based on task_id

Classes:
    TrafficEnvironment: OpenEnv Environment subclass controlling a 4-way intersection.
"""
