"""
tasks.py — Task definitions for the Traffic Control Environment.

Defines three tasks of increasing difficulty:

Task 1 (EASY):   Steady State Control
    - Balanced, predictable traffic (1.5 vehicles/step per direction)
    - Low arrival noise (std=0.3)
    - No emergency vehicles
    - 50 steps per episode
    - Graded purely on throughput

Task 2 (MEDIUM): Rush Hour + Emergencies
    - Asymmetric traffic load (heavy N/S, light E/W)
    - Moderate arrival noise (std=0.8)
    - ~4 emergency vehicles per episode (probability=0.08)
    - 50 steps per episode
    - Graded 60% throughput, 40% emergency response

Task 3 (HARD):   Crisis Management
    - High traffic on all approaches (3.5–4.0 vehicles/step)
    - High arrival noise (std=1.5, traffic surges)
    - Frequent emergencies (probability=0.15), up to 2 simultaneous
    - 60 steps per episode
    - Graded 40% throughput, 40% emergency, 20% queue balance

Each task is a dict with keys:
    task_id, description, max_steps, arrival_rates, arrival_noise_std,
    emergency_probability, success_threshold, and task-specific extras.
"""
