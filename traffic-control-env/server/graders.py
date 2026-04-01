"""
graders.py — Grader functions for scoring episodes per task.

Each grader takes an episode_history (list of step dicts) and returns
a float score in [0.0, 1.0].

Functions:
    grade_easy(episode_history)   -> float
        Score = total_vehicles_cleared / max_possible_clearance (200)

    grade_medium(episode_history) -> float
        60% throughput + 40% emergency_response_score

    grade_hard(episode_history)   -> float
        40% throughput + 40% emergency_response + 20% queue_balance

    grade_episode(task_id, episode_history) -> float
        Router that dispatches to the correct grader and clamps output to [0.0, 1.0].
"""
