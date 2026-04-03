from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional

from openai import OpenAI

try:
    from traffic_control_env.models import (
        TaskId,
        TrafficCommand,
        TrafficControlAction,
        TrafficControlObservation,
    )
    from traffic_control_env.server import TrafficControlEnvironment
except ModuleNotFoundError:
    from models import TaskId, TrafficCommand, TrafficControlAction, TrafficControlObservation
    from server import TrafficControlEnvironment

API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "openai/gpt-4.1-mini")
HF_TOKEN = os.getenv("HF_TOKEN")
TASK_ID_FILTER = os.getenv("TASK_ID")
MAX_EXTRA_STEPS = 5
ENV_NAME = "traffic_control_env"

if HF_TOKEN is None:
    raise ValueError("HF_TOKEN environment variable is required")


@dataclass
class EpisodeResult:
    task_id: str
    success: bool
    steps: int
    rewards: list[float]


def heuristic_policy(observation: TrafficControlObservation) -> TrafficCommand:
    if observation.emergency_direction is not None:
        direction = observation.emergency_direction.value
        if direction in ("north", "south"):
            if observation.current_phase.value == "ns_green":
                return TrafficCommand.HOLD_CURRENT_PHASE
            return TrafficCommand.SET_NS_GREEN
        if direction in ("east", "west"):
            if observation.current_phase.value == "ew_green":
                return TrafficCommand.HOLD_CURRENT_PHASE
            return TrafficCommand.SET_EW_GREEN

    ns_pressure = (
        observation.queue_north
        + observation.queue_south
        + observation.avg_wait_north
        + observation.avg_wait_south
    )
    ew_pressure = (
        observation.queue_east
        + observation.queue_west
        + observation.avg_wait_east
        + observation.avg_wait_west
    )

    if ns_pressure >= ew_pressure:
        return (
            TrafficCommand.HOLD_CURRENT_PHASE
            if observation.current_phase.value == "ns_green"
            else TrafficCommand.SET_NS_GREEN
        )

    return (
        TrafficCommand.HOLD_CURRENT_PHASE
        if observation.current_phase.value == "ew_green"
        else TrafficCommand.SET_EW_GREEN
    )


def llm_policy(observation: TrafficControlObservation) -> Optional[TrafficCommand]:
    if HF_TOKEN is None or HF_TOKEN.strip().lower() in {"dummy", "test", "local"}:
        return None

    try:
        client = OpenAI(
            base_url=API_BASE_URL,
            api_key=HF_TOKEN,
        )
    except Exception:
        return None

    prompt = f"""
You control one 4-way traffic intersection.
Choose exactly one action from:
- set_ns_green
- set_ew_green
- hold_current_phase
- set_all_red

State:
- current_phase: {observation.current_phase.value}
- current_timestep: {observation.current_timestep}
- queue_north: {observation.queue_north}
- queue_south: {observation.queue_south}
- queue_east: {observation.queue_east}
- queue_west: {observation.queue_west}
- avg_wait_north: {observation.avg_wait_north:.2f}
- avg_wait_south: {observation.avg_wait_south:.2f}
- avg_wait_east: {observation.avg_wait_east:.2f}
- avg_wait_west: {observation.avg_wait_west:.2f}
- emergency_present: {str(observation.emergency_present).lower()}
- emergency_direction: {observation.emergency_direction.value if observation.emergency_direction else "none"}
- time_since_last_phase_change: {observation.time_since_last_phase_change}

Reply with only the action string.
""".strip()

    try:
        response = client.responses.create(
            model=MODEL_NAME,
            input=prompt,
        )
        text = getattr(response, "output_text", "").strip().lower()
        if not text:
            return None
        command_text = text.splitlines()[0].strip()
        allowed = {command.value: command for command in TrafficCommand}
        return allowed.get(command_text)
    except Exception:
        return None


def choose_action(observation: TrafficControlObservation) -> TrafficCommand:
    llm_choice = llm_policy(observation)
    if llm_choice is not None:
        return llm_choice
    return heuristic_policy(observation)


def format_bool(value: bool) -> str:
    return "true" if value else "false"


def format_reward(value: float) -> str:
    return f"{value:.2f}"


def format_rewards(values: list[float]) -> str:
    return ",".join(format_reward(value) for value in values)


def run_episode(task_id: TaskId) -> EpisodeResult:
    env = TrafficControlEnvironment()
    rewards: list[float] = []
    success = False
    step_count = 0

    print(f"[START] task={task_id.value} env={ENV_NAME} model={MODEL_NAME}")

    try:
        observation = env.reset(task_id=task_id.value)
        safety_limit = env._task.horizon_steps + MAX_EXTRA_STEPS

        while not observation.done and env.state.step_count < safety_limit:
            command = choose_action(observation)
            action_text = command.value
            error_text = "null"

            try:
                observation = env.step(TrafficControlAction(command=command))
                reward = float(observation.reward or 0.0)
            except Exception as exc:
                reward = 0.0
                observation = TrafficControlObservation(
                    reward=reward,
                    done=True,
                    status_message=str(exc),
                )
                error_text = str(exc)

            step_count += 1
            rewards.append(reward)

            print(
                f"[STEP] step={step_count} action={action_text} "
                f"reward={format_reward(reward)} done={format_bool(observation.done)} "
                f"error={error_text}"
            )

        success = bool(env.state.final_score is not None and env.state.final_score > 0.0)
        return EpisodeResult(
            task_id=task_id.value,
            success=success,
            steps=step_count,
            rewards=rewards,
        )
    finally:
        print(
            f"[END] success={format_bool(success)} steps={step_count} "
            f"rewards={format_rewards(rewards)}"
        )


def main() -> None:
    task_ids = [TaskId(TASK_ID_FILTER)] if TASK_ID_FILTER else list(TaskId)
    for task_id in task_ids:
        run_episode(task_id)


if __name__ == "__main__":
    main()
