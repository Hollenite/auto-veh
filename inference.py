from __future__ import annotations

import asyncio
import os
from dataclasses import dataclass
from typing import Optional

from openai import OpenAI

from models import SignalCommand, TrafficAction, TrafficObservation
from client.client import TrafficEnv
from server.tasks import ALL_TASKS

API_BASE_URL = os.getenv("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4.1-mini")
HF_TOKEN = os.getenv("HF_TOKEN")
LOCAL_IMAGE_NAME = os.getenv("LOCAL_IMAGE_NAME")
TASK_ID_FILTER = os.getenv("TASK_ID")
MAX_EXTRA_STEPS = 5
ENV_NAME = "traffic-control-env"




@dataclass
class EpisodeResult:
    task_id: str
    success: bool
    steps: int
    rewards: list[float]


def heuristic_policy(observation: TrafficObservation) -> SignalCommand:
    if observation.emergency_present and observation.emergency_direction:
        direction = observation.emergency_direction
        if direction in ("NORTH", "SOUTH"):
            if observation.current_phase == "NS_GREEN":
                return SignalCommand.HOLD_CURRENT_PHASE
            return SignalCommand.SET_NS_GREEN
        if direction in ("EAST", "WEST"):
            if observation.current_phase == "EW_GREEN":
                return SignalCommand.HOLD_CURRENT_PHASE
            return SignalCommand.SET_EW_GREEN

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
            SignalCommand.HOLD_CURRENT_PHASE
            if observation.current_phase == "NS_GREEN"
            else SignalCommand.SET_NS_GREEN
        )

    return (
        SignalCommand.HOLD_CURRENT_PHASE
        if observation.current_phase == "EW_GREEN"
        else SignalCommand.SET_EW_GREEN
    )


def llm_policy(observation: TrafficObservation) -> Optional[SignalCommand]:
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
- current_phase: {observation.current_phase}
- queue_north: {observation.queue_north}
- queue_south: {observation.queue_south}
- queue_east: {observation.queue_east}
- queue_west: {observation.queue_west}
- avg_wait_north: {observation.avg_wait_north:.2f}
- avg_wait_south: {observation.avg_wait_south:.2f}
- avg_wait_east: {observation.avg_wait_east:.2f}
- avg_wait_west: {observation.avg_wait_west:.2f}
- emergency_present: {str(observation.emergency_present).lower()}
- emergency_direction: {observation.emergency_direction if observation.emergency_direction else "none"}

Reply with only the action string.
""".strip()

    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            max_tokens=50,
        )
        text = (response.choices[0].message.content or "").strip().lower()
        if not text:
            return None
        command_text = text.splitlines()[0].strip()
        allowed = {command.value: command for command in SignalCommand}
        return allowed.get(command_text)
    except Exception:
        return None


def choose_action(observation: TrafficObservation) -> SignalCommand:
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


MAX_REWARD_PER_STEP = 3.0  # upper bound per step from openenv.yaml reward range


async def run_episode(task_id: str) -> EpisodeResult:
    rewards: list[float] = []
    success = False
    step_count = 0
    score = 0.0
    env = None

    try:
        print(f"[START] task={task_id} env={ENV_NAME} model={MODEL_NAME}", flush=True)

        env = await TrafficEnv.from_docker_image(LOCAL_IMAGE_NAME, task_id=task_id)
        result = await env.reset()
        observation = result.observation

        task_config = ALL_TASKS[task_id]
        max_steps = task_config["max_steps"]
        safety_limit = max_steps + MAX_EXTRA_STEPS

        while not observation.done and step_count < safety_limit:
            command = choose_action(observation)
            action_text = command.value
            error_text = "null"

            try:
                result = await env.step(TrafficAction(action=command))
                observation = result.observation
                reward = float(observation.reward or 0.0)
            except Exception as exc:
                reward = 0.0
                error_text = str(exc)
                observation.done = True

            step_count += 1
            rewards.append(reward)

            print(
                f"[STEP] step={step_count} action={action_text} "
                f"reward={format_reward(reward)} done={format_bool(observation.done)} "
                f"error={error_text}",
                flush=True,
            )

        max_total_reward = task_config["max_steps"] * MAX_REWARD_PER_STEP
        score = min(1.0, max(0.0, sum(rewards) / max_total_reward)) if max_total_reward > 0 else 0.0

        threshold = task_config.get("success_threshold", 0.0)
        success = (observation.reward is not None and observation.reward > threshold) if hasattr(observation, 'reward') else False

        return EpisodeResult(
            task_id=task_id,
            success=success,
            steps=step_count,
            rewards=rewards,
        )
    finally:
        if env is not None:
            try:
                await env.close()
            except Exception as e:
                print(f"[DEBUG] env.close() error: {e}", flush=True)
        print(
            f"[END] success={format_bool(success)} steps={step_count} "
            f"score={score:.2f} rewards={format_rewards(rewards)}",
            flush=True,
        )


async def main() -> None:
    if HF_TOKEN is None:
        print("[DEBUG] HF_TOKEN not set — LLM policy disabled, falling back to heuristic.", flush=True)
    task_ids = [TASK_ID_FILTER] if TASK_ID_FILTER else ["easy", "medium", "hard"]
    for task_id in task_ids:
        await run_episode(task_id)


if __name__ == "__main__":
    asyncio.run(main())
