from __future__ import annotations

import asyncio
import os
from dataclasses import dataclass
from typing import Optional

from openai import OpenAI

from models import SignalCommand, TrafficAction, TrafficObservation
from client.client import TrafficEnv
from server.tasks import ALL_TASKS

API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "openai/gpt-4-turbo")
API_KEY = os.getenv("API_KEY")
HF_TOKEN = os.getenv("HF_TOKEN")  # Primary auth method
LOCAL_IMAGE_NAME = os.getenv("LOCAL_IMAGE_NAME", "traffic-control-env")
TASK_ID_FILTER = os.getenv("TASK_ID")
MAX_EXTRA_STEPS = 5
ENV_NAME = "traffic-control-env"

# Ensure we have valid credentials
if not (API_KEY or HF_TOKEN):
    raise ValueError("API_KEY or HF_TOKEN environment variable is required")
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
    # Use API_KEY or HF_TOKEN (validator provides one of these)
    api_key = API_KEY or HF_TOKEN
    
    if not api_key or api_key.strip().lower() in {"dummy", "test", "local"}:
        print(f"[DEBUG] Skipping LLM (no valid credentials)", flush=True)
        return None

    try:
        print(f"[DEBUG] Creating inference client: {API_BASE_URL}", flush=True)
        client = OpenAI(
            base_url=API_BASE_URL,
            api_key=api_key,
        )
    except Exception as e:
        print(f"[ERROR] Failed to create client: {e}", flush=True)
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
        print(f"[DEBUG] Calling {MODEL_NAME} via LLM proxy", flush=True)
        # Use responses.create() - this is what HuggingFace LLM router expects
        response = client.responses.create(
            model=MODEL_NAME,
            input=prompt,
        )
        print(f"[DEBUG] LLM response received", flush=True)
        text = getattr(response, "output_text", "").strip().lower()
        if not text:
            return None
        command_text = text.splitlines()[0].strip()
        allowed = {command.value: command for command in SignalCommand}
        result = allowed.get(command_text)
        print(f"[DEBUG] LLM action: {command_text} -> {result}", flush=True)
        return result
    except Exception as e:
        print(f"[ERROR] LLM call failed: {e}", flush=True)
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
    _EPSILON = 1e-6
    score = _EPSILON
    env = None
    observation = None
    
    try:
        print(f"[START] task={task_id} env={ENV_NAME} model={MODEL_NAME}", flush=True)

        try:
            env = await TrafficEnv.from_docker_image(LOCAL_IMAGE_NAME)
        except Exception as e:
            print(f"[ERROR] Failed to create environment: {e}", flush=True)
            raise

        try:
            result = await env.reset(task_id=task_id)
            observation = result.observation
        except Exception as e:
            print(f"[ERROR] Failed to reset environment: {e}", flush=True)
            raise

        task_config = ALL_TASKS[task_id]
        max_steps = task_config["max_steps"]
        safety_limit = max_steps + MAX_EXTRA_STEPS

        while observation and not observation.done and step_count < safety_limit:
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
                if observation:
                    observation.done = True

            step_count += 1
            rewards.append(reward)

            print(
                f"[STEP] step={step_count} action={action_text} "
                f"reward={format_reward(reward)} done={format_bool(observation.done if observation else True)} "
                f"error={error_text}",
                flush=True,
            )

        max_total_reward = task_config["max_steps"] * MAX_REWARD_PER_STEP
        score = min(1.0 - _EPSILON, max(_EPSILON, sum(rewards) / max_total_reward)) if max_total_reward > 0 else _EPSILON

        success = (observation.reward is not None and observation.reward > task_config.get("success_threshold", 0.0)) if observation else False

        return EpisodeResult(
            task_id=task_id,
            success=success,
            steps=step_count,
            rewards=rewards,
        )
    except Exception as e:
        print(f"[ERROR] Episode execution failed: {e}", flush=True)
        import traceback
        traceback.print_exc()
        return EpisodeResult(
            task_id=task_id,
            success=False,
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
            f"score={score:.6f} rewards={format_rewards(rewards)}",
            flush=True,
        )


async def main() -> None:
    try:
        print(f"[START] API_BASE_URL={API_BASE_URL} MODEL={MODEL_NAME}", flush=True)
        task_ids = [TASK_ID_FILTER] if TASK_ID_FILTER else ["easy", "medium", "hard"]
        for task_id in task_ids:
            try:
                await run_episode(task_id)
            except Exception as e:
                print(f"[ERROR] Episode failed for task {task_id}: {e}", flush=True)
                import traceback
                traceback.print_exc()
    except Exception as e:
        print(f"[FATAL] Main execution failed: {e}", flush=True)
        import traceback
        traceback.print_exc()
        raise


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except Exception as e:
        print(f"[CRITICAL] Script failed: {e}", flush=True)
        import traceback
        traceback.print_exc()
        exit(1)
