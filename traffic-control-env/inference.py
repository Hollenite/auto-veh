"""
inference.py — Baseline LLM agent for the Traffic Control Environment.

Runs a baseline agent powered by an OpenAI-compatible LLM against all
3 tasks (easy, medium, hard) and prints the grader scores.

The agent receives structured observations at each step and must respond
with a JSON action. It uses a carefully crafted system prompt that encodes
the traffic control domain knowledge.

Required environment variables:
    API_BASE_URL   — LLM endpoint (e.g. https://api.openai.com/v1)
    MODEL_NAME     — Model identifier (e.g. gpt-4o-mini)
    HF_TOKEN       — Hugging Face token
    OPENAI_API_KEY — API key (same as HF_TOKEN if using HF inference)

Optional:
    HF_SPACE_URL   — URL of deployed HF Space (default: http://localhost:8000)

Usage:
    API_BASE_URL=... MODEL_NAME=... OPENAI_API_KEY=... python inference.py
"""

from __future__ import annotations

import json
import logging
import os
import re
import sys
import time
from typing import Any

from dotenv import load_dotenv
from openai import OpenAI

from models import SignalAction, TrafficAction, TrafficObservation
from client.client import TrafficEnv
from server.graders import grade_episode
from server.tasks import ALL_TASKS

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger("inference")

# ---------------------------------------------------------------------------
# System Prompt
# ---------------------------------------------------------------------------

SYSTEM_PROMPT: str = """\
You are an AI traffic signal controller managing a 4-way intersection.
Each step you receive:
- Queue lengths at NORTH, SOUTH, EAST, WEST approaches (integer counts)
- Current signal phase: NS_GREEN, EW_GREEN, or ALL_RED
- How many steps the current phase has been active (phase_duration)
- Whether an emergency vehicle is present (and from which direction)

Your goal: maximize vehicle throughput and ALWAYS prioritize emergency vehicles.
An emergency vehicle must get green light as fast as possible.

Respond with exactly one JSON object, no other text:
{"action": "<KEEP_CURRENT|SWITCH_PHASE|EMERGENCY_OVERRIDE>", "emergency_direction": "<NORTH|SOUTH|EAST|WEST|null>"}

Action guide:
- KEEP_CURRENT: maintain current phase (good when current green direction has heavy traffic)
- SWITCH_PHASE: rotate to next phase in cycle (NS_GREEN → ALL_RED → EW_GREEN → ALL_RED)
- EMERGENCY_OVERRIDE: immediately switch to give emergency vehicle direction a green light
  (you MUST set emergency_direction to the direction with the emergency vehicle)

Strategy tips:
- Keep a phase active for at least 2 steps before switching (switching too fast is penalized)
- Give green to the direction with the longest queues
- ALWAYS use EMERGENCY_OVERRIDE when an emergency vehicle is present
- During ALL_RED, switch to the next phase as soon as possible
"""

# Valid action values for quick lookup
_VALID_ACTIONS: set[str] = {a.value for a in SignalAction}


# ---------------------------------------------------------------------------
# Prompt Construction
# ---------------------------------------------------------------------------

def build_user_prompt(obs: TrafficObservation, step: int) -> str:
    """Convert a TrafficObservation into a clear text prompt for the LLM.

    Presents all observation fields in a structured, readable format and
    includes contextual hints to guide the LLM's decision.

    Args:
        obs: The current environment observation.
        step: Current step number within the episode.

    Returns:
        Formatted string prompt for the LLM user message.
    """
    # Determine which directions currently have green
    if obs.current_phase == "NS_GREEN":
        green_dirs = "NORTH, SOUTH"
        red_dirs = "EAST, WEST"
        ns_total = obs.queue_north + obs.queue_south
        ew_total = obs.queue_east + obs.queue_west
    elif obs.current_phase == "EW_GREEN":
        green_dirs = "EAST, WEST"
        red_dirs = "NORTH, SOUTH"
        ns_total = obs.queue_north + obs.queue_south
        ew_total = obs.queue_east + obs.queue_west
    else:
        green_dirs = "NONE"
        red_dirs = "ALL"
        ns_total = obs.queue_north + obs.queue_south
        ew_total = obs.queue_east + obs.queue_west

    # Build the prompt
    lines = [
        f"=== Step {step} ===",
        f"",
        f"Queue lengths:",
        f"  NORTH: {obs.queue_north} vehicles",
        f"  SOUTH: {obs.queue_south} vehicles",
        f"  EAST:  {obs.queue_east} vehicles",
        f"  WEST:  {obs.queue_west} vehicles",
        f"  (N+S total: {ns_total}, E+W total: {ew_total})",
        f"",
        f"Signal state:",
        f"  Current phase: {obs.current_phase}",
        f"  Phase duration: {obs.phase_duration} steps",
        f"  Green for: {green_dirs}",
        f"  Red for: {red_dirs}",
    ]

    # Emergency information
    if obs.emergency_present:
        lines.extend([
            f"",
            f"⚠️  EMERGENCY VEHICLE DETECTED!",
            f"  Direction: {obs.emergency_direction}",
            f"  Urgency: {obs.emergency_urgency}",
            f"  → You MUST use EMERGENCY_OVERRIDE with emergency_direction=\"{obs.emergency_direction}\"",
        ])
    else:
        lines.extend([
            f"",
            f"No emergency vehicles present.",
        ])

    # Performance context
    lines.extend([
        f"",
        f"Performance:",
        f"  Total vehicles cleared: {obs.total_vehicles_cleared}",
        f"  Total wait time: {obs.total_wait_time}",
        f"  Last reward: {obs.reward:.4f}",
    ])

    # Contextual hint
    lines.append("")
    if obs.emergency_present:
        lines.append(
            f"HINT: Emergency at {obs.emergency_direction}! "
            f"Use EMERGENCY_OVERRIDE immediately."
        )
    elif obs.current_phase == "ALL_RED":
        lines.append("HINT: ALL_RED phase — SWITCH_PHASE to get traffic moving.")
    elif obs.phase_duration >= 3:
        if obs.current_phase == "NS_GREEN" and ew_total > ns_total + 2:
            lines.append(
                "HINT: E/W queues are building up. Consider SWITCH_PHASE."
            )
        elif obs.current_phase == "EW_GREEN" and ns_total > ew_total + 2:
            lines.append(
                "HINT: N/S queues are building up. Consider SWITCH_PHASE."
            )
        else:
            lines.append("HINT: Current phase is handling traffic well. KEEP_CURRENT is fine.")
    else:
        lines.append(
            f"HINT: Phase active for only {obs.phase_duration} steps. "
            f"Switching too soon is penalized."
        )

    lines.append("")
    lines.append("Respond with exactly one JSON object:")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Action Parsing
# ---------------------------------------------------------------------------

def parse_action(response_text: str) -> TrafficAction:
    """Parse LLM JSON response into a TrafficAction.

    Attempts multiple extraction strategies:
    1. Direct JSON parse of the full response.
    2. Regex extraction of a JSON object from surrounding text.
    3. Fallback to KEEP_CURRENT on any failure.

    Args:
        response_text: Raw text response from the LLM.

    Returns:
        Validated ``TrafficAction``. Defaults to ``KEEP_CURRENT`` on error.
    """
    fallback = TrafficAction(action=SignalAction.KEEP_CURRENT)

    if not response_text or not response_text.strip():
        logger.warning("Empty LLM response, using fallback KEEP_CURRENT")
        return fallback

    text = response_text.strip()

    # Strategy 1: Direct parse
    parsed = _try_parse_json(text)

    # Strategy 2: Extract JSON from markdown code blocks or surrounding text
    if parsed is None:
        json_match = re.search(r"\{[^{}]*\}", text, re.DOTALL)
        if json_match:
            parsed = _try_parse_json(json_match.group())

    if parsed is None:
        logger.warning("Failed to parse LLM response: %s", text[:200])
        return fallback

    # Extract and validate action
    action_str = parsed.get("action", "").lower()

    # Normalise common variations
    action_map = {
        "keep_current": "keep_current",
        "keepcurrent": "keep_current",
        "keep": "keep_current",
        "switch_phase": "switch_phase",
        "switchphase": "switch_phase",
        "switch": "switch_phase",
        "emergency_override": "emergency_override",
        "emergencyoverride": "emergency_override",
        "emergency": "emergency_override",
        "override": "emergency_override",
    }

    normalised = action_map.get(action_str, action_str)
    if normalised not in _VALID_ACTIONS:
        logger.warning("Invalid action '%s', using KEEP_CURRENT", action_str)
        return fallback

    # Extract emergency direction
    emergency_dir = parsed.get("emergency_direction")
    if isinstance(emergency_dir, str):
        emergency_dir = emergency_dir.upper()
        if emergency_dir in ("NULL", "NONE", ""):
            emergency_dir = None
        elif emergency_dir not in {"NORTH", "SOUTH", "EAST", "WEST"}:
            emergency_dir = None

    return TrafficAction(
        action=SignalAction(normalised),
        emergency_direction=emergency_dir,
    )


def _try_parse_json(text: str) -> dict[str, Any] | None:
    """Attempt to parse a string as JSON, returning None on failure."""
    try:
        result = json.loads(text)
        if isinstance(result, dict):
            return result
    except (json.JSONDecodeError, TypeError):
        pass
    return None


# ---------------------------------------------------------------------------
# Task Runner
# ---------------------------------------------------------------------------

def run_task(
    task_id: str,
    llm_client: OpenAI,
    model: str,
    env_url: str,
) -> tuple[float, list[dict]]:
    """Run one full episode for a given task using the LLM agent.

    Connects to the environment server at ``env_url`` via the
    ``TrafficEnv`` HTTP/WebSocket client, resets the environment,
    and loops through the episode calling the LLM for each decision.
    Collects episode history and computes the final grader score.

    Args:
        task_id: Task difficulty (``"easy"``, ``"medium"``, ``"hard"``).
        llm_client: Configured ``OpenAI`` client instance.
        model: Model name string to use for completions.
        env_url: Base URL of the environment server.

    Returns:
        Tuple of ``(score, episode_history)`` where score is in [0.0, 1.0].
    """
    task_config = ALL_TASKS[task_id]
    max_steps = task_config["max_steps"]

    logger.info("Starting task '%s' (%d steps) at %s", task_id, max_steps, env_url)

    # Connect to the environment server via HTTP client
    env = TrafficEnv(base_url=env_url, task_id=task_id)

    episode_history: list[dict] = []
    conversation: list[dict[str, str]] = [
        {"role": "system", "content": SYSTEM_PROMPT},
    ]

    with env.sync() as client_env:
        # Reset the environment and get initial observation
        result = client_env.reset()
        obs = result.observation

        for step in range(1, max_steps + 1):
            # Build the prompt from the current observation
            user_prompt = build_user_prompt(obs, step)
            conversation.append({"role": "user", "content": user_prompt})

            # Call the LLM
            try:
                response = llm_client.chat.completions.create(
                    model=model,
                    messages=conversation,
                    temperature=0.1,
                    max_tokens=100,
                )
                llm_text = response.choices[0].message.content or ""
            except Exception as e:
                logger.warning("LLM call failed at step %d: %s", step, e)
                llm_text = ""

            # Parse the action
            action = parse_action(llm_text)

            # Add assistant response to conversation for context continuity
            conversation.append({"role": "assistant", "content": llm_text})

            # Keep conversation history manageable (last 10 exchanges + system)
            if len(conversation) > 21:
                conversation = [conversation[0]] + conversation[-20:]

            # Execute action in the environment via HTTP client
            result = client_env.step(action)
            obs = result.observation

            # Record step state for grading
            step_state = {
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
            }
            episode_history.append(step_state)

            if step % 10 == 0:
                logger.info(
                    "  Step %d/%d — cleared=%d, wait=%d, reward=%.4f, action=%s",
                    step, max_steps,
                    obs.total_vehicles_cleared, obs.total_wait_time,
                    obs.reward, action.action.value,
                )

            if result.done:
                break

    # Grade the episode
    score = grade_episode(task_id, episode_history)
    logger.info(
        "Task '%s' complete — cleared=%d, wait=%d, score=%.4f",
        task_id, obs.total_vehicles_cleared, obs.total_wait_time, score,
    )
    return score, episode_history


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    """Entry point — run the baseline agent against all 3 tasks."""
    start_time = time.time()

    print("=" * 60)
    print("  Traffic Control Environment — Baseline Agent")
    print("=" * 60)

    # --- Load and validate environment variables ---
    api_base_url = os.environ.get("API_BASE_URL")
    model_name = os.environ.get("MODEL_NAME")
    api_key = os.environ.get("OPENAI_API_KEY") or os.environ.get("HF_TOKEN")

    missing = []
    if not api_base_url:
        missing.append("API_BASE_URL")
    if not model_name:
        missing.append("MODEL_NAME")
    if not api_key:
        missing.append("OPENAI_API_KEY (or HF_TOKEN)")

    if missing:
        logger.error(
            "Missing required environment variables: %s",
            ", ".join(missing),
        )
        print(f"\nError: Set these environment variables: {', '.join(missing)}")
        print("Example:")
        print('  export API_BASE_URL="https://api.openai.com/v1"')
        print('  export MODEL_NAME="gpt-4o-mini"')
        print('  export OPENAI_API_KEY="sk-..."')
        sys.exit(1)

    print(f"\nAPI Base:  {api_base_url}")
    print(f"Model:     {model_name}")
    print(f"API Key:   {api_key[:8]}...{api_key[-4:]}")
    print()

    # --- Create OpenAI client ---
    client = OpenAI(
        base_url=api_base_url,
        api_key=api_key,
    )

    # --- Run all 3 tasks ---
    task_ids = ["easy", "medium", "hard"]
    scores: dict[str, float] = {}
    all_success = True

    for task_id in task_ids:
        print(f"\n{'─' * 50}")
        print(f"  Task: {task_id.upper()}")
        print(f"{'─' * 50}")

        try:
            env_url = os.environ.get("HF_SPACE_URL", "http://localhost:8000")
            score, history = run_task(
                task_id=task_id,
                llm_client=client,
                model=model_name,
                env_url=env_url,
            )
            scores[task_id] = score
            threshold = ALL_TASKS[task_id]["success_threshold"]
            passed = score >= threshold
            status = "✓ PASS" if passed else "✗ FAIL"
            print(f"  Score: {score:.4f}  (threshold: {threshold})  [{status}]")

            if not passed:
                all_success = False

        except Exception as e:
            logger.exception("Task '%s' failed with error", task_id)
            scores[task_id] = 0.0
            print(f"  ERROR: {e}")
            all_success = False

    # --- Summary ---
    elapsed = time.time() - start_time
    overall = sum(scores.values()) / len(scores) if scores else 0.0

    print(f"\n{'=' * 60}")
    print(f"  RESULTS SUMMARY")
    print(f"{'=' * 60}")
    for task_id in task_ids:
        s = scores.get(task_id, 0.0)
        print(f"  {task_id:8s}: {s:.4f}")
    print(f"  {'─' * 20}")
    print(f"  Overall:  {overall:.4f}")
    print(f"  Time:     {elapsed:.1f}s")
    print(f"{'=' * 60}")

    sys.exit(0 if all_success else 1)


if __name__ == "__main__":
    main()
