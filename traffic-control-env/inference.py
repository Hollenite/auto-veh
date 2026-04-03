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

from models import SignalCommand, SignalAction, TrafficAction, TrafficObservation
from client.client import TrafficEnv
from server.graders import grade_episode
from server.tasks import ALL_TASKS

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

load_dotenv()

API_BASE_URL = os.getenv("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4.1-mini")
HF_TOKEN = os.getenv("HF_TOKEN")

if HF_TOKEN is None:
    raise ValueError("HF_TOKEN environment variable is required")

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
- Average wait time per direction (float, in simulation steps)
- Current signal phase: NS_GREEN, EW_GREEN, or ALL_RED
- Steps remaining in this episode
- Whether an emergency vehicle is present (and from which direction)

Your goal: maximize vehicle throughput, minimize wait times, and ALWAYS
prioritize emergency vehicles.

Respond with EXACTLY one JSON object, no other text:
{"action": "<COMMAND>", "emergency_direction": "<NORTH|SOUTH|EAST|WEST|null>"}

Available commands:
- set_ns_green:       Give green light to NORTH and SOUTH. EAST and WEST stop.
- set_ew_green:       Give green light to EAST and WEST. NORTH and SOUTH stop.
- hold_current_phase: Keep the current phase unchanged.
- set_all_red:        Stop all traffic (use only as transition, not strategy).

Strategy rules:
1. EMERGENCY RULE (highest priority): If emergency_present=true, immediately
   set the phase that gives green to emergency_direction:
   - NORTH or SOUTH emergency → use set_ns_green
   - EAST or WEST emergency → use set_ew_green
   Set emergency_direction to the direction shown in the observation.
2. BALANCE RULE: Switch to the phase serving the direction with longer queues
   and higher avg_wait. Compare (queue_north+queue_south+avg_wait_north+avg_wait_south)
   vs (queue_east+queue_west+avg_wait_east+avg_wait_west).
3. PATIENCE RULE: Don't switch if phase has been active fewer than 2 steps
   (unless emergency). Switching too fast wastes the transition cycle.
4. ENDGAME RULE: If steps_remaining < 5, prioritize the direction with the
   longest queue to maximize final throughput.
"""

# Valid action values for quick lookup
_VALID_ACTIONS: set[str] = {a.value for a in SignalCommand} | {a.value for a in SignalAction}


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
        ])
        if obs.emergency_direction in ("NORTH", "SOUTH"):
            phase_needed = "set_ns_green"
        else:
            phase_needed = "set_ew_green"
        lines.append(f"  → Use {phase_needed} with emergency_direction=\"{obs.emergency_direction}\"")
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

    lines.extend([
        f"",
        f"Average wait times (steps):",
        f"  NORTH: {obs.avg_wait_north:.1f}",
        f"  SOUTH: {obs.avg_wait_south:.1f}",
        f"  EAST:  {obs.avg_wait_east:.1f}",
        f"  WEST:  {obs.avg_wait_west:.1f}",
        f"  Steps remaining: {obs.steps_remaining}",
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
    """Parse LLM JSON response into a TrafficAction."""
    from models import SignalCommand
    fallback = TrafficAction(action=SignalCommand.HOLD_CURRENT_PHASE)
    
    if not response_text or not response_text.strip():
        logger.warning("Empty LLM response, using fallback HOLD_CURRENT_PHASE")
        return fallback
        
    text = response_text.strip()
    
    parsed = _try_parse_json(text)
    if parsed is None:
        import re
        json_match = re.search(r"\{[^{}]*\}", text, re.DOTALL)
        if json_match:
            parsed = _try_parse_json(json_match.group())
            
    if parsed is None:
        logger.warning("Failed to parse LLM response: %s", text[:200])
        return fallback

    action_str = parsed.get("action", "").lower()
    
    action_map = {
        "ns_green": "set_ns_green",
        "ns": "set_ns_green",
        "ew_green": "set_ew_green",
        "ew": "set_ew_green",
        "hold": "hold_current_phase",
        "keep": "hold_current_phase",
        "keep_current": "hold_current_phase",
        "all_red": "set_all_red",
        "red": "set_all_red",
        "set_ns_green": "set_ns_green",
        "set_ew_green": "set_ew_green",
        "hold_current_phase": "hold_current_phase",
        "set_all_red": "set_all_red",
    }
    
    emergency_dir = parsed.get("emergency_direction")
    if isinstance(emergency_dir, str):
        emergency_dir = emergency_dir.upper()
        if emergency_dir in ("NULL", "NONE", ""):
            emergency_dir = None
        elif emergency_dir not in {"NORTH", "SOUTH", "EAST", "WEST"}:
            emergency_dir = None
            
    if action_str in ("emergency_override", "emergencyoverride", "emergency", "override"):
        if emergency_dir in ("NORTH", "SOUTH"):
            action_str = "set_ns_green"
        elif emergency_dir in ("EAST", "WEST"):
            action_str = "set_ew_green"
        else:
            action_str = "hold_current_phase"
            
    normalised = action_map.get(action_str, action_str)
    
    if normalised not in _VALID_ACTIONS:
        logger.warning("Invalid action '%s', using HOLD_CURRENT_PHASE", action_str)
        return fallback
        
    return TrafficAction(
        action=SignalCommand(normalised),
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

def heuristic_policy(obs: TrafficObservation) -> TrafficAction:
    """Deterministic heuristic agent — used as fallback when no LLM is available.
    
    Strategy:
        1. Emergency vehicles get immediate priority.
        2. Otherwise serve the direction with highest combined queue + avg_wait pressure.
        3. Hold current phase if it's already optimal and recently switched.
    """
    from models import SignalCommand
    
    # Emergency priority
    if obs.emergency_present and obs.emergency_direction:
        if obs.emergency_direction in ("NORTH", "SOUTH"):
            if obs.current_phase == "NS_GREEN":
                return TrafficAction(action=SignalCommand.HOLD_CURRENT_PHASE)
            return TrafficAction(
                action=SignalCommand.SET_NS_GREEN,
                emergency_direction=obs.emergency_direction
            )
        else:
            if obs.current_phase == "EW_GREEN":
                return TrafficAction(action=SignalCommand.HOLD_CURRENT_PHASE)
            return TrafficAction(
                action=SignalCommand.SET_EW_GREEN,
                emergency_direction=obs.emergency_direction
            )
    
    # Pressure-based decision
    ns_pressure = (
        obs.queue_north + obs.queue_south
        + obs.avg_wait_north + obs.avg_wait_south
    )
    ew_pressure = (
        obs.queue_east + obs.queue_west
        + obs.avg_wait_east + obs.avg_wait_west
    )
    
    # Hold if phase was recently set (avoid thrashing)
    if obs.phase_duration < 2:
        return TrafficAction(action=SignalCommand.HOLD_CURRENT_PHASE)
    
    if ns_pressure >= ew_pressure:
        if obs.current_phase == "NS_GREEN":
            return TrafficAction(action=SignalCommand.HOLD_CURRENT_PHASE)
        return TrafficAction(action=SignalCommand.SET_NS_GREEN)
    else:
        if obs.current_phase == "EW_GREEN":
            return TrafficAction(action=SignalCommand.HOLD_CURRENT_PHASE)
        return TrafficAction(action=SignalCommand.SET_EW_GREEN)


def run_task(
    task_id: str,
    llm_client: OpenAI | None,
    model: str,
    env_url: str,
    step_log: list[dict],
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
            error_msg = None
            if llm_client is not None:
                # LLM path
                user_prompt = build_user_prompt(obs, step)
                conversation.append({"role": "user", "content": user_prompt})
                try:
                    response = llm_client.chat.completions.create(
                        model=model,
                        messages=conversation,
                        temperature=0.1,
                        max_tokens=100,
                    )
                    llm_text = response.choices[0].message.content or ""
                except Exception as e:
                    error_msg = str(e)
                    logger.warning("LLM call failed at step %d: %s", step, e)
                    llm_text = ""
                action = parse_action(llm_text)
                conversation.append({"role": "assistant", "content": llm_text})
                if len(conversation) > 21:
                    conversation = [conversation[0]] + conversation[-20:]
            else:
                # Heuristic fallback path
                action = heuristic_policy(obs)

            # Execute action in the environment via HTTP client
            try:
                result = client_env.step(action)
                obs = result.observation
                done = result.done
            except Exception as e:
                error_msg = str(e)
                done = True
                
            step_log.append({
                "step": step,
                "action": action.action.value,
                "reward": obs.reward if 'obs' in locals() else 0.0,
                "done": done,
                "error": error_msg
            })

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
                "avg_wait_north": obs.avg_wait_north,
                "avg_wait_south": obs.avg_wait_south,
                "avg_wait_east":  obs.avg_wait_east,
                "avg_wait_west":  obs.avg_wait_west,
            }
            episode_history.append(step_state)



            if result.done:
                break

    # Grade the episode
    score = grade_episode(task_id, episode_history)

    return score, episode_history


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    client = OpenAI(
        base_url=API_BASE_URL,
        api_key=HF_TOKEN,
    )

    task_ids = ["easy", "medium", "hard"]
    env_url = os.environ.get("HF_SPACE_URL", "http://localhost:8000")

    for task_id in task_ids:
        step_log = []
        score = 0.0
        success = False
        try:
            print(f"[START] task={task_id} env=traffic-control-env model={MODEL_NAME}")
            score, history = run_task(
                task_id=task_id,
                llm_client=client,
                model=MODEL_NAME,
                env_url=env_url,
                step_log=step_log
            )
            threshold = ALL_TASKS[task_id]["success_threshold"]
            success = score >= threshold
            
            for s in step_log:
                reward_str = f"{s['reward']:.2f}"
                done_str = "true" if s['done'] else "false"
                error_str = s['error'] if s['error'] is not None else "null"
                print(f"[STEP] step={s['step']} action={s['action']} reward={reward_str} done={done_str} error={error_str}")
        except BaseException as e:
            success = False
            for s in step_log:
                reward_str = f"{s['reward']:.2f}"
                done_str = "true" if s['done'] else "false"
                error_str = s['error'] if s['error'] is not None else "null"
                print(f"[STEP] step={s['step']} action={s['action']} reward={reward_str} done={done_str} error={error_str}")
        finally:
            success_str = "true" if success else "false"
            steps = len(step_log)
            rewards_str = ",".join(f"{s['reward']:.2f}" for s in step_log)
            print(f"[END] success={success_str} steps={steps} rewards={rewards_str}")

if __name__ == "__main__":
    main()
