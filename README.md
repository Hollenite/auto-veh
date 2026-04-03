---
title: Traffic Control Environment
emoji: 🚦
colorFrom: blue
colorTo: green
sdk: docker
app_port: 7860
pinned: false
---

# Autonomous Traffic Control Environment

## Overview

The Autonomous Traffic Control Environment is an OpenEnv-compliant simulation where an AI agent governs the traffic signal phases at a busy 4-way intersection. The environment accurately models vehicle queues, stochastic arrivals, traffic discharge rates, and unexpected emergency vehicle prioritization. By formulating this as a decision-making environment, it provides the Reinforcement Learning (RL) and AI agent community with a challenging, real-world continuous monitoring task requiring balancing immediate throughput against delayed penalties and out-of-distribution events (emergencies).

## Problem Statement

To minimize wait times and maximize total intersection throughput, while always prioritizing emergency vehicles that need to cross the intersection without delay. Managing traffic lights relies on looking ahead at expanding queue sizes, preventing intersection gridlocks, avoiding signal thrashing, and immediately breaking patterns to prioritize critical vehicles like ambulances or fire engines.

## Action Space

The action space controls the signal phase of the intersection.

| Action               | Meaning                                                                                                                                        |
| -------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------- |
| `KEEP_CURRENT`       | Maintain the current signal phase. Ideal when the current green direction still has substantial traffic to clear.                              |
| `SWITCH_PHASE`       | Rotate to the next signal phase in the standard cycle (`NS_GREEN` → `ALL_RED` → `EW_GREEN` → `ALL_RED`).                                       |
| `EMERGENCY_OVERRIDE` | Immediately force the signal green for the direction containing an active emergency vehicle, preempting standard cycles and minimum durations. |

_Note: Every `TrafficAction` also optionally accepts an `emergency_direction` parameter to target overrides explicitly._

## Observation Space

At each step, the environment provides the agent with a `TrafficObservation` containing the state of the intersection:

| Field                    | Type          | Description                                                       |
| ------------------------ | ------------- | ----------------------------------------------------------------- |
| `queue_north`            | `int`         | Current vehicle queue size at the North approach (0-20).          |
| `queue_south`            | `int`         | Current vehicle queue size at the South approach (0-20).          |
| `queue_east`             | `int`         | Current vehicle queue size at the East approach (0-20).           |
| `queue_west`             | `int`         | Current vehicle queue size at the West approach (0-20).           |
| `current_phase`          | `str`         | Active signal phase (`"NS_GREEN"`, `"EW_GREEN"`, or `"ALL_RED"`). |
| `phase_duration`         | `int`         | Number of consecutive steps the current phase has been active.    |
| `emergency_present`      | `bool`        | True if an emergency vehicle is currently waiting.                |
| `emergency_direction`    | `str \| None` | Direction the emergency vehicle is originating from.              |
| `emergency_urgency`      | `str \| None` | Urgency of the emergency (`"LOW"`, `"HIGH"`, `"CRITICAL"`).       |
| `total_vehicles_cleared` | `int`         | Total number of vehicles discharged so far this episode.          |
| `total_wait_time`        | `int`         | Cumulative waiting time accumulated by all queuing vehicles.      |
| `current_step`           | `int`         | Current episode tick (1 step = ~10 real-world seconds).           |
| `reward`                 | `float`       | Scalar reward achieved in the immediate previous step.            |

## Reward Function

The environment utilizes a dense, informative, per-step reward function rather than a sparse binary success variable. The formula encompasses four core components:

$$Reward = R_{throughput} + R_{wait} + R_{emergency} + R_{switch}$$

- **Throughput**: $+0.1$ for each vehicle successfully discharged in the step.
- **Wait Penalty**: $-0.01 \times (\text{Total queued vehicles})$ per step to penalize large, stagnant queues.
- **Emergency**: $+1.0$ bonus when an emergency vehicle is granted green and cleared; $-0.2 \times \text{wait\_steps}$ penalty per step it is stuck waiting, and a severe $-5.0$ penalty if the timeout limit is breached.
- **Switching**: $-0.3$ penalty for attempting to switch signal phases before `MIN_PHASE_DURATION` (2 ticks) has elapsed, simulating real-world safety transitions and preventing thrashing.

This function is highly informative because it continuously nudges the agent to balance discharging traffic efficiently against letting conflicting queues grow too long, while treating emergencies as absolute mandates.

## Tasks

The environment supports three varied difficulty configurations:

### 1. Easy

- **Description:** A balanced, 4-way intersection with steady predictable traffic. No emergencies occur.
- **Rationale:** Tests the agent's fundamental ability to cycle phases without incurring wait penalties or signal thrashing.
- **Grader:** Score based purely on cumulative wait time vs. throughput. Max score is 1.0.

### 2. Medium

- **Description:** Models a rush-hour scenario with heavy asymmetric North-South load and occasional emergency vehicles.
- **Rationale:** Requires the agent to break 50/50 phase splits to favor the dominant traffic flow while staying vigilant for emergencies.
- **Grader:** Combines Wait/Throughput ($70\%$) and Emergency Response Time ($30\%$).

### 3. Hard

- **Description:** A traffic surge with high overall volume and multiple, overlapping simultaneous emergencies.
- **Rationale:** Strains intersection capacity. Tests the agent's absolute prioritization of emergencies under extreme pressure and their ability to prevent gridlock imbalances in opposing queues.
- **Grader:** Combines Wait/Throughput ($40\%$), Emergency Response Time ($40\%$), and Queue Balance uniformity ($20\%$).

## Environment Design

The simulation uses discrete time steps representing ~10-second intervals.

1. **Intersection Geometry:** 4 approaches (N, S, E, W), discharging up to 2 vehicles per tick when given green.
2. **Phase Transitions:** Transitions strictly follow `NS_GREEN` -> `ALL_RED` -> `EW_GREEN`. `ALL_RED` phases exist to simulate safety clearance intervals.
3. **Stochasticity:** Vehicle arrivals are controlled probabilistically over Gaussian distributions based on the difficulty configuration.
4. **Emergency Lifecycle:** Emergencies spawn unpredictably, growing in urgency (LOW $\rightarrow$ HIGH $\rightarrow$ CRITICAL) the longer they wait at red lights before they eventually time out (timeout penalty).

## Setup & Installation

### Local Installation

1. Clone the repository.
2. Ensure you have Python 3.10+ installed.
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Start the server:
   ```bash
   uvicorn server.app:app --host 0.0.0.0 --port 8000
   ```

### Docker

To run inside a pristine Docker container for deployment to HuggingFace spaces:

```bash
docker build -t traffic-env .
docker run -p 8000:8000 -e ENABLE_WEB_INTERFACE=true traffic-env
```

## Usage

Leverage the `TrafficEnv` OpenEnv client to interface seamlessly with the environment using Pydantic models natively.

```python
from models import TrafficAction, SignalAction
from client.client import TrafficEnv

# Instantiate HTTP/WS Client linked to your local or HF server
env = TrafficEnv(base_url="http://localhost:8000", task_id="medium").sync()

with env:
    # 1. Reset Environment
    result = env.reset()
    obs = result.observation

    while not result.done:
        # 2. Logic inference...
        if obs.emergency_present:
            action = TrafficAction(
                action=SignalAction.EMERGENCY_OVERRIDE,
                emergency_direction=obs.emergency_direction
            )
        else:
            action = TrafficAction(action=SignalAction.KEEP_CURRENT)

        # 3. Take Step
        result = env.step(action)
        obs = result.observation

    print(f"Final Score: {env.final_score}")
```

## Baseline Scores

The baseline agent utilizes an LLM passing system prompts, observing current metrics, and interpreting hints.

| Task   | Score | Model       |
| ------ | ----- | ----------- |
| Easy   | TBD   | gpt-4o-mini |
| Medium | TBD   | gpt-4o-mini |
| Hard   | TBD   | gpt-4o-mini |

## Why This Fills a Gap

Traffic signal control stands uniquely at the intersection of long-horizon planning and immediate reactivity. Standard RL environments often focus exclusively on physical manipulation or purely logic-based board games. This environment maps directly to high-impact cyber-physical systems optimization—demanding skills such as managing continuously changing probabilistic state, understanding rigid cyclic limitations (phase rotation limits), and dynamically overriding plans for sudden critical anomalies. It constitutes a perfect, intuitive sandbox for pushing modern AI agents beyond zero-sum games and into real-time infrastructure management.
