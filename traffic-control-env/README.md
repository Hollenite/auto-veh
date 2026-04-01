# Traffic Control Environment 🚦

> An OpenEnv-compliant environment for autonomous traffic signal control at a 4-way intersection.

## Overview

An AI agent controls traffic signal phases at a simulated 4-way intersection to:
- **Maximize vehicle throughput** across all approaches
- **Minimize average wait times** for queued vehicles
- **Prioritize emergency vehicles** (ambulances, fire trucks, police)

## Tasks

| Task | Difficulty | Description |
|------|-----------|-------------|
| `easy` | ⭐ | Balanced traffic, no emergencies |
| `medium` | ⭐⭐ | Rush hour with asymmetric load + emergency vehicles |
| `hard` | ⭐⭐⭐ | Traffic surge with multiple simultaneous emergencies |

## Quick Start

### Run with Docker
```bash
docker build -t traffic-control-env .
docker run -p 8000:8000 traffic-control-env
```

### Run Locally
```bash
pip install -r requirements.txt
uvicorn server.app:app --host 0.0.0.0 --port 8000
```

### Run Inference
```bash
export API_BASE_URL="https://api.openai.com/v1"
export MODEL_NAME="gpt-4o-mini"
export OPENAI_API_KEY="your-key-here"
python inference.py
```

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/reset` | Reset the environment, returns initial observation |
| `POST` | `/step` | Submit a `TrafficAction`, returns next observation |
| `GET` | `/state` | Retrieve current environment state |

## Action Space

| Action | Description |
|--------|-------------|
| `KEEP_CURRENT` | Maintain the current signal phase |
| `SWITCH_PHASE` | Switch to the next phase in rotation |
| `EMERGENCY_OVERRIDE` | Force green for the emergency vehicle direction |

## Observation Space

Each observation includes:
- **Queue lengths**: Vehicles waiting at NORTH, SOUTH, EAST, WEST (0–20)
- **Signal state**: Current phase and duration
- **Emergency info**: Presence, direction, and urgency level
- **Performance metrics**: Cumulative vehicles cleared and wait time
- **Episode info**: Reward, done flag, success flag

## Reward Function

```
reward = throughput_reward + wait_penalty + emergency_handling + switch_penalty
```

- `throughput_reward`: +0.1 per vehicle cleared
- `wait_penalty`: -0.01 per waiting vehicle per step
- `emergency_handling`: +1.0 for correct priority, escalating penalty for delays
- `switch_penalty`: -0.3 for switching phases too rapidly

## Project Structure

```
traffic-control-env/
├── server/
│   ├── environment.py    # OpenEnv Environment subclass
│   ├── simulation.py     # Intersection simulation engine
│   ├── tasks.py          # Task configurations (easy/medium/hard)
│   ├── graders.py        # Episode scoring functions
│   └── app.py            # FastAPI entry point
├── client/
│   └── client.py         # EnvClient subclass
├── models.py             # Pydantic models (Action, Observation, State)
├── inference.py          # Baseline LLM agent
├── openenv.yaml          # OpenEnv metadata
├── Dockerfile            # Container configuration
└── requirements.txt      # Python dependencies
```

## Environment Variables

| Variable | Description |
|----------|-------------|
| `API_BASE_URL` | LLM endpoint URL |
| `MODEL_NAME` | Model identifier |
| `HF_TOKEN` | Hugging Face token |
| `OPENAI_API_KEY` | API key for inference |

## License

MIT
