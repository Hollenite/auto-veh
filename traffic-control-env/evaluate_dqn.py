import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from client.client import TrafficEnv
from models import TrafficAction, SignalAction

ACTIONS = [
    SignalAction.KEEP_CURRENT,
    SignalAction.SWITCH_PHASE,
    SignalAction.EMERGENCY_OVERRIDE,
]


def obs_to_vec(obs, max_steps=50):
    phase_map = {
        "NS_GREEN": [1, 0, 0],
        "ALL_RED": [0, 1, 0],
        "EW_GREEN": [0, 0, 1],
    }
    dir_map = {
        None: [0, 0, 0, 0],
        "NORTH": [1, 0, 0, 0],
        "SOUTH": [0, 1, 0, 0],
        "EAST": [0, 0, 1, 0],
        "WEST": [0, 0, 0, 1],
    }
    urg_map = {
        None: [0, 0, 0],
        "LOW": [1, 0, 0],
        "HIGH": [0, 1, 0],
        "CRITICAL": [0, 0, 1],
    }

    vec = [
        obs.queue_north / 20.0,
        obs.queue_south / 20.0,
        obs.queue_east / 20.0,
        obs.queue_west / 20.0,
        *phase_map.get(obs.current_phase, [0, 0, 0]),
        min(obs.phase_duration, 10) / 10.0,
        1.0 if obs.emergency_present else 0.0,
        *dir_map.get(obs.emergency_direction, [0, 0, 0, 0]),
        *urg_map.get(obs.emergency_urgency, [0, 0, 0]),
        obs.current_step / float(max_steps),
    ]
    return np.array(vec, dtype=np.float32)


class QNet(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim),
        )

    def forward(self, x):
        return self.net(x)


def build_action(obs, a_idx):
    action_enum = ACTIONS[a_idx]

    emergency_dir = None
    if action_enum == SignalAction.EMERGENCY_OVERRIDE and obs.emergency_present:
        emergency_dir = obs.emergency_direction

    return TrafficAction(action=action_enum, emergency_direction=emergency_dir)


def choose_action_greedy(qnet, state, device):
    with torch.no_grad():
        state_t = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
        qvals = qnet(state_t)
        return int(torch.argmax(qvals, dim=1).item())


def infer_max_steps(task_id):
    return 60 if task_id == "hard" else 50


def default_model_path(task_id):
    candidates = [
        Path("checkpoints") / task_id / f"traffic_dqn_{task_id}_final.pt",
        Path(f"traffic_dqn_{task_id}.pt"),
        Path(f"traffic_dqn_{task_id}_final.pt"),
    ]
    for path in candidates:
        if path.exists():
            return path
    return candidates[0]


def run_evaluation(
    task_id: str,
    model_path: Path,
    episodes: int,
    base_url: str,
    verbose_every: int,
):
    max_steps = infer_max_steps(task_id)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"using device: {device}")
    print(f"task:        {task_id}")
    print(f"episodes:    {episodes}")
    print(f"base_url:    {base_url}")
    print(f"model_path:  {model_path}")

    env = TrafficEnv(base_url=base_url, task_id=task_id)

    with env.sync() as e:
        init_result = e.reset()
        init_obs = init_result.observation
        state_dim = len(obs_to_vec(init_obs, max_steps=max_steps))

        qnet = QNet(state_dim, len(ACTIONS)).to(device)
        state_dict = torch.load(model_path, map_location=device)
        qnet.load_state_dict(state_dict)
        qnet.eval()

        rewards = []
        cleared_counts = []
        scores = []
        steps_taken = []
        emergency_counts = []
        emergency_wait_totals = []

        for episode in range(episodes):
            result = e.reset()
            obs = result.observation
            state = obs_to_vec(obs, max_steps=max_steps)

            total_reward = 0.0
            step_count = 0
            emergency_seen_steps = 0
            emergency_wait_sum = 0

            while not result.done:
                a_idx = choose_action_greedy(qnet, state, device)
                action = build_action(obs, a_idx)

                result = e.step(action)
                obs = result.observation
                state = obs_to_vec(obs, max_steps=max_steps)

                total_reward += result.reward
                step_count += 1

                if getattr(obs, "emergency_present", False):
                    emergency_seen_steps += 1
                    emergency_wait_sum += 1

            final_obs = result.observation
            final_message = getattr(final_obs, "message", "")

            score = None
            if isinstance(final_message, str) and "Score:" in final_message:
                try:
                    score_str = final_message.split("Score:")[-1].strip()
                    score = float(score_str)
                except ValueError:
                    score = None

            rewards.append(total_reward)
            cleared_counts.append(getattr(final_obs, "total_vehicles_cleared", 0))
            steps_taken.append(step_count)
            emergency_counts.append(emergency_seen_steps)
            emergency_wait_totals.append(emergency_wait_sum)
            if score is not None:
                scores.append(score)

            if verbose_every > 0 and (episode % verbose_every == 0):
                score_display = f"{score:.4f}" if score is not None else "n/a"
                print(
                    f"episode={episode} "
                    f"reward={total_reward:.2f} "
                    f"cleared={getattr(final_obs, 'total_vehicles_cleared', 0)} "
                    f"steps={step_count} "
                    f"score={score_display} "
                    f"message={final_message}"
                )

    print("\n===== EVALUATION SUMMARY =====")
    print(f"task:                 {task_id}")
    print(f"episodes:             {episodes}")
    print(f"avg_reward:           {np.mean(rewards):.2f}")
    print(f"std_reward:           {np.std(rewards):.2f}")
    print(f"min_reward:           {np.min(rewards):.2f}")
    print(f"max_reward:           {np.max(rewards):.2f}")
    print(f"avg_cleared:          {np.mean(cleared_counts):.2f}")
    print(f"avg_steps:            {np.mean(steps_taken):.2f}")
    print(f"avg_emerg_steps:      {np.mean(emergency_counts):.2f}")
    print(f"avg_emerg_wait_sum:   {np.mean(emergency_wait_totals):.2f}")
    if scores:
        print(f"avg_score:            {np.mean(scores):.4f}")
        print(f"std_score:            {np.std(scores):.4f}")
        print(f"min_score:            {np.min(scores):.4f}")
        print(f"max_score:            {np.max(scores):.4f}")
    else:
        print("avg_score:            n/a (could not parse score from message)")


def main():
    parser = argparse.ArgumentParser(description="Evaluate a trained DQN traffic controller.")
    parser.add_argument(
        "--task",
        type=str,
        default="easy",
        choices=["easy", "medium", "hard"],
        help="Task difficulty to evaluate on.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Path to .pt model checkpoint. If omitted, a default path is guessed.",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=100,
        help="Number of evaluation episodes.",
    )
    parser.add_argument(
        "--base-url",
        type=str,
        default="http://localhost:8000",
        help="Environment server base URL.",
    )
    parser.add_argument(
        "--verbose-every",
        type=int,
        default=10,
        help="Print one episode summary every N episodes. Use 0 to disable.",
    )
    args = parser.parse_args()

    model_path = Path(args.model) if args.model else default_model_path(args.task)
    if not model_path.exists():
        raise FileNotFoundError(
            f"Model file not found: {model_path}\n"
            f"Pass --model explicitly, for example:\n"
            f"python evaluate_dqn.py --task {args.task} --model traffic_dqn_{args.task}.pt"
        )

    run_evaluation(
        task_id=args.task,
        model_path=model_path,
        episodes=args.episodes,
        base_url=args.base_url,
        verbose_every=args.verbose_every,
    )


if __name__ == "__main__":
    main()