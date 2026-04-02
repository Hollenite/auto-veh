import random
from collections import deque
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

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


class ReplayBuffer:
    def __init__(self, capacity=50000):
        self.buffer = deque(maxlen=capacity)

    def add(self, s, a, r, ns, d):
        self.buffer.append((s, a, r, ns, d))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        s, a, r, ns, d = zip(*batch)
        return (
            np.array(s, dtype=np.float32),
            np.array(a, dtype=np.int64),
            np.array(r, dtype=np.float32),
            np.array(ns, dtype=np.float32),
            np.array(d, dtype=np.float32),
        )

    def __len__(self):
        return len(self.buffer)


def choose_action(qnet, state, epsilon, device):
    if random.random() < epsilon:
        return random.randrange(len(ACTIONS))
    with torch.no_grad():
        state_t = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
        qvals = qnet(state_t)
        return int(torch.argmax(qvals, dim=1).item())


def build_action(obs, a_idx):
    action_enum = ACTIONS[a_idx]

    emergency_dir = None
    if action_enum == SignalAction.EMERGENCY_OVERRIDE and obs.emergency_present:
        emergency_dir = obs.emergency_direction

    return TrafficAction(action=action_enum, emergency_direction=emergency_dir)


def save_checkpoint(qnet, episode, path_prefix="traffic_dqn_easy"):
    path = f"{path_prefix}_ep{episode}.pt"
    torch.save(qnet.state_dict(), path)
    print(f"checkpoint saved to {path}")


def train():
    base_url = "http://localhost:8000"
    task_id = "easy"
    max_steps = 50
    num_episodes = 2000

    gamma = 0.99
    batch_size = 64
    learning_rate = 1e-3
    target_update_steps = 200

    epsilon = 1.0
    epsilon_min = 0.05
    epsilon_decay = 0.995

    checkpoint_every = 100
    final_model_path = "traffic_dqn_easy.pt"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"using device: {device}")

    env = TrafficEnv(base_url=base_url, task_id=task_id)

    recent_rewards = deque(maxlen=20)
    global_step = 0

    try:
        with env.sync() as e:
            init_result = e.reset()
            init_obs = init_result.observation
            state_dim = len(obs_to_vec(init_obs, max_steps=max_steps))

            qnet = QNet(state_dim, len(ACTIONS)).to(device)
            target = QNet(state_dim, len(ACTIONS)).to(device)
            target.load_state_dict(qnet.state_dict())
            target.eval()

            optimizer = optim.Adam(qnet.parameters(), lr=learning_rate)
            replay = ReplayBuffer()

            for episode in range(num_episodes):
                result = e.reset()
                obs = result.observation
                state = obs_to_vec(obs, max_steps=max_steps)

                total_reward = 0.0
                last_loss = None

                while not result.done:
                    a_idx = choose_action(qnet, state, epsilon, device)
                    action = build_action(obs, a_idx)

                    result = e.step(action)
                    next_obs = result.observation
                    next_state = obs_to_vec(next_obs, max_steps=max_steps)

                    replay.add(state, a_idx, result.reward, next_state, result.done)

                    state = next_state
                    obs = next_obs
                    total_reward += result.reward
                    global_step += 1

                    if len(replay) >= batch_size:
                        s, a, r, ns, d = replay.sample(batch_size)

                        s = torch.tensor(s, dtype=torch.float32, device=device)
                        a = torch.tensor(a, dtype=torch.int64, device=device)
                        r = torch.tensor(r, dtype=torch.float32, device=device)
                        ns = torch.tensor(ns, dtype=torch.float32, device=device)
                        d = torch.tensor(d, dtype=torch.float32, device=device)

                        qvals = qnet(s).gather(1, a.unsqueeze(1)).squeeze(1)
                        with torch.no_grad():
                            next_q = target(ns).max(dim=1).values
                            target_q = r + gamma * next_q * (1 - d)

                        loss = nn.MSELoss()(qvals, target_q)

                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()

                        last_loss = float(loss.item())

                    if global_step % target_update_steps == 0:
                        target.load_state_dict(qnet.state_dict())

                epsilon = max(epsilon_min, epsilon * epsilon_decay)
                recent_rewards.append(total_reward)
                avg20 = sum(recent_rewards) / len(recent_rewards)

                if episode % 10 == 0:
                    loss_str = f"{last_loss:.4f}" if last_loss is not None else "n/a"
                    print(
                        f"episode={episode} "
                        f"reward={total_reward:.2f} "
                        f"avg20={avg20:.2f} "
                        f"epsilon={epsilon:.3f} "
                        f"buffer={len(replay)} "
                        f"loss={loss_str}"
                    )

                if episode > 0 and episode % checkpoint_every == 0:
                    save_checkpoint(qnet, episode)

            torch.save(qnet.state_dict(), final_model_path)
            print(f"saved final model to {final_model_path}")

    except KeyboardInterrupt:
        print("\ntraining interrupted by user")
        try:
            if "qnet" in locals():
                interrupted_path = "traffic_dqn_easy_interrupted.pt"
                torch.save(qnet.state_dict(), interrupted_path)
                print(f"saved interrupted model to {interrupted_path}")
        except Exception as save_error:
            print(f"failed to save interrupted checkpoint: {save_error}")


if __name__ == "__main__":
    train()