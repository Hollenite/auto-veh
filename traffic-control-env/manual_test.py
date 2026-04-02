import sys
import random
sys.path.insert(0, ".")

from client.client import TrafficEnv
from models import TrafficAction, SignalAction

base = "http://localhost:8000"
actions = [SignalAction.KEEP_CURRENT, SignalAction.SWITCH_PHASE]

print("Connecting to environment...")
env = TrafficEnv(base_url=base, task_id="easy")

with env.sync() as e:
    result = e.reset()
    obs = result.observation
    print(f"Episode started | phase={obs.current_phase}")
    print("-" * 65)

    step = 0
    total_reward = 0.0

    while not result.done:
        action = TrafficAction(
            action=random.choice(actions),
            emergency_direction=None
        )
        result = e.step(action)
        obs = result.observation
        total_reward += result.reward
        step += 1

        if step % 10 == 0 or result.done:
            print(
                f"Step {step:3d} | phase={obs.current_phase:10s} | "
                f"queues=[N:{obs.queue_north} S:{obs.queue_south} "
                f"E:{obs.queue_east} W:{obs.queue_west}] | "
                f"cleared={obs.total_vehicles_cleared:3d} | "
                f"reward={result.reward:+.3f} | "
                f"emerg={obs.emergency_present}"
            )

    print("-" * 65)
    print(f"Episode done at step {step}")
    print(f"Total reward:     {total_reward:.3f}")
    print(f"Vehicles cleared: {obs.total_vehicles_cleared}")
    print(f"Final message:    {obs.message}")