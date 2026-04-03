from inference import heuristic_policy
from server.environment import TrafficEnvironment
env = TrafficEnvironment('easy')
obs = env.reset()
action = heuristic_policy(obs)
print('heuristic OK -- action:', action.action.value)
