from server.environment import TrafficEnvironment
from models import TrafficAction, SignalCommand
env = TrafficEnvironment('easy')
obs = env.reset()
assert hasattr(obs, 'avg_wait_north'), 'Missing avg_wait_north'
assert hasattr(obs, 'steps_remaining'), 'Missing steps_remaining'
assert obs.steps_remaining == 50
result = env.step(TrafficAction(action=SignalCommand.HOLD_CURRENT_PHASE))
assert result.steps_remaining == 49
print('env OK -- steps_remaining:', result.steps_remaining)
