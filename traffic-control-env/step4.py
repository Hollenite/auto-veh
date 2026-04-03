from server.graders import grade_episode
history = [{'total_vehicles_cleared': 10, 'queues': {'NORTH':1,'SOUTH':1,'EAST':1,'WEST':1}, 'total_wait_time': 20, 'emergency_present': False, 'current_phase': 'NS_GREEN', 'avg_wait': {'NORTH': 1.0, 'SOUTH': 1.0, 'EAST': 2.0, 'WEST': 2.0}} for _ in range(50)]
score = grade_episode('hard', history)
assert 0.0 <= score <= 1.0, f'Score out of range: {score}'
print('graders OK -- hard score:', score)
