import sys
import os

filepath = "tests/test_environment.py"
with open(filepath, "r", encoding="utf-8") as f:
    code = f.read()

# Change 1: Imports
old_imports = """from models import (
    SignalAction,
    TrafficAction,
    TrafficObservation,
    TrafficState,
    VALID_PHASES,
)"""
new_imports = """from models import (
    SignalAction,
    SignalCommand,
    TrafficAction,
    TrafficObservation,
    TrafficState,
    VALID_PHASES,
    VehicleType,
    VehicleRecord,
)"""
code = code.replace(old_imports, new_imports)

# Global replaces
code = code.replace("SignalAction.KEEP_CURRENT", "SignalCommand.HOLD_CURRENT_PHASE")

# Change 3: test_switch_phase_changes_signal and other SWITCH_PHASE
code = code.replace("action=SignalAction.SWITCH_PHASE", "action=SignalCommand.SET_EW_GREEN")

# Except we need to fix test_full_phase_rotation
test_full_old = """    def test_full_phase_rotation(self, easy_env: TrafficEnvironment):
        \"\"\"Cycling through all 4 phases in the rotation should work.\"\"\"
        keep = TrafficAction(action=SignalCommand.HOLD_CURRENT_PHASE)
        switch = TrafficAction(action=SignalCommand.SET_EW_GREEN)

        expected_phases = ["NS_GREEN", "ALL_RED", "EW_GREEN", "ALL_RED"]
        assert easy_env.state.current_phase == expected_phases[0]

        for i in range(1, 4):
            # Wait for MIN_PHASE_DURATION
            easy_env.step(keep)
            easy_env.step(keep)
            # Switch
            obs = easy_env.step(switch)
            assert obs.current_phase == expected_phases[i], (
                f"Step {i}: expected {expected_phases[i]}, got {obs.current_phase}"
            )"""
test_full_new = """    def test_full_phase_rotation(self, easy_env: TrafficEnvironment):
        \"\"\"Cycling through all 4 phases in the rotation should work.\"\"\"
        keep = TrafficAction(action=SignalCommand.HOLD_CURRENT_PHASE)
        
        expected_phases = ["NS_GREEN", "ALL_RED", "EW_GREEN", "ALL_RED"]
        assert easy_env.state.current_phase == expected_phases[0]
        
        actions_to_switch = [SignalCommand.SET_EW_GREEN, SignalCommand.SET_EW_GREEN, SignalCommand.SET_NS_GREEN]

        for i in range(1, 4):
            # Wait for MIN_PHASE_DURATION
            easy_env.step(keep)
            easy_env.step(keep)
            # Switch
            switch = TrafficAction(action=actions_to_switch[i-1])
            obs = easy_env.step(switch)
            assert obs.current_phase == expected_phases[i], (
                f"Step {i}: expected {expected_phases[i]}, got {obs.current_phase}"
            )"""
code = code.replace(test_full_old, test_full_new)

# Change 4: EMERGENCY_OVERRIDE
code = code.replace("SignalAction.EMERGENCY_OVERRIDE", "SignalCommand.SET_NS_GREEN")

# Change 5: Insert test_observation_has_avg_wait_fields
obs_test = """    def test_observation_has_avg_wait_fields(self):
        \"\"\"TrafficObservation should include avg_wait fields after a step.\"\"\"
        env = TrafficEnvironment("easy")
        env.reset()
        action = TrafficAction(action=SignalCommand.HOLD_CURRENT_PHASE)
        obs = env.step(action).observation if hasattr(env.step(action), 'observation') else env.step(action)
        # Check via direct environment step
        env2 = TrafficEnvironment("easy")
        env2.reset()
        result_obs = env2.step(TrafficAction(action=SignalCommand.HOLD_CURRENT_PHASE))
        assert hasattr(result_obs, 'avg_wait_north')
        assert hasattr(result_obs, 'avg_wait_south')
        assert hasattr(result_obs, 'avg_wait_east')
        assert hasattr(result_obs, 'avg_wait_west')
        assert hasattr(result_obs, 'steps_remaining')
        assert result_obs.steps_remaining == 49  # easy task: 50 - 1 step

    def test_traffic_state_roundtrip(self):"""
code = code.replace("    def test_traffic_state_roundtrip(self):", obs_test)

# Change 6: Add Inference tests
inf_tests = """
# =========================================================================
# 9. Test Inference Fallback
# =========================================================================

class TestInference:
    def test_heuristic_policy_returns_valid_action(self):
        from inference import heuristic_policy
        from server.environment import TrafficEnvironment
        env = TrafficEnvironment("easy")
        obs = env.reset()
        action = heuristic_policy(obs)
        assert isinstance(action, TrafficAction)
        assert action.action in list(SignalCommand)
    
    def test_heuristic_policy_prioritizes_emergency(self):
        from inference import heuristic_policy
        from server.environment import TrafficEnvironment
        env = TrafficEnvironment("medium")
        env.reset()
        # Manually inject an emergency
        env.sim.emergency = {"direction": "NORTH", "urgency": "CRITICAL", "steps_waiting": 5}
        obs = env._build_observation(env.sim._build_state_dict(), done=False)
        action = heuristic_policy(obs)
        assert action.action == SignalCommand.SET_NS_GREEN
"""
code += inf_tests

with open(filepath, "w", encoding="utf-8") as f:
    f.write(code)

print("Tests updated.")
