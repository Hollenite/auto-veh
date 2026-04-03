import sys
import re

with open("d:/PROJECTS/auto-veh/traffic-control-env/server/simulation.py", "r", encoding="utf-8") as f:
    content = f.read()

# Change 1
c1_target = """from __future__ import annotations

import numpy as np"""
c1_rep = """from __future__ import annotations

import numpy as np
from models import VehicleRecord, VehicleType, Direction"""
if c1_target in content: content = content.replace(c1_target, c1_rep, 1)

# Change 2a
c2a_target = "self.queues: dict[str, int] = {}"
c2a_rep = "self.queues: dict[str, list] = {}"
if c2a_target in content: content = content.replace(c2a_target, c2a_rep, 1)

# Change 2b
c2b_target = "self.queues = {d: 0 for d in ALL_DIRECTIONS}"
c2b_rep = "self.queues = {d: [] for d in ALL_DIRECTIONS}"
if c2b_target in content: content = content.replace(c2b_target, c2b_rep, 1)

# Change 3 & 10
c3_target = """        return {
            "queues": dict(self.queues),"""
c3_rep = """        queue_counts = {d: len(q) for d, q in self.queues.items()}
        avg_waits = {}
        for d, q in self.queues.items():
            avg_waits[d] = (sum(v.wait_time for v in q) / len(q)) if q else 0.0

        return {
            "queues": queue_counts,
            "avg_wait": avg_waits,"""
if c3_target in content: content = content.replace(c3_target, c3_rep, 1)

# Change 7 step
c7_target = """        # 2. Discharge vehicles on green approaches
        vehicles_cleared = self._discharge_vehicles()

        # 3. Stochastic vehicle arrivals
        self._arrive_vehicles()

        # 4. Emergency vehicle lifecycle
        self._update_emergency()

        # 5. Reward computation
        reward = self._calculate_reward(vehicles_cleared, action, emergency_handled)

        # 6. Bookkeeping
        self.phase_duration += 1
        self.total_wait_time += sum(self.queues.values())"""
c7_rep = """        # 2. Discharge vehicles on green approaches
        cleared_counts = self._discharge_vehicles()

        # 3. Stochastic vehicle arrivals
        self._arrive_vehicles()
        self._increment_wait_times()

        # 4. Emergency vehicle lifecycle
        self._update_emergency()

        # 5. Reward computation
        reward = self._calculate_reward(cleared_counts, action, emergency_handled)

        # 6. Bookkeeping
        self.phase_duration += 1
        self.total_wait_time += sum(len(q) for q in self.queues.values())"""
if c7_target in content: content = content.replace(c7_target, c7_rep, 1)

# Change 8 process action
c8_pat = re.compile(r'    def _process_action\(self, action: str\) -> None:\n.*?    def _advance_phase\(self\) -> None:', re.DOTALL)
c8_rep = """    def _process_action(self, action: str) -> None:
        \"\"\"Update signal phase based on agent action.
        
        Actions:
            set_ns_green:       Force NS_GREEN immediately.
            set_ew_green:       Force EW_GREEN immediately.
            set_all_red:        Force ALL_RED immediately.
            hold_current_phase: No change.
        
        MIN_PHASE_DURATION is enforced for set_ns_green and set_ew_green
        unless an emergency vehicle is present (emergency always overrides).
        \"\"\"
        if action == "hold_current_phase":
            return

        emergency_active = self.emergency is not None
        
        if action == "set_ns_green":
            if self.current_phase != "NS_GREEN":
                if self.phase_duration >= MIN_PHASE_DURATION or emergency_active:
                    self.current_phase = "NS_GREEN"
                    self.phase_index = PHASE_ROTATION.index("NS_GREEN")
                    self.phase_duration = 0
        
        elif action == "set_ew_green":
            if self.current_phase != "EW_GREEN":
                if self.phase_duration >= MIN_PHASE_DURATION or emergency_active:
                    self.current_phase = "EW_GREEN"
                    self.phase_index = PHASE_ROTATION.index("EW_GREEN")
                    self.phase_duration = 0
        
        elif action == "set_all_red":
            if self.current_phase != "ALL_RED":
                self.current_phase = "ALL_RED"
                self.phase_index = PHASE_ROTATION.index("ALL_RED")
                self.phase_duration = 0

        # Legacy support: old action names map to new ones
        elif action == "keep_current":
            return
        elif action == "switch_phase":
            if self.phase_duration >= MIN_PHASE_DURATION:
                self._advance_phase()
        elif action == "emergency_override":
            if self.emergency is not None:
                target = _PHASE_FOR_DIRECTION[self.emergency["direction"]]
                if self.current_phase != target:
                    self.phase_index = PHASE_ROTATION.index(target)
                    self.current_phase = target
                    self.phase_duration = 0

    def _advance_phase(self) -> None:"""
content = c8_pat.sub(c8_rep, content, count=1)

# Change 4, 5, 6
c456_pat = re.compile(r'    def _discharge_vehicles\(self\) -> int:\n.*?    def _update_emergency\(self\) -> None:', re.DOTALL)
c456_rep = """    def _discharge_vehicles(self) -> dict:
        \"\"\"Clear vehicles from approaches with green signal.
        
        Emergency vehicles are prioritized in their queue (popped first).
        Returns dict with keys "normal" and "emergency" counting cleared vehicles.
        \"\"\"
        green_dirs = _GREEN_DIRECTIONS.get(self.current_phase, [])
        cleared = {"normal": 0, "emergency": 0}

        for direction in green_dirs:
            queue = self.queues[direction]
            for _ in range(DISCHARGE_RATE_PER_DIRECTION):
                if not queue:
                    break
                # Prioritize emergency vehicles
                emerg_idx = next(
                    (i for i, v in enumerate(queue) if v.vehicle_type == VehicleType.EMERGENCY),
                    None
                )
                if emerg_idx is not None:
                    vehicle = queue.pop(emerg_idx)
                else:
                    vehicle = queue.pop(0)
                cleared[vehicle.vehicle_type.value] += 1
                self.total_vehicles_cleared += 1

        # Resolve emergency if its direction got green
        if self.emergency is not None:
            if self.emergency["direction"] in green_dirs:
                self.emergency = None

        return cleared

    # ------------------------------------------------------------------
    # Vehicle Arrivals
    # ------------------------------------------------------------------

    def _arrive_vehicles(self) -> None:
        \"\"\"Add new VehicleRecord objects to each approach queue.
        
        Arrival count = max(0, round(rate + N(0, noise_std))).
        Queues are capped at MAX_QUEUE_SIZE.
        \"\"\"
        arrival_rates = self.task_config["arrival_rates"]
        noise_std = self.task_config["arrival_noise_std"]

        for direction in ALL_DIRECTIONS:
            rate = arrival_rates[direction]
            noise = float(self.rng.normal(0, noise_std))
            arrivals = max(0, round(rate + noise))
            
            for i in range(arrivals):
                if len(self.queues[direction]) >= MAX_QUEUE_SIZE:
                    break
                self.queues[direction].append(VehicleRecord(
                    vehicle_id=f"{direction}-{self.step_count}-{i}",
                    direction=Direction(direction),
                    vehicle_type=VehicleType.NORMAL,
                    wait_time=0,
                    arrival_step=self.step_count,
                ))

    def _increment_wait_times(self) -> None:
        \"\"\"Increment wait_time for every vehicle currently in a queue.\"\"\"
        for queue in self.queues.values():
            for vehicle in queue:
                vehicle.wait_time += 1

    # ------------------------------------------------------------------
    # Emergency Vehicle Lifecycle
    # ------------------------------------------------------------------

    def _update_emergency(self) -> None:"""
content = c456_pat.sub(c456_rep, content, count=1)

# Change 9
c9_pat = re.compile(r'    def _calculate_reward\(self, vehicles_cleared: int, action: str, emergency_handled: bool\) -> float:\n.*?(?=    # ------------------------------------------------------------------\n    # State Serialisation)', re.DOTALL)
c9_rep = """    def _calculate_reward(
        self,
        cleared_counts: dict,
        action: str,
        emergency_handled: bool
    ) -> float:
        \"\"\"Compute per-step reward.
        
        Components:
            throughput:        +1.0 per normal vehicle, +3.0 per emergency vehicle cleared.
            queue_penalty:     -0.15 per vehicle currently waiting.
            wait_penalty:      -0.10 * average wait time across all queues.
            emergency_penalty: Escalating penalty when emergency waits; severe past timeout.
            switch_penalty:    -0.25 for switching before MIN_PHASE_DURATION.
            invalid_penalty:   -2.0 for unrecognized action strings.
        \"\"\"
        # Throughput — emergency vehicles worth 3x
        throughput = (
            cleared_counts.get("normal", 0) * 1.0
            + cleared_counts.get("emergency", 0) * 3.0
        )

        # Queue metrics from VehicleRecord lists
        all_vehicles = [v for q in self.queues.values() for v in q]
        total_waiting = len(all_vehicles)
        avg_wait = (
            sum(v.wait_time for v in all_vehicles) / total_waiting
            if total_waiting > 0 else 0.0
        )

        queue_penalty = total_waiting * 0.15
        wait_penalty = avg_wait * 0.10

        # Emergency handling
        emergency_reward = 0.0
        if emergency_handled:
            emergency_reward = 0.0  # Vehicle already cleared — captured in throughput
        elif self.emergency is not None:
            steps_waited = self.emergency["steps_waiting"]
            urgency_multiplier = {"LOW": 1.0, "HIGH": 1.5, "CRITICAL": 2.0}.get(
                self.emergency.get("urgency", "LOW"), 1.0
            )
            emergency_reward = -0.2 * steps_waited * urgency_multiplier
            if steps_waited > EMERGENCY_TIMEOUT:
                emergency_reward -= 5.0

        # Switch penalty — only when switching before min duration
        switch_penalty = 0.0
        premature_switch = (
            action in ("set_ns_green", "set_ew_green", "set_all_red", "switch_phase")
            and self.phase_duration < MIN_PHASE_DURATION
            and self.emergency is None  # no penalty when emergency forces switch
        )
        if premature_switch:
            switch_penalty = -0.25

        # Invalid action penalty
        valid_actions = {
            "set_ns_green", "set_ew_green", "hold_current_phase", "set_all_red",
            "keep_current", "switch_phase", "emergency_override"
        }
        invalid_penalty = -2.0 if action not in valid_actions else 0.0

        return throughput - queue_penalty - wait_penalty + emergency_reward - switch_penalty + invalid_penalty

"""
content = c9_pat.sub(c9_rep, content, count=1)

with open("d:/PROJECTS/auto-veh/traffic-control-env/server/simulation.py", "w", encoding="utf-8") as f:
    f.write(content)

print("Done replacing.")
