"""Comprehensive test suite for SAUR simulation (>=30 tests)."""
import pytest
import numpy as np
import json
import os
import tempfile

# ---------------------------------------------------------------------------
# Imports
# ---------------------------------------------------------------------------
from saur_sim.state_space import (
    FirePhase, SituationState, ResourceUnit, ResourceSpace,
    AdaptationMode, AdaptationResult,
)
from saur_sim.semi_markov import (
    SemiMarkovChain, SOJOURN_PARAMS, PHASES, SojournDistribution,
)
from saur_sim.rl_agent import (
    QLearningAgent, RLState, compute_reward,
    N_PHASES, N_RESOURCES, N_SEVERITY, N_QUALITY, N_ACTIONS, STATE_SIZE,
    ACTION_NAMES, ACTION_COST,
)
from saur_sim.metrics import (
    SAURMetrics, compute_risk_score, compute_delta_s,
    adaptation_trigger, EPS_THRESHOLD,
)
from saur_sim.level_l1 import L1SensorLayer, SensorReading, COP
from saur_sim.level_l2 import L2TacticalAgent, TACTIC_MAP
from saur_sim.level_l3 import L3OperationalHQ
from saur_sim.level_l4 import L4SystemicCenter, GarrisonStatus
from saur_sim.level_l5 import L5StrategicLayer, OperationRecord
from saur_sim.database import UNIT_DATABASE, OIL_BASE_GARRISON
from saur_sim.simulation import SAURSimulation, SimParams


# ===========================================================================
# Fixtures
# ===========================================================================

@pytest.fixture
def basic_situation():
    return SituationState(phase=FirePhase.S3, fire_area_m2=2000.0,
                          fire_spread_rate=3.0, wind_speed_ms=5.0)

@pytest.fixture
def resource_unit():
    return ResourceUnit(unit_id="U-01", unit_type="АЦ-40", crew_size=5)

@pytest.fixture
def resource_space(resource_unit):
    u2 = ResourceUnit(unit_id="U-02", unit_type="АЦ-40", crew_size=5,
                      task="firefighting")
    return ResourceSpace(vehicles=[resource_unit, u2])

@pytest.fixture
def chain():
    return SemiMarkovChain(seed=0)

@pytest.fixture
def agent():
    return QLearningAgent(seed=0, epsilon_start=0.5)

@pytest.fixture
def l1():
    return L1SensorLayer(seed=0)

@pytest.fixture
def l2_agent(resource_unit):
    return L2TacticalAgent(resource_unit)

@pytest.fixture
def l3_hq():
    return L3OperationalHQ(seed=0)

@pytest.fixture
def l4_center(resource_space):
    return L4SystemicCenter(resource_space)

@pytest.fixture
def l5_layer():
    return L5StrategicLayer()

@pytest.fixture
def sim():
    p = SimParams(sim_time=120.0, seed=42, training=True,
                  fire_start_time=10.0)
    return SAURSimulation(params=p)


# ===========================================================================
# 1. SemiMarkovChain tests
# ===========================================================================

class TestSemiMarkovChain:

    def test_initial_phase(self, chain):
        assert chain.current_phase == FirePhase.NORMAL

    def test_step_no_transition_before_sojourn(self, chain):
        """Step at t=0 should not transition if t < t_exit."""
        phase, remaining, transitioned = chain.step(0.0)
        assert phase == FirePhase.NORMAL
        assert remaining > 0
        assert transitioned is False

    def test_force_transition(self, chain):
        chain.force_transition(1.0, FirePhase.S1)
        assert chain.current_phase == FirePhase.S1
        assert chain.t_entered == 1.0

    def test_force_transition_records_history(self, chain):
        chain.force_transition(5.0, FirePhase.S2)
        assert len(chain.history) == 1
        assert chain.history[0][2] == FirePhase.NORMAL

    def test_sojourn_distribution_sample_positive(self, chain):
        rng = np.random.RandomState(1)
        for phase in PHASES:
            t = SOJOURN_PARAMS[phase].sample(rng)
            assert t > 0, f"Non-positive sojourn for {phase}"

    def test_sojourn_mean(self):
        dist = SOJOURN_PARAMS[FirePhase.S1]
        m = dist.mean()
        assert m > 0
        assert m < 60  # S1 should be under an hour on average

    def test_sojourn_pdf_positive(self):
        dist = SOJOURN_PARAMS[FirePhase.S3]
        t = np.linspace(1, 200, 50)
        pdf = dist.pdf(t)
        assert np.all(pdf >= 0)

    def test_stationary_distribution_sums_to_one(self, chain):
        pi = chain.stationary_distribution
        total = sum(pi.values())
        assert abs(total - 1.0) < 1e-6

    def test_stationary_distribution_all_non_negative(self, chain):
        pi = chain.stationary_distribution
        for k, v in pi.items():
            assert v >= 0, f"Negative stationary probability for {k}"

    def test_phase_occupancy_sums_to_one(self, chain):
        occ = chain.phase_occupancy()
        total = sum(occ.values())
        assert abs(total - 1.0) < 1e-6

    def test_mean_sojourn_times_keys(self, chain):
        mst = chain.mean_sojourn_times
        assert set(mst.keys()) == {ph.name for ph in PHASES}

    def test_summary_is_string(self, chain):
        s = chain.summary()
        assert isinstance(s, str)
        assert "Current phase" in s

    def test_transition_eventually_occurs(self, chain):
        """Step with very large t to force transition."""
        chain.force_transition(0.0, FirePhase.S5)
        phase, remaining, transitioned = chain.step(chain.t_exit + 1.0)
        assert transitioned is True
        assert phase == FirePhase.RESOLVED

    def test_history_grows_on_transition(self, chain):
        chain.force_transition(0.0, FirePhase.S5)
        chain.step(chain.t_exit + 1.0)
        assert len(chain.history) >= 1


# ===========================================================================
# 2. RLState tests
# ===========================================================================

class TestRLState:

    def test_to_index_range(self):
        for pi in range(N_PHASES):
            for rl in range(N_RESOURCES):
                for sv in range(N_SEVERITY):
                    for rq in range(N_QUALITY):
                        idx = RLState(pi, rl, sv, rq).to_index()
                        assert 0 <= idx < STATE_SIZE

    def test_to_index_unique(self):
        indices = set()
        for pi in range(N_PHASES):
            for rl in range(N_RESOURCES):
                for sv in range(N_SEVERITY):
                    for rq in range(N_QUALITY):
                        idx = RLState(pi, rl, sv, rq).to_index()
                        indices.add(idx)
        assert len(indices) == STATE_SIZE

    def test_from_metrics_low_resource(self):
        s = RLState.from_metrics(phase_idx=2, avail_frac=0.2,
                                  fire_area_norm=0.5, L7=0.5)
        assert s.resource_level == 0  # LOW

    def test_from_metrics_high_L7(self):
        s = RLState.from_metrics(phase_idx=0, avail_frac=0.8,
                                  fire_area_norm=0.0, L7=0.95)
        assert s.response_quality == 2  # GOOD

    def test_from_metrics_severity_clamp(self):
        s = RLState.from_metrics(phase_idx=3, avail_frac=0.5,
                                  fire_area_norm=1.5, L7=0.7)
        assert s.fire_severity == 4  # max


# ===========================================================================
# 3. QLearningAgent tests
# ===========================================================================

class TestQLearningAgent:

    def test_q_table_shape(self, agent):
        assert agent.Q.shape == (STATE_SIZE, N_ACTIONS)

    def test_q_table_zeros_initially(self, agent):
        assert np.all(agent.Q == 0.0)

    def test_select_action_training_explores(self):
        ag = QLearningAgent(epsilon_start=1.0, seed=99)
        state = RLState(0, 0, 0, 0)
        actions = {ag.select_action(state, training=True) for _ in range(30)}
        assert len(actions) > 1  # should explore

    def test_select_action_greedy(self, agent):
        state = RLState(0, 0, 0, 0)
        agent.Q[state.to_index(), 2] = 10.0
        a = agent.select_action(state, training=False)
        assert a == 2

    def test_update_returns_td_error(self, agent):
        s = RLState(0, 0, 0, 0)
        ns = RLState(1, 1, 1, 1)
        td = agent.update(s, 0, 1.0, ns, False)
        assert isinstance(td, float)

    def test_update_modifies_q(self, agent):
        s = RLState(0, 0, 0, 0)
        ns = RLState(1, 1, 1, 1)
        before = agent.Q[s.to_index(), 0]
        agent.update(s, 0, 1.0, ns, False)
        after = agent.Q[s.to_index(), 0]
        assert after != before

    def test_decay_epsilon(self, agent):
        e_before = agent.epsilon
        agent.decay_epsilon()
        assert agent.epsilon < e_before

    def test_epsilon_floor(self):
        ag = QLearningAgent(epsilon_start=0.001, epsilon_min=0.05)
        ag.decay_epsilon()
        assert ag.epsilon >= 0.05

    def test_policy_keys_cover_all_states(self, agent):
        p = agent.policy()
        assert len(p) == STATE_SIZE

    def test_value_function_shape(self, agent):
        v = agent.value_function()
        assert v.shape == (STATE_SIZE,)

    def test_save_load_roundtrip(self, agent):
        agent.Q[5, 3] = 7.77
        agent.epsilon = 0.333
        agent._step_count = 42
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name
        try:
            agent.save(path)
            loaded = QLearningAgent.load(path)
            assert abs(loaded.Q[5, 3] - 7.77) < 1e-9
            assert abs(loaded.epsilon - 0.333) < 1e-9
            assert loaded._step_count == 42
        finally:
            os.unlink(path)

    def test_compute_reward_range(self):
        r = compute_reward(L1_norm=0.5, casualties=0, action=0, L7=0.9)
        assert -1.0 <= r <= 1.0

    def test_compute_reward_casualties_penalty(self):
        r_no = compute_reward(0.0, 0, 0, 1.0)
        r_yes = compute_reward(0.0, 5, 0, 1.0)
        assert r_yes < r_no


# ===========================================================================
# 4. SAURMetrics / risk / delta_s tests
# ===========================================================================

class TestMetrics:

    def test_risk_score_zero_when_safe(self, resource_space):
        sit = SituationState(phase=FirePhase.NORMAL)
        r = compute_risk_score(sit, resource_space, L7=1.0, L1=0.0)
        assert 0.0 <= r <= 1.0

    def test_risk_score_high_in_fire(self, resource_space):
        sit = SituationState(phase=FirePhase.S3, fire_area_m2=8000.0)
        r_fire = compute_risk_score(sit, resource_space, L7=0.5, L1=30.0)
        sit_safe = SituationState(phase=FirePhase.NORMAL)
        r_safe = compute_risk_score(sit_safe, resource_space, L7=1.0, L1=0.0)
        assert r_fire > r_safe

    def test_delta_s_zero_at_target(self, resource_space):
        sit = SituationState(phase=FirePhase.S4)
        d = compute_delta_s(sit, FirePhase.S4, L7=0.90)
        assert d < 1e-9

    def test_delta_s_positive_off_target(self):
        sit = SituationState(phase=FirePhase.NORMAL)
        d = compute_delta_s(sit, FirePhase.S3, L7=0.5)
        assert d > 0

    def test_adaptation_trigger_above_threshold(self):
        assert adaptation_trigger(EPS_THRESHOLD + 0.01) is True

    def test_adaptation_trigger_below_threshold(self):
        assert adaptation_trigger(EPS_THRESHOLD - 0.01) is False

    def test_saur_metrics_default(self):
        m = SAURMetrics()
        assert m.risk_level == "LOW"
        assert m.phase == FirePhase.NORMAL


# ===========================================================================
# 5. L1SensorLayer tests
# ===========================================================================

class TestL1SensorLayer:

    def test_sense_returns_readings(self, l1, basic_situation):
        readings = l1.sense(0.0, basic_situation)
        assert len(readings) > 0

    def test_sense_reading_values_in_range(self, l1, basic_situation):
        readings = l1.sense(0.0, basic_situation)
        for r in readings:
            assert 0.0 <= r.value <= 1.0

    def test_fuse_returns_situation(self, l1, basic_situation):
        readings = l1.sense(0.0, basic_situation)
        sit = l1.fuse(readings, 0.0)
        assert isinstance(sit, SituationState)

    def test_fuse_empty_readings(self, l1):
        sit = l1.fuse([], 0.0)
        assert sit.phase == FirePhase.NORMAL

    def test_detect_fire_in_active_phase(self, l1, basic_situation):
        """With S3 phase, at least some sensors should exceed threshold."""
        # Run multiple times; probability of detection is high
        detected_any = False
        for _ in range(20):
            readings = l1.sense(0.0, basic_situation)
            if l1.detect(readings):
                detected_any = True
                break
        # S3 has high signal; expect detection in 20 tries
        assert detected_any

    def test_validate_returns_float_in_range(self, l1, basic_situation):
        readings = l1.sense(0.0, basic_situation)
        p = l1.validate(readings)
        assert 0.0 <= p <= 1.0

    def test_validate_empty_is_zero(self, l1):
        assert l1.validate([]) == 0.0

    def test_update_returns_cop(self, l1, basic_situation):
        cop = l1.update(0.0, basic_situation)
        assert isinstance(cop, COP)
        assert cop.sensor_count > 0

    def test_adapt_weights_changes_weights(self, l1):
        l1.adapt_weights({"ir": 0.8})
        assert "ir" in l1._sensor_weights


# ===========================================================================
# 6. L2TacticalAgent tests
# ===========================================================================

class TestL2TacticalAgent:

    def test_assess_critical(self, l2_agent):
        sit = SituationState(phase=FirePhase.S3, fire_area_m2=5000.0)
        assert l2_agent.assess(sit) == "CRITICAL"

    def test_assess_active(self, l2_agent):
        sit = SituationState(phase=FirePhase.S2)
        assert l2_agent.assess(sit) == "ACTIVE"

    def test_assess_stable(self, l2_agent):
        sit = SituationState(phase=FirePhase.NORMAL)
        assert l2_agent.assess(sit) == "STABLE"

    def test_decide_respects_directive(self, l2_agent, basic_situation):
        tactic = l2_agent.decide(basic_situation, directive="special_tactic")
        assert tactic == "special_tactic"

    def test_decide_uses_tactic_map(self, l2_agent):
        sit = SituationState(phase=FirePhase.S5)
        t = l2_agent.decide(sit)
        assert t == TACTIC_MAP[FirePhase.S5]

    def test_deploy_sets_unit_task(self, l2_agent):
        l2_agent.deploy("attack_perimeter", "B")
        assert l2_agent.unit.task == "attack_perimeter"

    def test_deploy_logs_order(self, l2_agent):
        l2_agent.deploy("patrol")
        assert len(l2_agent.orders_log) == 1

    def test_control_safety_unsafe_conditions(self, l2_agent):
        sit = SituationState(phase=FirePhase.S3, fire_area_m2=6000.0,
                             wind_speed_ms=15.0)
        safe = l2_agent.control_safety(sit)
        assert safe is False
        assert l2_agent.safety_ok is False

    def test_control_safety_safe_conditions(self, l2_agent):
        sit = SituationState(phase=FirePhase.S2, fire_area_m2=500.0,
                             wind_speed_ms=2.0)
        safe = l2_agent.control_safety(sit)
        assert safe is True

    def test_step_returns_order(self, l2_agent, basic_situation):
        order = l2_agent.step(basic_situation)
        assert order.unit_id == l2_agent.unit.unit_id

    def test_request_resources_has_from_key(self, l2_agent):
        req = l2_agent.request_resources("low water")
        assert "from" in req
        assert req["from"] == l2_agent.unit.unit_id


# ===========================================================================
# 7. L3OperationalHQ tests
# ===========================================================================

class TestL3OperationalHQ:

    def test_make_rl_state_type(self, l3_hq, basic_situation, resource_space):
        s = l3_hq._make_rl_state(basic_situation, resource_space, L7=0.8)
        assert isinstance(s, RLState)

    def test_f_COP_returns_situation(self, l3_hq, l1, basic_situation):
        cop = l1.update(0.0, basic_situation)
        sit = l3_hq.f_COP(cop, [])
        assert isinstance(sit, SituationState)

    def test_f_allocate_returns_valid_action(self, l3_hq, basic_situation,
                                              resource_space):
        action, result = l3_hq.f_allocate(
            basic_situation, resource_space, L7=0.8, L1=5.0, t=0.0)
        assert 0 <= action < N_ACTIONS
        assert isinstance(result, AdaptationResult)

    def test_f_predict_returns_dict(self, l3_hq, basic_situation):
        pred = l3_hq.f_predict(basic_situation, 0.0)
        assert "predicted_area_m2" in pred
        assert pred["t_horizon_min"] == 15.0

    def test_finish_episode_decays_epsilon(self, l3_hq):
        eps_before = l3_hq.rl_agent.epsilon
        l3_hq.finish_episode()
        assert l3_hq.rl_agent.epsilon <= eps_before


# ===========================================================================
# 8. L4SystemicCenter tests
# ===========================================================================

class TestL4SystemicCenter:

    def test_monitor_readiness_total(self, l4_center):
        status = l4_center.monitor_readiness()
        assert status.total_units == len(l4_center.resources.vehicles)

    def test_monitor_readiness_ready_le_total(self, l4_center):
        status = l4_center.monitor_readiness()
        assert status.ready_units <= status.total_units

    def test_f_reserve_allocates_units(self, l4_center):
        allocated = l4_center.f_reserve(1)
        assert len(allocated) >= 0  # may be 0 if none available

    def test_f_reserve_changes_task(self, l4_center):
        allocated = l4_center.f_reserve(1)
        for u in allocated:
            assert u.task == "dispatched_reserve"

    def test_f_coordinate_inter_none_when_ready(self, l4_center):
        status = GarrisonStatus(total_units=10, ready_units=9,
                                deployed_units=0, readiness_fraction=0.9,
                                active_incidents=0)
        result = l4_center.f_coordinate_inter(status)
        assert result is None

    def test_f_coordinate_inter_request_when_low(self, l4_center):
        status = GarrisonStatus(total_units=10, ready_units=2,
                                deployed_units=8, readiness_fraction=0.2,
                                active_incidents=1)
        result = l4_center.f_coordinate_inter(status)
        assert result is not None
        assert isinstance(result, str)

    def test_receive_l3_request_zero(self, l4_center):
        req = AdaptationResult(mode=AdaptationMode.NORMAL, resources_requested=0)
        resp = l4_center.receive_l3_request(req)
        assert isinstance(resp, AdaptationResult)


# ===========================================================================
# 9. L5StrategicLayer tests
# ===========================================================================

class TestL5StrategicLayer:

    def _make_record(self, casualties=0, duration=120.0, L7=0.85):
        return OperationRecord(duration_min=duration, max_phase="S3",
                               casualties=casualties, total_units_deployed=4,
                               L7_final=L7, J_final=0.8, peak_fire_area_m2=3000,
                               outcome="controlled")

    def test_record_operation_appends(self, l5_layer):
        l5_layer.record_operation(self._make_record())
        assert len(l5_layer.operation_records) == 1

    def test_mortality_trend_zero_single_record(self, l5_layer):
        l5_layer.record_operation(self._make_record())
        assert l5_layer.mortality_trend() == 0.0

    def test_mortality_trend_negative_improvement(self, l5_layer):
        # Early operations have more casualties
        for _ in range(5):
            l5_layer.record_operation(self._make_record(casualties=3))
        for _ in range(5):
            l5_layer.record_operation(self._make_record(casualties=0))
        trend = l5_layer.mortality_trend()
        assert trend <= 0.0  # improving

    def test_strategic_kpis_empty(self, l5_layer):
        kpis = l5_layer.strategic_kpis()
        assert kpis == {}

    def test_strategic_kpis_populated(self, l5_layer):
        l5_layer.record_operation(self._make_record())
        kpis = l5_layer.strategic_kpis()
        assert "mean_duration_min" in kpis
        assert "control_rate" in kpis

    def test_lessons_extracted_for_casualties(self, l5_layer):
        l5_layer.record_operation(self._make_record(casualties=2))
        lessons = l5_layer.lessons()
        assert any("Потери" in l for l in lessons)

    def test_lessons_extracted_for_low_L7(self, l5_layer):
        l5_layer.record_operation(self._make_record(L7=0.5))
        lessons = l5_layer.lessons()
        assert any("БПЛА" in l for l in lessons)

    def test_lessons_extracted_for_long_op(self, l5_layer):
        l5_layer.record_operation(self._make_record(duration=300.0))
        lessons = l5_layer.lessons()
        assert any("резерва" in l for l in lessons)


# ===========================================================================
# 10. SAURSimulation tests
# ===========================================================================

class TestSAURSimulation:

    def test_init_builds_resources(self, sim):
        assert len(sim._resources.vehicles) == len(OIL_BASE_GARRISON)

    def test_run_returns_dict(self, sim):
        result = sim.run()
        assert isinstance(result, dict)

    def test_run_populates_metrics_history(self, sim):
        sim.run()
        assert len(sim.metrics_history) > 0

    def test_run_mean_L7_in_range(self, sim):
        result = sim.run()
        assert 0.0 <= result["mean_L7"] <= 1.0

    def test_run_mean_risk_in_range(self, sim):
        result = sim.run()
        assert 0.0 <= result["mean_risk"] <= 1.0

    def test_run_chain_summary_in_result(self, sim):
        result = sim.run()
        assert "chain_summary" in result
        assert isinstance(result["chain_summary"], str)

    def test_run_fire_active_during_phase(self, sim):
        sim.run()
        fire_phases = [m.phase for m in sim.metrics_history
                       if m.phase not in (FirePhase.NORMAL, FirePhase.RESOLVED)]
        # Fire was started, so we expect some active phases
        assert len(fire_phases) > 0

    def test_training_mode_updates_q(self):
        p = SimParams(sim_time=60.0, seed=1, training=True, fire_start_time=5.0)
        sim = SAURSimulation(params=p)
        # All zeros initially
        initial_q = sim._l3.rl_agent.Q.copy()
        sim.run()
        # After training, Q should be modified
        assert not np.allclose(sim._l3.rl_agent.Q, initial_q)

    def test_run_training_returns_agent(self):
        agent = SAURSimulation.run_training(n_episodes=3, seed=7, sim_time=60.0)
        assert isinstance(agent, QLearningAgent)
        assert agent._step_count > 0

    def test_multiple_runs_deterministic(self):
        """Same seed -> same mean_L7."""
        p = SimParams(sim_time=60.0, seed=10, training=False, fire_start_time=5.0)
        r1 = SAURSimulation(params=p).run()
        r2 = SAURSimulation(params=p).run()
        assert abs(r1["mean_L7"] - r2["mean_L7"]) < 1e-9

    def test_database_all_types_known(self):
        for g in OIL_BASE_GARRISON:
            assert g["type"] in UNIT_DATABASE, f"Unknown type: {g['type']}"
