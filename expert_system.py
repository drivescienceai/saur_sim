"""
expert_system.py — Экспертная система управления тушением пожара РВС.
═══════════════════════════════════════════════════════════════════════════════
Продукционная система правил «ЕСЛИ-ТО» для принятия решений РТП.
Реализована как набор правил, структурированных по фазам пожара,
с учётом текущего состояния (площадь, ресурсы, стволы, ПНС, пена и др.).

Предназначена для СРАВНЕНИЯ с агентами обучения с подкреплением:
- ЭС: фиксированные правила, детерминированный выбор, прозрачная логика
- ОП: обучаемые веса, вероятностный выбор, адаптация к новым сценариям

Каждое правило содержит:
  - условие (набор проверок текущего состояния)
  - действие (индекс 0–14)
  - приоритет (чем выше — тем раньше проверяется)
  - обоснование (текст для объяснения решения)
═══════════════════════════════════════════════════════════════════════════════
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple, Callable

try:
    from .rl_agent import N_ACTIONS, RLState
except ImportError:
    from rl_agent import N_ACTIONS, RLState

# Индексы действий
S1, S2, S3, S4, S5 = 0, 1, 2, 3, 4
T1, T2, T3, T4 = 5, 6, 7, 8
O1, O2, O3, O4, O5, O6 = 9, 10, 11, 12, 13, 14

ACTION_NAMES_RU = [
    "С1 Спасение людей", "С2 Защита соседнего РВС", "С3 Локализация",
    "С4 Пенная атака", "С5 Предотвращение вскипания",
    "Т1 Создать БУ", "Т2 Перегруппировка", "Т3 Вызов доп. сил",
    "Т4 Установить ПНС",
    "О1 Охлаждение горящего", "О2 Охлаждение соседнего",
    "О3 Пенная атака", "О4 Разведка", "О5 Ликвидация розлива",
    "О6 Сигнал отхода",
]


@dataclass
class ESState:
    """Состояние для экспертной системы — расширенное, непосредственно из симуляции."""
    phase: str              # "S1".."S5"
    fire_area: float        # м²
    water_flow: float       # л/с
    n_trunks_burn: int      # стволов на горящий РВС
    n_trunks_nbr: int       # стволов на соседний РВС
    n_pns: int              # ПНС на водоисточнике
    n_bu: int               # боевых участков
    has_shtab: bool         # штаб создан
    foam_ready: bool        # готовность к пенной атаке
    foam_conc: float        # запас пенообразователя (т)
    foam_attacks: int       # число проведённых пенных атак
    spill: bool             # розлив горящего топлива
    secondary_fire: bool    # вторичный очаг
    localized: bool         # пожар локализован
    extinguished: bool      # пожар ликвидирован
    risk: float             # индекс риска [0; 1]
    roof_obstruction: float # препятствие крыши [0; 1]
    t: int                  # текущее время (мин)
    rvs_diameter: float = 40.0  # диаметр РВС (м)

    @classmethod
    def from_sim(cls, sim) -> "ESState":
        """Извлечь состояние из объекта TankFireSim."""
        return cls(
            phase=sim.phase,
            fire_area=sim.fire_area,
            water_flow=sim.water_flow,
            n_trunks_burn=sim.n_trunks_burn,
            n_trunks_nbr=sim.n_trunks_nbr,
            n_pns=sim.n_pns,
            n_bu=sim.n_bu,
            has_shtab=sim.has_shtab,
            foam_ready=sim.foam_ready,
            foam_conc=sim.foam_conc,
            foam_attacks=sim.foam_attacks,
            spill=sim.spill,
            secondary_fire=sim.secondary_fire,
            localized=sim.localized,
            extinguished=sim.extinguished,
            risk=sim._risk(),
            roof_obstruction=sim.roof_obstruction,
            t=sim.t,
            rvs_diameter=sim._cfg.get("rvs_diameter_m", 40.0),
        )


@dataclass
class Rule:
    """Продукционное правило ЭС."""
    name: str
    phase: str                  # фаза, в которой правило действует ("*" = любая)
    priority: int               # приоритет (100 = наивысший)
    condition: Callable[[ESState], bool]
    action: int                 # индекс действия 0–14
    rationale: str              # обоснование (для объяснения)


class ExpertSystem:
    """Экспертная система управления тушением пожара РВС.

    База знаний: ~40 правил, структурированных по фазам S1–S5.
    Стратегия выбора: первое сработавшее правило с наивысшим приоритетом.
    """

    def __init__(self):
        self.rules: List[Rule] = []
        self._build_rules()
        self.decision_log: List[Dict] = []

    def _build_rules(self):
        """Создать базу правил (БУПО, Справочник РТП, СП 155.13130)."""
        R = self.rules

        # ═══════════════════════════════════════════════════════════════════
        # ФАЗА S1 — ОБНАРУЖЕНИЕ И ВЫЕЗД
        # ═══════════════════════════════════════════════════════════════════
        R.append(Rule("Разведка при прибытии", "S1", 100,
                       lambda s: s.n_trunks_burn == 0,
                       O4, "Первое действие по прибытии — разведка (БУПО §3.1)"))

        R.append(Rule("Вызов сил по рангу", "S1", 90,
                       lambda s: s.n_pns == 0,
                       T3, "Направить подразделения по рангу пожара (БУПО §4.2)"))

        R.append(Rule("Установить ПНС", "S1", 80,
                       lambda s: s.n_pns == 0 and s.t >= 5,
                       T4, "Установить ПНС на водоисточник для обеспечения подачи ОВ"))

        R.append(Rule("Спасение людей при угрозе", "S1", 95,
                       lambda s: s.risk > 0.6,
                       S1, "Спасение людей — решающее направление при высоком риске"))

        # ═══════════════════════════════════════════════════════════════════
        # ФАЗА S2 — БОЕВОЕ РАЗВЁРТЫВАНИЕ
        # ═══════════════════════════════════════════════════════════════════
        R.append(Rule("Охлаждение горящего (мало стволов)", "S2", 100,
                       lambda s: s.n_trunks_burn < 3,
                       O1, "Подать стволы на охлаждение горящего РВС (СП 155, §9.3)"))

        R.append(Rule("Охлаждение соседнего", "S2", 85,
                       lambda s: s.n_trunks_burn >= 3 and s.n_trunks_nbr < 2,
                       O2, "Охлаждать соседний РВС в зоне теплового воздействия"))

        R.append(Rule("Создать БУ", "S2", 80,
                       lambda s: s.n_bu == 0 and s.n_trunks_burn >= 2,
                       T1, "Создать боевые участки по секторам (БУПО §5.1)"))

        R.append(Rule("Установить ПНС в S2", "S2", 75,
                       lambda s: s.n_pns < 2 and s.water_flow < 200,
                       T4, "Обеспечить бесперебойную подачу ОВ — установить ПНС"))

        R.append(Rule("Защита соседнего РВС", "S2", 70,
                       lambda s: s.n_trunks_nbr >= 2,
                       S2, "Подтвердить защиту соседнего РВС как решающее направление"))

        R.append(Rule("Вызов доп. сил в S2", "S2", 60,
                       lambda s: s.n_trunks_burn < 5 and s.t > 30,
                       T3, "Запросить дополнительные силы и средства"))

        # ═══════════════════════════════════════════════════════════════════
        # ФАЗА S3 — АКТИВНОЕ ГОРЕНИЕ / ЛОКАЛИЗАЦИЯ
        # ═══════════════════════════════════════════════════════════════════
        R.append(Rule("Ликвидация розлива (экстренно)", "S3", 100,
                       lambda s: s.spill,
                       O5, "Ликвидировать розлив горящего топлива — экстренно"))

        R.append(Rule("Наращивание стволов до нормы", "S3", 95,
                       lambda s: not s.spill and s.n_trunks_burn < 7,
                       O1, "Довести число стволов на горящий РВС до нормативного (≥7)"))

        R.append(Rule("Вызов доп. сил (нехватка)", "S3", 90,
                       lambda s: s.water_flow < 400 and s.n_pns < 3,
                       T3, "Расход ОВ ниже нормы — вызвать дополнительные силы"))

        R.append(Rule("Установить ПНС в S3", "S3", 85,
                       lambda s: s.n_pns < 4 and s.water_flow < 500,
                       T4, "Установить дополнительные ПНС на водоисточники"))

        R.append(Rule("Перегруппировка в S3", "S3", 70,
                       lambda s: s.n_trunks_burn >= 7 and not s.foam_ready,
                       T2, "Перегруппировать силы в ожидании готовности к пенной атаке"))

        R.append(Rule("Подготовка пенной атаки", "S3", 80,
                       lambda s: s.n_trunks_burn >= 5 and s.foam_conc > 2 and not s.foam_ready,
                       O3, "Подготовить пенную атаку (проверка готовности)"))

        R.append(Rule("Предотвращение вскипания", "S3", 88,
                       lambda s: s.fire_area > 2000 and s.t > 500,
                       S5, "Опасность вскипания нефтепродукта — контроль уровня"))

        R.append(Rule("Разведка в S3 (обновление)", "S3", 50,
                       lambda s: s.n_trunks_burn >= 7 and s.foam_ready,
                       O4, "Уточнить обстановку перед пенной атакой"))

        # ═══════════════════════════════════════════════════════════════════
        # ФАЗА S4 — ПЕННАЯ АТАКА / ЛИКВИДАЦИЯ
        # ═══════════════════════════════════════════════════════════════════
        R.append(Rule("Пенная атака (низкое препятствие)", "S4", 100,
                       lambda s: s.foam_ready and s.roof_obstruction < 0.30,
                       O3, "Пенная атака — препятствие крыши минимально"))

        R.append(Rule("Пенная атака (высокое препятствие, АКП-50)", "S4", 95,
                       lambda s: s.foam_ready and s.roof_obstruction >= 0.30
                                 and s.foam_attacks >= 3,
                       O3, "Пенная атака через АКП-50 после 3+ неудачных попыток"))

        R.append(Rule("Ожидание АКП-50 (высокое препятствие)", "S4", 90,
                       lambda s: s.foam_ready and s.roof_obstruction >= 0.50
                                 and s.foam_attacks < 3,
                       O4, "Препятствие крыши ≥50%, запросить АКП-50 (ГОСТ Р 51043)"))

        R.append(Rule("Запрос доп. пенообразователя", "S4", 85,
                       lambda s: s.foam_conc < 3 and not s.localized,
                       T3, "Запас пенообразователя менее 3 т — запросить подвоз"))

        R.append(Rule("Непрерывное охлаждение при атаке", "S4", 80,
                       lambda s: s.n_trunks_burn < 6,
                       O1, "Не прекращать охлаждение во время пенной атаки"))

        R.append(Rule("Сигнал отхода при угрозе", "S4", 98,
                       lambda s: s.risk > 0.85,
                       O6, "Экстренный вывод личного состава при критическом риске"))

        # ═══════════════════════════════════════════════════════════════════
        # ФАЗА S5 — ЛИКВИДАЦИЯ ПОСЛЕДСТВИЙ
        # ═══════════════════════════════════════════════════════════════════
        R.append(Rule("Контроль после ликвидации", "S5", 100,
                       lambda s: True,
                       O4, "Контроль обстановки и дежурство"))

        R.append(Rule("Ликвидация розлива в S5", "S5", 95,
                       lambda s: s.spill,
                       O5, "Ликвидировать остаточный розлив топлива"))

        R.append(Rule("Охлаждение после ликвидации", "S5", 80,
                       lambda s: s.n_trunks_nbr < 2,
                       O2, "Продолжить охлаждение резервуаров ≥6 ч (СП 155)"))

        # Сортировка: по фазе, затем по приоритету (убывание)
        self.rules.sort(key=lambda r: (-r.priority,))

    def select_action(self, state: ESState) -> Tuple[int, str, str]:
        """Выбрать действие по правилам ЭС.

        Возвращает: (action_idx, rule_name, rationale)
        """
        phase = state.phase

        for rule in self.rules:
            if rule.phase != "*" and rule.phase != phase:
                continue
            try:
                if rule.condition(state):
                    self.decision_log.append({
                        "t": state.t,
                        "phase": phase,
                        "action": rule.action,
                        "rule": rule.name,
                        "rationale": rule.rationale,
                    })
                    return rule.action, rule.name, rule.rationale
            except Exception:
                continue

        # Действие по умолчанию — разведка
        return O4, "По умолчанию", "Нет подходящего правила — провести разведку"

    def select_action_from_sim(self, sim) -> Tuple[int, str, str]:
        """Выбрать действие из объекта TankFireSim."""
        return self.select_action(ESState.from_sim(sim))

    def reset(self):
        """Сбросить журнал решений."""
        self.decision_log = []

    def stats(self) -> Dict:
        """Статистика использования правил."""
        if not self.decision_log:
            return {"n_decisions": 0}
        rule_counts = {}
        action_counts = {}
        for entry in self.decision_log:
            rule_counts[entry["rule"]] = rule_counts.get(entry["rule"], 0) + 1
            a = entry["action"]
            action_counts[a] = action_counts.get(a, 0) + 1
        return {
            "n_decisions": len(self.decision_log),
            "n_unique_rules": len(rule_counts),
            "top_rules": sorted(rule_counts.items(), key=lambda x: -x[1])[:5],
            "action_distribution": action_counts,
        }

    @property
    def n_rules(self) -> int:
        return len(self.rules)

    def explain_last(self) -> str:
        """Объяснение последнего решения."""
        if not self.decision_log:
            return "Нет решений"
        last = self.decision_log[-1]
        return (f"Правило: {last['rule']}\n"
                f"Действие: {ACTION_NAMES_RU[last['action']]}\n"
                f"Обоснование: {last['rationale']}")


# ═══════════════════════════════════════════════════════════════════════════
# СРАВНИТЕЛЬНЫЙ ПРОГОН: ЭС vs ОП
# ═══════════════════════════════════════════════════════════════════════════
def compare_es_vs_rl(scenario: str = "tuapse", n_episodes: int = 10,
                     seed: int = 42) -> Dict:
    """Провести сравнительный эксперимент: ЭС vs Табличный ОП vs Иерарх. ОП.

    Возвращает: словарь с метриками для каждого агента.
    """
    import numpy as np
    try:
        from .tank_fire_sim import TankFireSim, PHASE_VALID
    except ImportError:
        from tank_fire_sim import TankFireSim, PHASE_VALID

    results = {"es": [], "rl_flat": []}
    es = ExpertSystem()

    for ep in range(n_episodes):
        # ── Прогон ЭС ────────────────────────────────────────────────────
        sim_es = TankFireSim(seed=seed + ep, training=False, scenario=scenario)
        es.reset()
        while sim_es.t < sim_es._cfg["total_min"] and not sim_es.extinguished:
            action, _, _ = es.select_action_from_sim(sim_es)
            sim_es.step(dt=5, action=action)

        results["es"].append({
            "extinguished": sim_es.extinguished,
            "t_final": sim_es.t,
            "foam_attacks": sim_es.foam_attacks,
            "total_reward": sum(sim_es.h_reward),
            "risk_max": max((r for _, r in sim_es.h_risk), default=0),
            "n_rules_fired": len(es.decision_log),
        })

        # ── Прогон Табличный ОП ───────────────────────────────────────────
        sim_rl = TankFireSim(seed=seed + ep, training=False, scenario=scenario)
        while sim_rl.t < sim_rl._cfg["total_min"] and not sim_rl.extinguished:
            sim_rl.step(dt=5)  # агент выбирает действие внутри step()

        results["rl_flat"].append({
            "extinguished": sim_rl.extinguished,
            "t_final": sim_rl.t,
            "foam_attacks": sim_rl.foam_attacks,
            "total_reward": sum(sim_rl.h_reward),
            "risk_max": max((r for _, r in sim_rl.h_risk), default=0),
        })

    # Агрегация
    summary = {}
    for agent_name, episodes in results.items():
        arr = {k: [e[k] for e in episodes] for k in episodes[0]}
        summary[agent_name] = {
            "success_rate": np.mean([e["extinguished"] for e in episodes]),
            "t_final_mean": np.mean(arr["t_final"]),
            "foam_attacks_mean": np.mean(arr["foam_attacks"]),
            "reward_mean": np.mean(arr["total_reward"]),
            "risk_max_mean": np.mean(arr["risk_max"]),
        }

    return summary


if __name__ == "__main__":
    es = ExpertSystem()
    print(f"Экспертная система: {es.n_rules} правил")
    print("\nСравнительный эксперимент (5 эпизодов):")
    results = compare_es_vs_rl(n_episodes=5)
    for agent, metrics in results.items():
        print(f"\n  {agent}:")
        for k, v in metrics.items():
            print(f"    {k}: {v:.3f}")
