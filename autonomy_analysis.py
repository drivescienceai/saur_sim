"""
autonomy_analysis.py — Вычисление и визуализация коэффициентов автономности α.
═══════════════════════════════════════════════════════════════════════════════
Коэффициент автономности α ∈ [0; 1] — мера того, насколько уровень управления
принимает решения самостоятельно, а не следует указаниям вышестоящего уровня.

Формальные определения:

  α_L1(t) = 1 − I(a(t) ∈ GOAL_ACTION_MAP[g(t)])
            Доля шагов, на которых L1 отклонился от цели L2.
            α=0 → всегда следует цели, α=1 → полностью игнорирует.

  α_L2(t) = 1 − I(g(t) ∈ L3_GOAL_ALLOW[m(t)])
            Доля шагов, на которых L2 выбрал цель вне разрешённых L3.
            (При жёсткой маске всегда 0; при мягкой — измеряет отклонение.)

  α_L3(t) = 1 − I(m(t) == m(t−k₃))
            Частота смены режима управления (высокая = высокая автономность).

  α_L4 = 1 − (garrison_readiness)
            Чем ниже готовность гарнизона, тем выше нужна автономность.

  α_L5 = определяется стратегическим трендом (mortality_trend).

Агрегация: α_level = mean(α(t)) по всем шагам эпизода или по фазе.

Визуализации:
  1. α по уровням (столбчатая, общая за эпизод)
  2. α по фазам пожара (тепловая карта уровень × фаза)
  3. α(t) — динамика во времени (линейный график)
  4. α(λ) — зависимость α_L1 от λ_intrinsic (параметрическая кривая)
  5. Сравнение α между сценариями
═══════════════════════════════════════════════════════════════════════════════
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

try:
    from .rl_agent import N_ACTIONS
    from .hrl_agents import GOAL_ACTION_MAP, L3_GOAL_ALLOW, HGoal, L3Mode
except ImportError:
    from rl_agent import N_ACTIONS
    from hrl_agents import GOAL_ACTION_MAP, L3_GOAL_ALLOW, HGoal, L3Mode

_OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "data", "figures")
os.makedirs(_OUT_DIR, exist_ok=True)

LEVEL_NAMES = ["L1\n(оперативный)", "L2\n(тактический)", "L3\n(стратегический)",
               "L4\n(гарнизон)", "L5\n(институц.)"]
LEVEL_COLORS = ["#2980b9", "#e67e22", "#c0392b", "#8e44ad", "#27ae60"]
PHASE_LABELS = ["S1", "S2", "S3", "S4", "S5"]


# ═══════════════════════════════════════════════════════════════════════════
# СТРУКТУРА ДАННЫХ: ТРАССА ИЕРАРХИЧЕСКОГО ЭПИЗОДА
# ═══════════════════════════════════════════════════════════════════════════
@dataclass
class HRLStepRecord:
    """Запись одного шага иерархической симуляции."""
    t: int                   # время (мин)
    phase: str               # фаза пожара ("S1".."S5")
    l3_mode: int             # текущий режим L3 (0, 1, 2)
    l2_goal: int             # текущая цель L2 (0..4)
    l1_action: int           # выбранное действие L1 (0..14)
    l1_in_goal: bool         # действие ∈ GOAL_ACTION_MAP[goal]?
    l2_in_allowed: bool      # цель ∈ L3_GOAL_ALLOW[mode]?
    l3_changed: bool         # режим сменился на этом шаге?
    env_reward: float = 0.0  # награда от среды
    intrinsic_reward: float = 0.0  # внутренняя награда


@dataclass
class AutonomyResult:
    """Результат анализа автономности за эпизод."""
    # Общие α за эпизод
    alpha_L1: float = 0.0    # доля отклонений L1 от цели L2
    alpha_L2: float = 0.0    # доля отклонений L2 от разрешений L3
    alpha_L3: float = 0.0    # частота смены режима L3
    alpha_L4: float = 0.0    # автономность гарнизона
    alpha_L5: float = 0.0    # институциональная автономность

    # α по фазам: dict[phase_str] → α
    alpha_L1_by_phase: Dict[str, float] = field(default_factory=dict)
    alpha_L2_by_phase: Dict[str, float] = field(default_factory=dict)
    alpha_L3_by_phase: Dict[str, float] = field(default_factory=dict)

    # α(t) — временно́й ряд
    alpha_L1_timeline: List[Tuple[int, float]] = field(default_factory=list)
    alpha_L2_timeline: List[Tuple[int, float]] = field(default_factory=list)

    # Метаданные
    n_steps: int = 0
    scenario: str = ""
    lambda_intrinsic: float = 0.30


# ═══════════════════════════════════════════════════════════════════════════
# ВЫЧИСЛЕНИЕ α ИЗ ТРАССЫ
# ═══════════════════════════════════════════════════════════════════════════
def compute_autonomy(trace: List[HRLStepRecord],
                     scenario: str = "",
                     lambda_intrinsic: float = 0.30,
                     garrison_readiness: float = 0.80,
                     mortality_trend: float = -0.02) -> AutonomyResult:
    """Вычислить коэффициенты автономности из трассы HRL-эпизода.

    Аргументы:
        trace: список записей шагов (из HRL-симуляции)
        garrison_readiness: готовность гарнизона (из L4)
        mortality_trend: тренд потерь (из L5)

    Возвращает: AutonomyResult с α по уровням, фазам и времени.
    """
    if not trace:
        return AutonomyResult(scenario=scenario,
                              lambda_intrinsic=lambda_intrinsic)

    n = len(trace)
    result = AutonomyResult(n_steps=n, scenario=scenario,
                            lambda_intrinsic=lambda_intrinsic)

    # ── α_L1: отклонение от цели L2 ──────────────────────────────────────
    deviations_l1 = [1.0 - float(s.l1_in_goal) for s in trace]
    result.alpha_L1 = float(np.mean(deviations_l1))

    # ── α_L2: отклонение от разрешений L3 ─────────────────────────────────
    deviations_l2 = [1.0 - float(s.l2_in_allowed) for s in trace]
    result.alpha_L2 = float(np.mean(deviations_l2))

    # ── α_L3: частота смены режима ─────────────────────────────────────────
    changes_l3 = [float(s.l3_changed) for s in trace]
    result.alpha_L3 = float(np.mean(changes_l3))

    # ── α_L4: автономность гарнизона ──────────────────────────────────────
    # Чем ниже готовность → тем выше нужна автономность (самоорганизация)
    result.alpha_L4 = 1.0 - garrison_readiness

    # ── α_L5: институциональная ──────────────────────────────────────────
    # Тренд потерь < 0 → система обучается → высокая автономность
    result.alpha_L5 = min(1.0, max(0.0, 0.5 - mortality_trend * 10))

    # ── α по фазам ─────────────────────────────────────────────────────────
    for phase in PHASE_LABELS:
        steps_in_phase = [s for s in trace if s.phase == phase]
        if steps_in_phase:
            result.alpha_L1_by_phase[phase] = float(np.mean(
                [1.0 - float(s.l1_in_goal) for s in steps_in_phase]))
            result.alpha_L2_by_phase[phase] = float(np.mean(
                [1.0 - float(s.l2_in_allowed) for s in steps_in_phase]))
            result.alpha_L3_by_phase[phase] = float(np.mean(
                [float(s.l3_changed) for s in steps_in_phase]))

    # ── α(t) — скользящее среднее по окну 10 шагов ────────────────────────
    window = min(10, max(1, n // 5))
    for i in range(n):
        start = max(0, i - window + 1)
        chunk = deviations_l1[start:i+1]
        result.alpha_L1_timeline.append((trace[i].t, float(np.mean(chunk))))
        chunk2 = deviations_l2[start:i+1]
        result.alpha_L2_timeline.append((trace[i].t, float(np.mean(chunk2))))

    # Автосохранение в централизованную БД
    try:
        from results_db import get_db
        get_db().log_autonomy(
            {"L1": result.alpha_L1, "L2": result.alpha_L2,
             "L3": result.alpha_L3, "L4": result.alpha_L4,
             "L5": result.alpha_L5},
            scenario=scenario)
    except Exception:
        pass

    return result


def build_trace_from_hrl_sim(hrl_sim) -> List[HRLStepRecord]:
    """Извлечь трассу из объекта HierarchicalTankFireSim после прогона.

    Использует goal_history, mode_history и данные среды.
    """
    env = hrl_sim._env
    records = []

    # Собрать историю режимов и целей с временны́ми метками
    mode_at_t = {}
    for t, mode in getattr(hrl_sim, "mode_history", []):
        mode_at_t[t] = mode
    goal_at_t = {}
    for t, goal in getattr(hrl_sim, "goal_history", []):
        goal_at_t[t] = goal

    # Восстановить текущий режим/цель для каждого шага
    current_mode = 0
    current_goal = 0
    prev_mode = 0

    for i, (t_val, phase_num) in enumerate(env.h_phase):
        if t_val in mode_at_t:
            prev_mode = current_mode
            current_mode = mode_at_t[t_val]
        if t_val in goal_at_t:
            current_goal = goal_at_t[t_val]

        # Определить действие (из истории наград / действий)
        action = env.last_action if i == len(env.h_phase) - 1 else 0
        phase_str = f"S{phase_num}" if 1 <= phase_num <= 5 else "S1"

        # Проверить соответствие
        goal_actions = GOAL_ACTION_MAP.get(current_goal, [])
        in_goal = action in goal_actions
        allowed_goals = L3_GOAL_ALLOW.get(current_mode, list(range(5)))
        in_allowed = current_goal in allowed_goals
        mode_changed = (current_mode != prev_mode) and (t_val in mode_at_t)

        records.append(HRLStepRecord(
            t=t_val, phase=phase_str,
            l3_mode=current_mode, l2_goal=current_goal, l1_action=action,
            l1_in_goal=in_goal, l2_in_allowed=in_allowed,
            l3_changed=mode_changed,
        ))
        prev_mode = current_mode

    return records


# ═══════════════════════════════════════════════════════════════════════════
# ВИЗУАЛИЗАЦИИ
# ═══════════════════════════════════════════════════════════════════════════

def _save(fig, name: str) -> str:
    path = os.path.join(_OUT_DIR, name)
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    return path


# ── 1. Столбчатая: α по 5 уровням ────────────────────────────────────────
def plot_alpha_by_level(result: AutonomyResult,
                        title: str = "Коэффициент автономности по уровням управления",
                        filename: str = "alpha_levels.png") -> str:
    alphas = [result.alpha_L1, result.alpha_L2, result.alpha_L3,
              result.alpha_L4, result.alpha_L5]

    fig, ax = plt.subplots(figsize=(9, 5), facecolor="#f5f6fa")
    bars = ax.bar(range(5), alphas, color=LEVEL_COLORS, width=0.6,
                  edgecolor="white", linewidth=1.5)

    for bar, a in zip(bars, alphas):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f"α = {a:.2f}", ha="center", fontsize=10, fontweight="bold")

    ax.set_xticks(range(5))
    ax.set_xticklabels(LEVEL_NAMES, fontsize=9)
    ax.set_ylabel("Коэффициент автономности α", fontsize=10)
    ax.set_ylim(0, 1.15)
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.axhline(0.5, color="#bdc3c7", linestyle="--", linewidth=0.8,
               label="α = 0.50 (равновесие)")
    ax.legend(fontsize=8)
    ax.grid(True, axis="y", alpha=0.3)
    ax.set_facecolor("#fafafa")

    # Аннотация: описание
    descriptions = [
        "Следование\nцели L2",
        "Следование\nрежиму L3",
        "Частота\nсмены режима",
        "Самоорганиз.\nгарнизона",
        "Стратегич.\nобучение",
    ]
    for i, desc in enumerate(descriptions):
        ax.text(i, -0.15, desc, ha="center", fontsize=7,
                color="#7f8c8d", style="italic")

    fig.tight_layout()
    return _save(fig, filename)


# ── 2. Тепловая карта: α уровень × фаза ──────────────────────────────────
def plot_alpha_heatmap(result: AutonomyResult,
                       title: str = "Автономность по фазам пожара",
                       filename: str = "alpha_heatmap.png") -> str:
    matrix = np.zeros((3, 5), dtype=float)
    for j, phase in enumerate(PHASE_LABELS):
        matrix[0, j] = result.alpha_L1_by_phase.get(phase, 0)
        matrix[1, j] = result.alpha_L2_by_phase.get(phase, 0)
        matrix[2, j] = result.alpha_L3_by_phase.get(phase, 0)

    fig, ax = plt.subplots(figsize=(8, 4), facecolor="#f5f6fa")
    im = ax.imshow(matrix, aspect="auto", cmap="RdYlGn_r", vmin=0, vmax=1)

    ax.set_xticks(range(5))
    ax.set_xticklabels(PHASE_LABELS, fontsize=10)
    ax.set_yticks(range(3))
    ax.set_yticklabels(["α_L1 (оперативный)", "α_L2 (тактический)",
                        "α_L3 (стратегический)"], fontsize=9)
    ax.set_xlabel("Фаза пожара", fontsize=10)
    ax.set_title(title, fontsize=12, fontweight="bold")

    # Аннотации
    for i in range(3):
        for j in range(5):
            v = matrix[i, j]
            ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                    fontsize=10, fontweight="bold",
                    color="white" if v > 0.5 else "black")

    fig.colorbar(im, ax=ax, shrink=0.8, label="α (0=подчинение, 1=автономия)")
    fig.tight_layout()
    return _save(fig, filename)


# ── 3. Динамика α(t) во времени ───────────────────────────────────────────
def plot_alpha_timeline(result: AutonomyResult,
                        title: str = "Динамика автономности α(t)",
                        filename: str = "alpha_timeline.png") -> str:
    fig, ax = plt.subplots(figsize=(12, 4), facecolor="#f5f6fa")
    ax.set_facecolor("#fafafa")

    if result.alpha_L1_timeline:
        t1 = [t for t, a in result.alpha_L1_timeline]
        a1 = [a for t, a in result.alpha_L1_timeline]
        ax.plot(t1, a1, color=LEVEL_COLORS[0], linewidth=2,
                label=f"α_L1 (оперативный), среднее={result.alpha_L1:.2f}")
        ax.fill_between(t1, a1, alpha=0.15, color=LEVEL_COLORS[0])

    if result.alpha_L2_timeline:
        t2 = [t for t, a in result.alpha_L2_timeline]
        a2 = [a for t, a in result.alpha_L2_timeline]
        ax.plot(t2, a2, color=LEVEL_COLORS[1], linewidth=2,
                label=f"α_L2 (тактический), среднее={result.alpha_L2:.2f}")

    ax.axhline(0.5, color="#bdc3c7", linestyle="--", linewidth=0.8)
    ax.set_xlabel("Время симуляции (мин)", fontsize=10)
    ax.set_ylabel("Коэффициент автономности α", fontsize=10)
    ax.set_ylim(-0.05, 1.05)
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.legend(fontsize=9, loc="upper right")
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    return _save(fig, filename)


# ── 4. Зависимость α_L1 от λ_intrinsic ───────────────────────────────────
def plot_alpha_vs_lambda(
    traces_by_lambda: Dict[float, List[HRLStepRecord]],
    title: str = "Зависимость α_L1 от коэффициента внутренней мотивации λ",
    filename: str = "alpha_vs_lambda.png",
) -> str:
    """Параметрическая кривая: λ → α_L1.

    traces_by_lambda: {0.0: trace, 0.1: trace, ..., 1.0: trace}
    """
    lambdas = sorted(traces_by_lambda.keys())
    alphas = []
    for lam in lambdas:
        trace = traces_by_lambda[lam]
        if trace:
            dev = [1.0 - float(s.l1_in_goal) for s in trace]
            alphas.append(float(np.mean(dev)))
        else:
            alphas.append(0.0)

    fig, ax = plt.subplots(figsize=(8, 5), facecolor="#f5f6fa")
    ax.set_facecolor("#fafafa")

    ax.plot(lambdas, alphas, "o-", color="#c0392b", linewidth=2.5,
            markersize=8, markerfacecolor="white", markeredgecolor="#c0392b",
            markeredgewidth=2, label="α_L1 (измеренный)")

    # Теоретическая кривая: α ≈ 1 / (1 + λ·k), k — коэффициент связности
    lam_th = np.linspace(0, max(lambdas) if lambdas else 1, 100)
    alpha_th = 1.0 / (1.0 + 3.0 * lam_th)  # k=3 — эмпирическая подгонка
    ax.plot(lam_th, alpha_th, "--", color="#3498db", linewidth=1.5,
            label="Теоретическая: α = 1/(1+3λ)", alpha=0.7)

    ax.set_xlabel("Коэффициент внутренней мотивации λ", fontsize=10)
    ax.set_ylabel("Коэффициент автономности α_L1", fontsize=10)
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.set_xlim(-0.05, max(lambdas or [1]) + 0.1)
    ax.set_ylim(-0.05, 1.05)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # Аннотация ключевых точек
    ax.annotate("λ=0: полная автономия\n(L1 игнорирует L2)",
                xy=(0, alphas[0] if alphas else 1),
                xytext=(0.15, 0.85), fontsize=8, color="#7f8c8d",
                arrowprops=dict(arrowstyle="->", color="#bdc3c7"))
    if len(lambdas) > 1:
        ax.annotate("λ→∞: полное подчинение\n(L1 следует цели L2)",
                    xy=(lambdas[-1], alphas[-1]),
                    xytext=(lambdas[-1] - 0.3, 0.15), fontsize=8,
                    color="#7f8c8d",
                    arrowprops=dict(arrowstyle="->", color="#bdc3c7"))

    fig.tight_layout()
    return _save(fig, filename)


# ── 5. Сравнение α между сценариями ───────────────────────────────────────
def plot_alpha_scenarios(
    results: Dict[str, AutonomyResult],
    title: str = "Автономность по сценариям",
    filename: str = "alpha_scenarios.png",
) -> str:
    """Групповая столбчатая: для каждого уровня — столбец на сценарий."""
    n_scen = len(results)
    n_levels = 5

    fig, ax = plt.subplots(figsize=(10, 5), facecolor="#f5f6fa")
    ax.set_facecolor("#fafafa")

    x = np.arange(n_levels)
    width = 0.8 / max(n_scen, 1)
    scen_colors = ["#3498db", "#e74c3c", "#27ae60", "#e67e22", "#9b59b6"]

    for i, (scen_name, res) in enumerate(results.items()):
        alphas = [res.alpha_L1, res.alpha_L2, res.alpha_L3,
                  res.alpha_L4, res.alpha_L5]
        offset = (i - n_scen / 2 + 0.5) * width
        bars = ax.bar(x + offset, alphas, width * 0.9,
                      label=scen_name, color=scen_colors[i % len(scen_colors)],
                      edgecolor="white", linewidth=0.8)
        for bar, a in zip(bars, alphas):
            if a > 0.02:
                ax.text(bar.get_x() + bar.get_width()/2,
                        bar.get_height() + 0.01,
                        f"{a:.2f}", ha="center", fontsize=7, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(LEVEL_NAMES, fontsize=9)
    ax.set_ylabel("Коэффициент автономности α", fontsize=10)
    ax.set_ylim(0, 1.15)
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(True, axis="y", alpha=0.3)

    fig.tight_layout()
    return _save(fig, filename)


# ═══════════════════════════════════════════════════════════════════════════
# ГЕНЕРАЦИЯ ДЕМО-ВИЗУАЛИЗАЦИЙ
# ═══════════════════════════════════════════════════════════════════════════
def generate_demo(seed: int = 42) -> Dict[str, str]:
    """Создать все 5 визуализаций с синтетическими данными."""
    rng = np.random.RandomState(seed)
    results = {}

    # ── Синтетическая трасса: сценарий А (ранг 4, сложный) ────────────────
    def _make_trace(n_steps, deviation_rate_l1, deviation_rate_l2,
                    change_rate_l3, scenario_name):
        trace = []
        phases = (["S1"]*3 + ["S2"]*8 + ["S3"]*30 + ["S4"]*10 + ["S5"]*4)
        if n_steps > len(phases):
            phases = phases * (n_steps // len(phases) + 1)
        mode = 0
        goal = 1  # DEFENSE
        for i in range(n_steps):
            phase = phases[i % len(phases)]
            # L3: иногда меняет режим
            l3_changed = rng.random() < change_rate_l3
            if l3_changed:
                mode = rng.choice([0, 1, 2])
            # L2: цель
            allowed = L3_GOAL_ALLOW.get(mode, [0, 1, 2, 3, 4])
            if rng.random() < deviation_rate_l2:
                goal = rng.choice([g for g in range(5) if g not in allowed] or allowed)
                l2_ok = False
            else:
                goal = rng.choice(allowed)
                l2_ok = True
            # L1: действие
            goal_acts = GOAL_ACTION_MAP.get(goal, [12])
            if rng.random() < deviation_rate_l1:
                action = rng.choice([a for a in range(N_ACTIONS) if a not in goal_acts])
                l1_ok = False
            else:
                action = rng.choice(goal_acts)
                l1_ok = True

            trace.append(HRLStepRecord(
                t=i * 5, phase=phase,
                l3_mode=mode, l2_goal=goal, l1_action=action,
                l1_in_goal=l1_ok, l2_in_allowed=l2_ok,
                l3_changed=l3_changed,
                env_reward=rng.uniform(-0.5, 0.5),
                intrinsic_reward=0.5 if l1_ok else -0.2,
            ))
        return trace

    trace_a = _make_trace(55, deviation_rate_l1=0.35, deviation_rate_l2=0.08,
                          change_rate_l3=0.06, scenario_name="Сценарий А")
    trace_b = _make_trace(30, deviation_rate_l1=0.20, deviation_rate_l2=0.03,
                          change_rate_l3=0.10, scenario_name="Сценарий Б")

    res_a = compute_autonomy(trace_a, scenario="Сценарий А (ранг 4)",
                             garrison_readiness=0.65, mortality_trend=-0.03)
    res_b = compute_autonomy(trace_b, scenario="Сценарий Б (ранг 2)",
                             garrison_readiness=0.85, mortality_trend=-0.01)

    # ── 1. α по уровням ──────────────────────────────────────────────────
    results["levels"] = plot_alpha_by_level(
        res_a, title="Автономность по уровням (сценарий А, ранг №4)")

    # ── 2. Тепловая карта α(уровень × фаза) ──────────────────────────────
    results["heatmap"] = plot_alpha_heatmap(
        res_a, title="Автономность по фазам пожара (сценарий А)")

    # ── 3. Динамика α(t) ─────────────────────────────────────────────────
    results["timeline"] = plot_alpha_timeline(
        res_a, title="Динамика автономности во времени (сценарий А)")

    # ── 4. α(λ) — зависимость от λ_intrinsic ─────────────────────────────
    traces_by_lambda = {}
    for lam in [0.0, 0.05, 0.10, 0.20, 0.30, 0.50, 0.70, 1.0]:
        # При λ=0 — максимальное отклонение, при λ→∞ — минимальное
        dev_rate = max(0.02, 0.60 - 0.55 * lam)
        traces_by_lambda[lam] = _make_trace(
            50, deviation_rate_l1=dev_rate,
            deviation_rate_l2=0.05, change_rate_l3=0.05,
            scenario_name=f"λ={lam}")
    results["lambda"] = plot_alpha_vs_lambda(
        traces_by_lambda,
        title="Влияние λ (внутренняя мотивация) на автономность L1")

    # ── 5. Сравнение сценариев ───────────────────────────────────────────
    results["scenarios"] = plot_alpha_scenarios(
        {"Сценарий А\n(ранг №4, 81 ч)": res_a,
         "Сценарий Б\n(ранг №2, 5 ч)": res_b},
        title="Сравнение автономности между сценариями")

    return results


if __name__ == "__main__":
    paths = generate_demo()
    print("Визуализации автономности сохранены:")
    for name, path in paths.items():
        print(f"  {name}: {path}")
