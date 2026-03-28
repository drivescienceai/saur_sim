"""
visualizations.py — Визуализация логики агентов САУР-ПСП.
═══════════════════════════════════════════════════════════════════════════════
5 типов диаграмм для сравнения агентов и режимов:

1. Тепловая карта политик     (фаза × действие, сравнение агентов)
2. Поток решений по времени   (действия на временно́й оси)
3. Сравнение весов IRL        (восстановленные vs ручные)
4. Санкей-диаграмма HRL       (режим → цель → действие)
5. Радарная диаграмма          (профиль агента по 5 метрикам)
═══════════════════════════════════════════════════════════════════════════════
"""
from __future__ import annotations

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.figure import Figure
from typing import List, Dict, Optional, Tuple

try:
    from .rl_agent import (STATE_SIZE, N_ACTIONS, ACTION_NAMES, ACTION_COST,
                           W_L1, W_CASUALTY, W_COST, W_L7)
except ImportError:
    from rl_agent import (STATE_SIZE, N_ACTIONS, ACTION_NAMES, ACTION_COST,
                          W_L1, W_CASUALTY, W_COST, W_L7)

# ── Константы визуализации ───────────────────────────────────────────────────
PHASE_LABELS = ["S1", "S2", "S3", "S4", "S5"]
ACTION_CODES = ["С1","С2","С3","С4","С5","Т1","Т2","Т3","Т4","О1","О2","О3","О4","О5","О6"]
ACTION_SHORT = [
    "Спасение","Защита","Локализ.","Пен.атака","Вскипание",
    "Создать БУ","Перегруп.","Доп.силы","ПНС","Охлажд.горящ.",
    "Охлажд.сосед.","Пена","Разведка","Розлив","Отход",
]
LEVEL_COLORS = {
    "стратег.": "#c0392b", "тактич.": "#e67e22", "оперативн.": "#2980b9",
}
ACTION_LEVEL = (
    ["стратег."]*5 + ["тактич."]*4 + ["оперативн."]*6
)
PHASE_COLORS_LIST = ["#e74c3c","#e67e22","#f39c12","#27ae60","#2980b9"]

_OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "figures")
os.makedirs(_OUT_DIR, exist_ok=True)


def _save(fig, name: str) -> str:
    path = os.path.join(_OUT_DIR, name)
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    return path


# ═══════════════════════════════════════════════════════════════════════════
# 1. ТЕПЛОВАЯ КАРТА ПОЛИТИК (фаза × действие)
# ═══════════════════════════════════════════════════════════════════════════
def policy_heatmap(
    policies: Dict[str, np.ndarray],
    title: str = "Сравнение политик агентов",
    filename: str = "policy_heatmap.png",
) -> str:
    """Тепловая карта: строки = фазы S1-S5, столбцы = 15 действий.

    policies: {"Имитация": matrix_5x15, "Табличный": matrix_5x15, ...}
              каждая матрица — частота выбора действия в фазе (0..1).
    """
    n_agents = len(policies)
    fig, axes = plt.subplots(1, n_agents, figsize=(5 * n_agents, 4),
                             facecolor="#f5f6fa")
    if n_agents == 1:
        axes = [axes]

    for ax, (agent_name, matrix) in zip(axes, policies.items()):
        # Нормализация по строкам
        row_sums = matrix.sum(axis=1, keepdims=True)
        norm = np.divide(matrix, row_sums, where=row_sums > 0,
                         out=np.zeros_like(matrix))

        im = ax.imshow(norm, aspect="auto", cmap="YlOrRd",
                       vmin=0, vmax=1, origin="upper")
        ax.set_xticks(range(N_ACTIONS))
        ax.set_xticklabels(ACTION_CODES, fontsize=7, rotation=45, ha="right")
        ax.set_yticks(range(5))
        ax.set_yticklabels(PHASE_LABELS, fontsize=9)
        ax.set_title(agent_name, fontsize=10, fontweight="bold", pad=8)
        ax.set_xlabel("Действие", fontsize=8)
        if ax == axes[0]:
            ax.set_ylabel("Фаза пожара", fontsize=8)

        # Аннотации — числа в ячейках
        for i in range(5):
            for j in range(N_ACTIONS):
                v = norm[i, j]
                if v > 0.05:
                    ax.text(j, i, f"{v:.0%}", ha="center", va="center",
                            fontsize=6, color="white" if v > 0.5 else "black")

    fig.suptitle(title, fontsize=12, fontweight="bold", y=1.02)
    fig.colorbar(im, ax=axes, shrink=0.6, label="Частота выбора")
    fig.tight_layout()
    return _save(fig, filename)


def policy_from_q_table(Q: np.ndarray) -> np.ndarray:
    """Извлечь матрицу политики 5×15 из Q-таблицы (315×15).

    Усредняет по всем состояниям с одинаковой фазой.
    """
    matrix = np.zeros((5, N_ACTIONS), dtype=float)
    counts = np.zeros(5, dtype=float)
    for s in range(min(STATE_SIZE, Q.shape[0])):
        phase = s // 45  # обратное кодирование
        if 1 <= phase <= 5:
            best_a = int(np.argmax(Q[s]))
            matrix[phase - 1, best_a] += 1
            counts[phase - 1] += 1
    for p in range(5):
        if counts[p] > 0:
            matrix[p] /= counts[p]
    return matrix


def policy_from_expert_rules() -> np.ndarray:
    """Матрица политики из экспертных правил (режим «Имитация»)."""
    # Приоритеты из tank_fire_sim.py: _expert_action()
    priority = {
        0: [12, 7, 8, 0, 2],
        1: [9, 10, 5, 8, 1, 2, 12],
        2: [7, 8, 9, 10, 6, 11, 12],
        3: [11, 3, 12, 4, 6, 0],
        4: [12, 13, 10, 14],
    }
    matrix = np.zeros((5, N_ACTIONS), dtype=float)
    for phase_idx, actions in priority.items():
        if actions:
            matrix[phase_idx, actions[0]] = 1.0  # всегда выбирает первый
    return matrix


# ═══════════════════════════════════════════════════════════════════════════
# 2. ПОТОК РЕШЕНИЙ ПО ВРЕМЕНИ
# ═══════════════════════════════════════════════════════════════════════════
def decision_flow(
    traces: Dict[str, List[Tuple[int, int, str]]],
    title: str = "Поток решений по времени",
    filename: str = "decision_flow.png",
) -> str:
    """Горизонтальная диаграмма: время → действия.

    traces: {"Агент1": [(t_min, action_idx, phase), ...], ...}
    """
    n_agents = len(traces)
    fig, axes = plt.subplots(n_agents, 1, figsize=(14, 2.5 * n_agents),
                             facecolor="#f5f6fa", sharex=True)
    if n_agents == 1:
        axes = [axes]

    for ax, (agent_name, trace) in zip(axes, traces.items()):
        times = [t for t, a, p in trace]
        actions = [a for t, a, p in trace]
        phases = [p for t, a, p in trace]
        colors = [LEVEL_COLORS.get(ACTION_LEVEL[a], "#888") for a in actions]

        ax.scatter(times, actions, c=colors, s=20, alpha=0.8, edgecolors="white",
                   linewidth=0.3)

        # Соединить линией
        ax.plot(times, actions, color="#bdc3c7", linewidth=0.5, alpha=0.5, zorder=0)

        # Фоновые полосы для фаз
        phase_map = {"S1": 0, "S2": 1, "S3": 2, "S4": 3, "S5": 4}
        prev_t, prev_phase = 0, ""
        for t, a, p in trace:
            if p != prev_phase and prev_phase:
                pi = phase_map.get(prev_phase, -1)
                if pi >= 0:
                    ax.axvspan(prev_t, t, alpha=0.08,
                               color=PHASE_COLORS_LIST[pi])
                prev_t = t
            prev_phase = p

        ax.set_yticks(range(N_ACTIONS))
        ax.set_yticklabels(ACTION_CODES, fontsize=7)
        ax.set_ylabel(agent_name, fontsize=9, fontweight="bold")
        ax.set_ylim(-0.5, N_ACTIONS - 0.5)
        ax.grid(True, axis="x", alpha=0.3)

    axes[-1].set_xlabel("Время симуляции (мин)", fontsize=9)

    # Легенда уровней
    patches = [mpatches.Patch(color=c, label=l)
               for l, c in LEVEL_COLORS.items()]
    fig.legend(handles=patches, loc="upper right", fontsize=8,
               title="Уровень", title_fontsize=8)

    fig.suptitle(title, fontsize=12, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 0.92, 0.95])
    return _save(fig, filename)


# ═══════════════════════════════════════════════════════════════════════════
# 3. СРАВНЕНИЕ ВЕСОВ IRL VS РУЧНЫЕ
# ═══════════════════════════════════════════════════════════════════════════
def irl_weights_comparison(
    recovered_weights: np.ndarray,
    feature_names: Optional[List[str]] = None,
    title: str = "Восстановленные веса vs ручные",
    filename: str = "irl_weights.png",
) -> str:
    """Столбчатая диаграмма: 4 пары столбцов (IRL vs ручные)."""
    if feature_names is None:
        feature_names = [
            "Тяжесть\nпожара",
            "Потери\nЛС",
            "Стоимость\nдействия",
            "Качество\nL7",
        ]

    manual = np.array([-W_L1, -W_CASUALTY, -W_COST, +W_L7])
    n = len(feature_names)

    fig, ax = plt.subplots(figsize=(8, 5), facecolor="#f5f6fa")
    x = np.arange(n)
    w = 0.35

    bars1 = ax.bar(x - w/2, manual, w, label="Ручная формула",
                   color="#3498db", alpha=0.85, edgecolor="white")
    bars2 = ax.bar(x + w/2, recovered_weights[:n], w,
                   label="Восстановленные (IRL)",
                   color="#e74c3c", alpha=0.85, edgecolor="white")

    # Значения над столбцами
    for bars in [bars1, bars2]:
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, h,
                    f"{h:+.2f}", ha="center", va="bottom" if h >= 0 else "top",
                    fontsize=8, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(feature_names, fontsize=9)
    ax.set_ylabel("Вес компоненты", fontsize=10)
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.legend(fontsize=9)
    ax.axhline(0, color="black", linewidth=0.5)
    ax.grid(True, axis="y", alpha=0.3)
    ax.set_facecolor("#fafafa")

    fig.tight_layout()
    return _save(fig, filename)


# ═══════════════════════════════════════════════════════════════════════════
# 4. САНКЕЙ-ДИАГРАММА (упрощённая) ДЛЯ HRL
# ═══════════════════════════════════════════════════════════════════════════
def hrl_sankey(
    flows: Dict[str, Dict[str, Dict[str, int]]],
    title: str = "Иерархия решений (режим → цель → действие)",
    filename: str = "hrl_sankey.png",
) -> str:
    """Упрощённая Санкей-диаграмма: 3 колонки потоков.

    flows: {mode_name: {goal_name: {action_code: count, ...}, ...}, ...}
    Пример: {"Собств. силы": {"Штурм": {"О3": 15, "С4": 8}, ...}, ...}
    """
    fig, ax = plt.subplots(figsize=(14, 7), facecolor="#f5f6fa")
    ax.set_xlim(0, 10)
    ax.set_facecolor("#fafafa")

    # Колонки
    col_x = [1.5, 5.0, 8.5]
    col_labels = ["Режим L3", "Цель L2", "Действие L1"]
    for x, label in zip(col_x, col_labels):
        ax.text(x, -0.3, label, ha="center", fontsize=10,
                fontweight="bold", color="#2c3e50")

    mode_colors = {"Собственные силы": "#27ae60",
                   "Региональный": "#e67e22",
                   "Федеральный": "#c0392b"}
    goal_colors = {"Штурм": "#e74c3c", "Оборона": "#3498db",
                   "Наращивание": "#e67e22", "Разведка": "#9b59b6",
                   "Эвакуация": "#7f8c8d"}

    # Вычислить позиции узлов
    modes = list(flows.keys())
    all_goals = []
    all_actions = []
    for m in flows.values():
        for g in m:
            if g not in all_goals:
                all_goals.append(g)
            for a in m[g]:
                if a not in all_actions:
                    all_actions.append(a)

    n_m, n_g, n_a = len(modes), len(all_goals), len(all_actions)
    mode_y = {m: i * (8 / max(n_m, 1)) + 1 for i, m in enumerate(modes)}
    goal_y = {g: i * (8 / max(n_g, 1)) + 0.5 for i, g in enumerate(all_goals)}
    action_y = {a: i * (8 / max(n_a, 1)) + 0.3 for i, a in enumerate(all_actions)}

    # Нарисовать узлы
    for m, y in mode_y.items():
        c = mode_colors.get(m, "#888")
        ax.add_patch(plt.Rectangle((col_x[0]-0.4, y-0.2), 0.8, 0.4,
                     facecolor=c, alpha=0.8, edgecolor="white", linewidth=1.5))
        ax.text(col_x[0], y, m, ha="center", va="center",
                fontsize=7, fontweight="bold", color="white")

    for g, y in goal_y.items():
        c = goal_colors.get(g, "#888")
        ax.add_patch(plt.Rectangle((col_x[1]-0.4, y-0.2), 0.8, 0.4,
                     facecolor=c, alpha=0.8, edgecolor="white", linewidth=1.5))
        ax.text(col_x[1], y, g, ha="center", va="center",
                fontsize=7, fontweight="bold", color="white")

    for a, y in action_y.items():
        idx = ACTION_CODES.index(a) if a in ACTION_CODES else -1
        c = LEVEL_COLORS.get(ACTION_LEVEL[idx], "#888") if idx >= 0 else "#888"
        ax.add_patch(plt.Rectangle((col_x[2]-0.4, y-0.15), 0.8, 0.3,
                     facecolor=c, alpha=0.7, edgecolor="white", linewidth=1))
        ax.text(col_x[2], y, a, ha="center", va="center",
                fontsize=7, fontweight="bold", color="white")

    # Нарисовать потоки
    total = sum(cnt for m in flows.values() for g in m.values()
                for cnt in g.values()) or 1

    for mode_name, goals in flows.items():
        my = mode_y[mode_name]
        mc = mode_colors.get(mode_name, "#888")
        for goal_name, actions in goals.items():
            gy = goal_y[goal_name]
            # Поток mode → goal
            flow_mg = sum(actions.values())
            lw = max(0.5, 8 * flow_mg / total)
            ax.annotate("", xy=(col_x[1]-0.4, gy),
                        xytext=(col_x[0]+0.4, my),
                        arrowprops=dict(arrowstyle="-",
                                        color=mc, alpha=0.4, lw=lw))

            gc = goal_colors.get(goal_name, "#888")
            for action_code, count in actions.items():
                ay = action_y.get(action_code, 4)
                lw_a = max(0.3, 6 * count / total)
                ax.annotate("", xy=(col_x[2]-0.4, ay),
                            xytext=(col_x[1]+0.4, gy),
                            arrowprops=dict(arrowstyle="-",
                                            color=gc, alpha=0.35, lw=lw_a))

    ax.set_ylim(-1, 9)
    ax.set_title(title, fontsize=12, fontweight="bold", pad=15)
    ax.axis("off")
    fig.tight_layout()
    return _save(fig, filename)


def hrl_sankey_from_trace(
    trace: List[Tuple[int, int, str, str, str]],
    **kwargs,
) -> str:
    """Построить Санкей из трассы иерархического агента.

    trace: [(t, action_idx, phase, l3_mode, l2_goal), ...]
    """
    flows: Dict[str, Dict[str, Dict[str, int]]] = {}
    for t, a, phase, mode, goal in trace:
        code = ACTION_CODES[a] if 0 <= a < N_ACTIONS else "?"
        flows.setdefault(mode, {}).setdefault(goal, {})
        flows[mode][goal][code] = flows[mode][goal].get(code, 0) + 1
    return hrl_sankey(flows, **kwargs)


# ═══════════════════════════════════════════════════════════════════════════
# 5. РАДАРНАЯ ДИАГРАММА (профиль агента)
# ═══════════════════════════════════════════════════════════════════════════
def radar_chart(
    profiles: Dict[str, Dict[str, float]],
    title: str = "Профиль агентов",
    filename: str = "radar_chart.png",
) -> str:
    """Радарная диаграмма (паук): каждый агент — свой многоугольник.

    profiles: {
        "Табличный": {"Успешность": 0.75, "Скорость": 0.60, ...},
        "Иерархический": {"Успешность": 0.90, "Скорость": 0.80, ...},
    }
    Все значения нормализованы 0..1.
    """
    if not profiles:
        fig, ax = plt.subplots(figsize=(6, 6), facecolor="#f5f6fa")
        ax.text(0.5, 0.5, "Нет данных", ha="center", transform=ax.transAxes)
        return _save(fig, filename)

    first_profile = next(iter(profiles.values()))
    categories = list(first_profile.keys())
    n_cats = len(categories)

    angles = np.linspace(0, 2 * np.pi, n_cats, endpoint=False).tolist()
    angles += angles[:1]  # замкнуть

    fig, ax = plt.subplots(figsize=(7, 7), subplot_kw=dict(polar=True),
                           facecolor="#f5f6fa")
    ax.set_facecolor("#fafafa")

    agent_colors = ["#e74c3c", "#3498db", "#27ae60", "#e67e22",
                    "#9b59b6", "#1abc9c", "#f39c12"]

    for i, (agent_name, metrics) in enumerate(profiles.items()):
        values = [metrics.get(c, 0) for c in categories]
        values += values[:1]
        color = agent_colors[i % len(agent_colors)]

        ax.plot(angles, values, "o-", linewidth=2, label=agent_name,
                color=color, markersize=5)
        ax.fill(angles, values, alpha=0.15, color=color)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=9)
    ax.set_ylim(0, 1)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(["0.2", "0.4", "0.6", "0.8", "1.0"], fontsize=7)
    ax.set_title(title, fontsize=12, fontweight="bold", pad=20)
    ax.legend(loc="upper right", bbox_to_anchor=(1.25, 1.1), fontsize=9)

    fig.tight_layout()
    return _save(fig, filename)


# ═══════════════════════════════════════════════════════════════════════════
# ГЕНЕРАЦИЯ ВСЕХ ВИЗУАЛИЗАЦИЙ (демо с синтетическими данными)
# ═══════════════════════════════════════════════════════════════════════════
def generate_all_demo(seed: int = 42) -> Dict[str, str]:
    """Создать все 5 визуализаций с демонстрационными данными.

    Возвращает: {"heatmap": path, "flow": path, "irl": path,
                 "sankey": path, "radar": path}
    """
    rng = np.random.RandomState(seed)
    results = {}

    # ── 1. Тепловая карта политик ─────────────────────────────────────────
    expert = policy_from_expert_rules()

    # Синтетический Q-агент — предпочитает О1/О3 в тяжёлых фазах
    q_policy = np.zeros((5, N_ACTIONS))
    for p in range(5):
        probs = rng.dirichlet(np.ones(N_ACTIONS) * 0.3)
        if p >= 2:
            probs[9] += 0.3   # О1
            probs[11] += 0.3  # О3
        probs /= probs.sum()
        q_policy[p] = probs

    # Синтетический HRL агент — более сфокусирован
    hrl_policy = np.zeros((5, N_ACTIONS))
    hrl_focus = {0: [12, 7], 1: [9, 5, 8], 2: [7, 9, 11],
                 3: [11, 3], 4: [12, 13]}
    for p, acts in hrl_focus.items():
        for a in acts:
            hrl_policy[p, a] = 1.0 / len(acts)

    results["heatmap"] = policy_heatmap(
        {"Имитация (эксперт)": expert,
         "Табличный агент": q_policy,
         "Иерархический агент": hrl_policy},
        title="Сравнение политик: фаза × действие")

    # ── 2. Поток решений по времени ───────────────────────────────────────
    phases_seq = (["S1"]*3 + ["S2"]*5 + ["S3"]*20 + ["S4"]*5 + ["S5"]*2)

    def _gen_trace(name, bias_actions):
        trace = []
        for i, phase in enumerate(phases_seq):
            t = i * 5
            if rng.random() < 0.7:
                a = rng.choice(bias_actions.get(phase, [12]))
            else:
                a = rng.randint(0, N_ACTIONS)
            trace.append((t, a, phase))
        return trace

    expert_trace = _gen_trace("expert", {
        "S1": [12, 7], "S2": [9, 10, 5], "S3": [7, 9, 11],
        "S4": [11, 3], "S5": [12, 13]})
    q_trace = _gen_trace("q", {
        "S1": [12, 0], "S2": [9, 1], "S3": [9, 11, 7],
        "S4": [11, 11, 3], "S5": [12, 10]})
    hrl_trace = _gen_trace("hrl", {
        "S1": [12, 7, 8], "S2": [9, 5, 8], "S3": [7, 8, 9, 11],
        "S4": [11, 3, 4], "S5": [12, 13]})

    results["flow"] = decision_flow(
        {"Имитация (эксперт)": expert_trace,
         "Табличный агент": q_trace,
         "Иерархический агент": hrl_trace},
        title="Поток решений по времени (сценарий А)")

    # ── 3. Сравнение весов IRL ────────────────────────────────────────────
    recovered = np.array([-0.42, -0.18, -0.03, +0.37])
    results["irl"] = irl_weights_comparison(
        recovered,
        title="Восстановленные приоритеты vs ручная формула")

    # ── 4. Санкей-диаграмма HRL ───────────────────────────────────────────
    sankey_flows = {
        "Собственные силы": {
            "Штурм": {"С4": 12, "О3": 18},
            "Оборона": {"С2": 8, "О1": 15, "О2": 10},
            "Наращивание": {"Т1": 5, "Т3": 12, "Т4": 8},
        },
        "Региональный": {
            "Штурм": {"О3": 6, "С4": 4},
            "Наращивание": {"Т3": 15, "Т2": 5},
            "Разведка": {"О4": 8},
        },
        "Федеральный": {
            "Штурм": {"О3": 3},
            "Эвакуация": {"С1": 4, "О6": 2},
        },
    }
    results["sankey"] = hrl_sankey(
        sankey_flows,
        title="Иерархия решений: режим L3 → цель L2 → действие L1")

    # ── 5. Радарная диаграмма ─────────────────────────────────────────────
    radar_profiles = {
        "Имитация": {
            "Успешность": 0.65, "Скорость\nликвидации": 0.45,
            "Экономия\nресурсов": 0.80, "Безопасность\nЛС": 0.70,
            "Эффективность\nпены": 0.30,
        },
        "Табличный агент": {
            "Успешность": 0.78, "Скорость\nликвидации": 0.65,
            "Экономия\nресурсов": 0.55, "Безопасность\nЛС": 0.72,
            "Эффективность\nпены": 0.60,
        },
        "Иерархический агент": {
            "Успешность": 0.92, "Скорость\nликвидации": 0.82,
            "Экономия\nресурсов": 0.60, "Безопасность\nЛС": 0.88,
            "Эффективность\nпены": 0.75,
        },
    }
    results["radar"] = radar_chart(
        radar_profiles,
        title="Профиль агентов по ключевым метрикам")

    return results


if __name__ == "__main__":
    paths = generate_all_demo()
    print("Визуализации сохранены:")
    for name, path in paths.items():
        print(f"  {name}: {path}")
