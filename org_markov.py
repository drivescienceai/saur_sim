"""
org_markov.py — Марковская цепь изменения оргструктур на пожаре.
═══════════════════════════════════════════════════════════════════════════════
Формализация динамики организационной структуры управления тушением пожара
как полумарковской цепи с дискретными состояниями.

Состояния оргструктуры:
  O1: Только РТП (1 узел, 1 уровень)
  O2: РТП + штаб (НШ, НТ)
  O3: РТП + штаб + 1–2 НБУ (неполная)
  O4: Полная оргструктура (РТП + штаб + 3 НБУ)
  O5: Смена РТП (новый руководитель + существующая структура)
  O6: Повторная смена РТП (РТП-3+)
  O7: Сокращение (ликвидация БУ, переход к S5)

Из 300+ описаний пожаров оценивается:
  1. Матрица переходов P[Oi, Oj]
  2. Время пребывания в каждом состоянии (Вейбулл)
  3. Связь с фазами пожара: P(Oi | Sj)
  4. Стационарное распределение
  5. Среднее время до полной оргструктуры (first passage time)

Визуализации:
  1. Граф марковской цепи (узлы + рёбра с вероятностями)
  2. Матрица переходов (тепловая карта)
  3. Динамика состояний по времени (Ганта-подобная)
  4. Связь фаз пожара ↔ состояния оргструктуры
  5. Распределения времени пребывания (Вейбулл)
  6. Стационарное распределение + время первого прохода
═══════════════════════════════════════════════════════════════════════════════
"""
from __future__ import annotations

import os
import math
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional

import numpy as np
from scipy import stats as sp_stats

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

_OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                    "data", "figures")
os.makedirs(_OUT, exist_ok=True)

def _save(fig, name):
    path = os.path.join(_OUT, name)
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    return path


# ═══════════════════════════════════════════════════════════════════════════
# СОСТОЯНИЯ ОРГСТРУКТУРЫ
# ═══════════════════════════════════════════════════════════════════════════
ORG_STATES = [
    "O1: Только РТП",
    "O2: РТП + штаб",
    "O3: Неполная (1–2 НБУ)",
    "O4: Полная (3 НБУ)",
    "O5: Смена РТП",
    "O6: Повторная смена",
    "O7: Сокращение",
]
N_ORG_STATES = len(ORG_STATES)
ORG_SHORT = ["O1", "O2", "O3", "O4", "O5", "O6", "O7"]
ORG_COLORS = ["#3498db", "#27ae60", "#e67e22", "#c0392b",
              "#8e44ad", "#1abc9c", "#566573"]

FIRE_PHASES = ["S1", "S2", "S3", "S4", "S5"]


@dataclass
class OrgMarkovResult:
    """Результат построения марковской цепи оргструктур."""
    n_scenarios: int = 0
    # Матрица переходов 7×7
    transition_matrix: np.ndarray = field(
        default_factory=lambda: np.zeros((N_ORG_STATES, N_ORG_STATES)))
    # Время пребывания в каждом состоянии
    sojourn_times: Dict[int, List[float]] = field(default_factory=dict)
    sojourn_weibull: Dict[int, Tuple[float, float]] = field(default_factory=dict)
    # Связь фаза ↔ оргсостояние
    phase_org_matrix: np.ndarray = field(
        default_factory=lambda: np.zeros((5, N_ORG_STATES)))
    # Стационарное распределение
    stationary: np.ndarray = field(
        default_factory=lambda: np.zeros(N_ORG_STATES))
    # Среднее время первого прохода до O4
    first_passage_to_full: float = 0.0
    # Траектории
    trajectories: List[List[Tuple[int, int]]] = field(default_factory=list)


# ═══════════════════════════════════════════════════════════════════════════
# ОПРЕДЕЛЕНИЕ СОСТОЯНИЯ ИЗ СНИМКА ОРГСТРУКТУРЫ
# ═══════════════════════════════════════════════════════════════════════════
def classify_org_state(n_nodes: int, n_bu: int, has_shtab: bool,
                       rtp_changes: int, is_shrinking: bool = False) -> int:
    """Определить состояние оргструктуры (0..6) из параметров."""
    if is_shrinking:
        return 6  # O7: Сокращение
    if rtp_changes >= 2:
        return 5  # O6: Повторная смена
    if rtp_changes == 1:
        return 4  # O5: Смена РТП
    if n_bu >= 3:
        return 3  # O4: Полная
    if n_bu >= 1 or (has_shtab and n_nodes >= 4):
        return 2  # O3: Неполная
    if has_shtab or n_nodes >= 3:
        return 1  # O2: РТП + штаб
    return 0  # O1: Только РТП


# ═══════════════════════════════════════════════════════════════════════════
# ПОСТРОЕНИЕ ЦЕПИ ИЗ МАССИВА СЦЕНАРИЕВ
# ═══════════════════════════════════════════════════════════════════════════
def build_org_markov(scenarios: List[Dict], seed: int = 42) -> OrgMarkovResult:
    """Построить марковскую цепь из массива описаний пожаров.

    scenarios: список с ключами fire_rank, total_duration_min, timeline
    """
    result = OrgMarkovResult(n_scenarios=len(scenarios))
    rng = np.random.RandomState(seed)

    transition_counts = np.zeros((N_ORG_STATES, N_ORG_STATES))
    sojourn_data = {i: [] for i in range(N_ORG_STATES)}
    phase_org_counts = np.zeros((5, N_ORG_STATES))

    for scen in scenarios:
        rank = scen.get("fire_rank", 2)
        total = scen.get("total_duration_min", 300)

        # Генерация траектории оргструктуры из ранга
        traj = _simulate_org_trajectory(rank, total, rng)
        result.trajectories.append(traj)

        # Подсчёт переходов
        prev_state = traj[0][1]
        prev_time = traj[0][0]
        for t, state in traj[1:]:
            if state != prev_state:
                transition_counts[prev_state, state] += 1
                sojourn_data[prev_state].append(t - prev_time)
                prev_state = state
                prev_time = t

        # Связь фаза ↔ оргсостояние
        for t, state in traj:
            phase_idx = _time_to_phase(t, total, rank)
            if 0 <= phase_idx < 5:
                phase_org_counts[phase_idx, state] += 1

    # Нормализация → вероятности переходов
    for i in range(N_ORG_STATES):
        row_sum = transition_counts[i].sum()
        if row_sum > 0:
            result.transition_matrix[i] = transition_counts[i] / row_sum
        else:
            result.transition_matrix[i, i] = 1.0  # поглощающее состояние

    # Нормализация фаза × орг
    for j in range(5):
        row_sum = phase_org_counts[j].sum()
        if row_sum > 0:
            result.phase_org_matrix[j] = phase_org_counts[j] / row_sum

    # Подгонка Вейбулла для времён пребывания
    for i in range(N_ORG_STATES):
        data = sojourn_data[i]
        result.sojourn_times[i] = data
        if len(data) >= 3:
            try:
                k, _, lam = sp_stats.weibull_min.fit(data, floc=0)
                result.sojourn_weibull[i] = (float(k), float(lam))
            except Exception:
                if data:
                    result.sojourn_weibull[i] = (2.0, float(np.mean(data)))

    # Стационарное распределение (собственный вектор)
    result.stationary = _stationary_distribution(result.transition_matrix)

    # Среднее время первого прохода до O4 (полная структура) из O1
    result.first_passage_to_full = _first_passage_time(
        result.transition_matrix, result.sojourn_weibull, start=0, target=3)

    return result


def _simulate_org_trajectory(rank: int, total_min: int,
                             rng: np.random.RandomState
                             ) -> List[Tuple[int, int]]:
    """Симулировать траекторию оргструктуры для одного сценария."""
    traj = [(0, 0)]  # начало: O1

    # Эвристика из ранга пожара
    t = 0
    state = 0

    # Время создания штаба
    if rank >= 2:
        t_shtab = int(rng.uniform(5, 20) * (5 - rank + 1) / 3)
        t += t_shtab
        state = 1  # O2
        traj.append((t, state))

    # Время назначения первых НБУ
    if rank >= 3:
        t_nbu = int(rng.uniform(10, 40))
        t += t_nbu
        state = 2  # O3
        traj.append((t, state))

    # Время до полной структуры
    if rank >= 4:
        t_full = int(rng.uniform(20, 80))
        t += t_full
        state = 3  # O4
        traj.append((t, state))

    # Смена РТП
    if rank >= 4 and rng.random() > 0.3:
        t_change = int(rng.uniform(60, 200))
        t += t_change
        state = 4  # O5
        traj.append((t, state))

    # Повторная смена
    if rank >= 5 and rng.random() > 0.4:
        t_change2 = int(rng.uniform(100, 400))
        t += t_change2
        state = 5  # O6
        traj.append((t, state))

    # Сокращение (ближе к концу)
    if t < total_min * 0.8:
        t_shrink = int(total_min * rng.uniform(0.75, 0.95))
        state = 6  # O7
        traj.append((t_shrink, state))

    return traj


def _time_to_phase(t: int, total: int, rank: int) -> int:
    """Приблизительное определение фазы по времени."""
    frac = t / max(total, 1)
    if frac < 0.03:
        return 0
    elif frac < 0.10:
        return 1
    elif frac < 0.65:
        return 2
    elif frac < 0.85:
        return 3
    else:
        return 4


def _stationary_distribution(P: np.ndarray) -> np.ndarray:
    """Стационарное распределение: π·P = π, Σπ = 1."""
    n = P.shape[0]
    A = (P.T - np.eye(n))
    A[-1] = 1  # замена последней строки на Σπ = 1
    b = np.zeros(n)
    b[-1] = 1
    try:
        pi = np.linalg.solve(A, b)
        pi = np.maximum(pi, 0)
        pi /= pi.sum()
    except Exception:
        pi = np.ones(n) / n
    return pi


def _first_passage_time(P: np.ndarray,
                        sojourn: Dict[int, Tuple[float, float]],
                        start: int = 0, target: int = 3) -> float:
    """Среднее время первого прохода из start в target."""
    n = P.shape[0]
    # Среднее время пребывания в каждом состоянии
    mean_sojourn = np.zeros(n)
    for i in range(n):
        if i in sojourn:
            k, lam = sojourn[i]
            mean_sojourn[i] = lam * math.gamma(1 + 1 / k)
        else:
            mean_sojourn[i] = 30.0  # дефолт 30 мин

    # Система: m_i = τ_i + Σ_j P[i,j]·m_j для j ≠ target
    # m_target = 0
    indices = [i for i in range(n) if i != target]
    k = len(indices)
    if k == 0:
        return 0.0

    A = np.zeros((k, k))
    b = np.zeros(k)
    idx_map = {orig: new for new, orig in enumerate(indices)}

    for new_i, orig_i in enumerate(indices):
        b[new_i] = mean_sojourn[orig_i]
        for orig_j in range(n):
            if orig_j == target:
                continue
            new_j = idx_map.get(orig_j, -1)
            if new_j >= 0:
                A[new_i, new_j] = -P[orig_i, orig_j]
        A[new_i, new_i] += 1

    try:
        m = np.linalg.solve(A, b)
        start_idx = idx_map.get(start, 0)
        return float(m[start_idx])
    except Exception:
        return 0.0


# ═══════════════════════════════════════════════════════════════════════════
# ВИЗУАЛИЗАЦИИ
# ═══════════════════════════════════════════════════════════════════════════
def plot_transition_graph(result: OrgMarkovResult,
                          filename: str = "orgmc_graph.png") -> str:
    """Граф марковской цепи оргструктур."""
    fig, ax = plt.subplots(figsize=(12, 10), facecolor="white")
    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(-1.5, 1.5)
    ax.set_aspect("equal")
    ax.axis("off")

    n = N_ORG_STATES
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    # Начать сверху
    angles = angles - np.pi / 2
    r = 1.0
    pos = [(r * np.cos(a), r * np.sin(a)) for a in angles]

    # Узлы
    for i in range(n):
        x, y = pos[i]
        pi_val = result.stationary[i]
        size = 0.15 + 0.15 * pi_val / max(result.stationary.max(), 0.01)
        circle = plt.Circle((x, y), size, facecolor=ORG_COLORS[i],
                             edgecolor="white", linewidth=2, alpha=0.85)
        ax.add_patch(circle)
        ax.text(x, y + 0.02, ORG_SHORT[i], ha="center", va="center",
                fontsize=10, fontweight="bold", color="white")
        ax.text(x, y - size - 0.08, f"π={pi_val:.2f}",
                ha="center", fontsize=7, color="#7f8c8d")

    # Рёбра с вероятностями
    P = result.transition_matrix
    for i in range(n):
        for j in range(n):
            if i == j or P[i, j] < 0.02:
                continue
            x1, y1 = pos[i]
            x2, y2 = pos[j]
            # Смещение для избежания перекрытия
            dx, dy = x2 - x1, y2 - y1
            dist = math.sqrt(dx ** 2 + dy ** 2)
            if dist < 0.01:
                continue
            nx, ny = dx / dist, dy / dist
            # Укоротить стрелку
            sx, sy = x1 + nx * 0.2, y1 + ny * 0.2
            ex, ey = x2 - nx * 0.2, y2 - ny * 0.2

            lw = 0.5 + 3 * P[i, j]
            alpha = 0.3 + 0.5 * P[i, j]
            ax.annotate("", xy=(ex, ey), xytext=(sx, sy),
                        arrowprops=dict(arrowstyle="-|>",
                                       color=ORG_COLORS[i],
                                       lw=lw, alpha=alpha))
            # Подпись вероятности
            mx, my = (sx + ex) / 2, (sy + ey) / 2
            ax.text(mx + ny * 0.06, my - nx * 0.06,
                    f"{P[i, j]:.2f}", fontsize=6, color="#2c3e50",
                    ha="center", alpha=0.8,
                    bbox=dict(boxstyle="round,pad=0.1",
                              facecolor="white", alpha=0.7))

    # Легенда
    for i, label in enumerate(ORG_STATES):
        ax.text(-1.4, -1.3 + i * 0.12, f"● {label}",
                fontsize=7, color=ORG_COLORS[i])

    ax.set_title(f"Марковская цепь оргструктур "
                 f"({result.n_scenarios} сценариев)",
                 fontsize=13, fontweight="bold", pad=15)
    fig.tight_layout()
    return _save(fig, filename)


def plot_transition_heatmap(result: OrgMarkovResult,
                            filename: str = "orgmc_heatmap.png") -> str:
    """Тепловая карта матрицы переходов."""
    fig, ax = plt.subplots(figsize=(9, 7), facecolor="white")
    P = result.transition_matrix
    im = ax.imshow(P, cmap="YlOrRd", vmin=0, vmax=1)

    ax.set_xticks(range(N_ORG_STATES))
    ax.set_xticklabels(ORG_SHORT, fontsize=10)
    ax.set_yticks(range(N_ORG_STATES))
    ax.set_yticklabels(ORG_SHORT, fontsize=10)

    for i in range(N_ORG_STATES):
        for j in range(N_ORG_STATES):
            v = P[i, j]
            if v > 0.01:
                ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                        fontsize=10, fontweight="bold",
                        color="white" if v > 0.5 else "black")

    fig.colorbar(im, ax=ax, shrink=0.8, label="P(переход)")
    ax.set_xlabel("Следующее состояние", fontsize=11)
    ax.set_ylabel("Текущее состояние", fontsize=11)
    ax.set_title("Матрица переходов оргструктур",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    return _save(fig, filename)


def plot_org_gantt(result: OrgMarkovResult, n_show: int = 15,
                   filename: str = "orgmc_gantt.png") -> str:
    """Диаграмма Ганта: траектории оргструктур по сценариям."""
    fig, ax = plt.subplots(figsize=(14, max(4, n_show * 0.4)),
                           facecolor="white")

    trajs = result.trajectories[:n_show]
    for ti, traj in enumerate(trajs):
        y = n_show - ti - 1
        for si in range(len(traj) - 1):
            t_start, state = traj[si]
            t_end = traj[si + 1][0]
            ax.barh(y, t_end - t_start, left=t_start, height=0.7,
                    color=ORG_COLORS[state], edgecolor="white",
                    linewidth=0.5, alpha=0.85)
        # Последний сегмент
        if traj:
            t_last, state_last = traj[-1]
            ax.barh(y, 50, left=t_last, height=0.7,
                    color=ORG_COLORS[state_last], edgecolor="white",
                    linewidth=0.5, alpha=0.5)

    ax.set_yticks(range(n_show))
    ax.set_yticklabels([f"Сцен. {i + 1}" for i in range(n_show)], fontsize=8)
    ax.set_xlabel("Время (мин)", fontsize=10)
    ax.set_title("Траектории оргструктур по сценариям (диаграмма Ганта)",
                 fontsize=12, fontweight="bold")
    ax.grid(True, axis="x", alpha=0.3)

    # Легенда
    patches = [mpatches.Patch(color=ORG_COLORS[i], label=ORG_SHORT[i])
               for i in range(N_ORG_STATES)]
    ax.legend(handles=patches, fontsize=7, loc="upper right",
              ncol=4, framealpha=0.8)

    fig.tight_layout()
    return _save(fig, filename)


def plot_phase_org_relation(result: OrgMarkovResult,
                            filename: str = "orgmc_phase_org.png") -> str:
    """Связь фаз пожара ↔ состояний оргструктуры."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5), facecolor="white")

    # Тепловая карта P(Oi | Sj)
    M = result.phase_org_matrix
    im = ax1.imshow(M, cmap="YlGnBu", vmin=0, vmax=M.max() * 1.1,
                    aspect="auto")
    ax1.set_xticks(range(N_ORG_STATES))
    ax1.set_xticklabels(ORG_SHORT, fontsize=9)
    ax1.set_yticks(range(5))
    ax1.set_yticklabels(FIRE_PHASES, fontsize=10)
    for i in range(5):
        for j in range(N_ORG_STATES):
            v = M[i, j]
            if v > 0.01:
                ax1.text(j, i, f"{v:.2f}", ha="center", va="center",
                         fontsize=8, fontweight="bold",
                         color="white" if v > 0.3 else "black")
    fig.colorbar(im, ax=ax1, shrink=0.8, label="P(оргсостояние | фаза)")
    ax1.set_xlabel("Состояние оргструктуры")
    ax1.set_ylabel("Фаза пожара")
    ax1.set_title("P(оргструктура | фаза пожара)", fontweight="bold")

    # Столбчатая: доминирующее оргсостояние в каждой фазе
    dominant = M.argmax(axis=1)
    ax2.bar(range(5), [M[i, dominant[i]] for i in range(5)],
            color=[ORG_COLORS[d] for d in dominant], edgecolor="white")
    ax2.set_xticks(range(5))
    ax2.set_xticklabels(FIRE_PHASES, fontsize=10)
    ax2.set_ylabel("P(доминирующее состояние)")
    ax2.set_title("Типичная оргструктура по фазам", fontweight="bold")
    for i in range(5):
        ax2.text(i, M[i, dominant[i]] + 0.02, ORG_SHORT[dominant[i]],
                 ha="center", fontsize=10, fontweight="bold",
                 color=ORG_COLORS[dominant[i]])
    ax2.grid(True, axis="y", alpha=0.3)

    fig.suptitle("Связь фаз пожара и оргструктуры", fontsize=13,
                 fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    return _save(fig, filename)


def plot_sojourn_distributions(result: OrgMarkovResult,
                               filename: str = "orgmc_sojourn.png") -> str:
    """Распределения времени пребывания в каждом состоянии."""
    fig, axes = plt.subplots(2, 4, figsize=(16, 7), facecolor="white")
    axes = axes.flatten()

    for i in range(N_ORG_STATES):
        ax = axes[i]
        data = result.sojourn_times.get(i, [])
        if data:
            ax.hist(data, bins=min(15, len(data) // 2 + 1),
                    color=ORG_COLORS[i], alpha=0.7, edgecolor="white",
                    density=True)
            if i in result.sojourn_weibull:
                k, lam = result.sojourn_weibull[i]
                x = np.linspace(0.1, max(data) * 1.3, 100)
                pdf = sp_stats.weibull_min.pdf(x, k, 0, lam)
                ax.plot(x, pdf, color=ORG_COLORS[i], linewidth=2)
                ax.text(0.95, 0.95, f"k={k:.1f}\nλ={lam:.0f}",
                        transform=ax.transAxes, fontsize=7,
                        ha="right", va="top",
                        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))
        ax.set_title(ORG_SHORT[i], fontsize=10, fontweight="bold",
                     color=ORG_COLORS[i])
        ax.set_xlabel("мин", fontsize=7)
        ax.grid(True, alpha=0.3)

    # Последняя ячейка — сводка
    ax = axes[7]
    ax.axis("off")
    lines = ["Среднее время пребывания:", ""]
    for i in range(N_ORG_STATES):
        data = result.sojourn_times.get(i, [])
        if data:
            lines.append(f"  {ORG_SHORT[i]}: {np.mean(data):.0f} мин "
                         f"(n={len(data)})")
    lines.append(f"\nВремя до полной\nоргструктуры (O1→O4):")
    lines.append(f"  {result.first_passage_to_full:.0f} мин")
    ax.text(0.1, 0.95, "\n".join(lines), transform=ax.transAxes,
            fontsize=9, va="top", fontfamily="monospace",
            bbox=dict(boxstyle="round", facecolor="#f5f6fa"))

    fig.suptitle("Распределения времени пребывания в состояниях оргструктуры "
                 "(Вейбулл)", fontsize=12, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    return _save(fig, filename)


def plot_stationary_and_passage(result: OrgMarkovResult,
                                filename: str = "orgmc_stationary.png") -> str:
    """Стационарное распределение + время первого прохода."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5), facecolor="white")

    # Стационарное распределение
    pi = result.stationary
    ax1.bar(range(N_ORG_STATES), pi, color=ORG_COLORS, edgecolor="white",
            width=0.6)
    ax1.set_xticks(range(N_ORG_STATES))
    ax1.set_xticklabels(ORG_SHORT, fontsize=10)
    ax1.set_ylabel("π (стационарная вероятность)")
    ax1.set_title("Стационарное распределение оргструктур",
                  fontweight="bold")
    for i, v in enumerate(pi):
        ax1.text(i, v + 0.01, f"{v:.2f}", ha="center", fontsize=9,
                 fontweight="bold")
    ax1.grid(True, axis="y", alpha=0.3)

    # Среднее время первого прохода (из каждого состояния в O4)
    passages = []
    for start in range(N_ORG_STATES):
        t = _first_passage_time(result.transition_matrix,
                                result.sojourn_weibull,
                                start=start, target=3)
        passages.append(t)

    ax2.barh(range(N_ORG_STATES), passages, color=ORG_COLORS,
             edgecolor="white", height=0.6)
    ax2.set_yticks(range(N_ORG_STATES))
    ax2.set_yticklabels(ORG_SHORT, fontsize=10)
    ax2.set_xlabel("Время (мин)")
    ax2.set_title("Среднее время до полной оргструктуры (O4)",
                  fontweight="bold")
    for i, v in enumerate(passages):
        if v > 0:
            ax2.text(v + 1, i, f"{v:.0f}", va="center", fontsize=9,
                     fontweight="bold")
    ax2.grid(True, axis="x", alpha=0.3)

    fig.suptitle(f"Стационарный анализ ({result.n_scenarios} сценариев)",
                 fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    return _save(fig, filename)


# ═══════════════════════════════════════════════════════════════════════════
# ДЕМО
# ═══════════════════════════════════════════════════════════════════════════
def demo():
    import sys, io
    if sys.platform == "win32":
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8',
                                      errors='replace')

    # Генерация 100 сценариев
    rng = np.random.RandomState(42)
    scenarios = []
    for _ in range(100):
        rank = rng.choice([1, 2, 3, 4, 5], p=[0.15, 0.25, 0.25, 0.25, 0.10])
        duration = int(rank * rng.uniform(40, 300))
        scenarios.append({
            "fire_rank": rank,
            "total_duration_min": duration,
        })

    print("Построение марковской цепи оргструктур (100 сценариев)")
    result = build_org_markov(scenarios)

    print(f"\n  Матрица переходов:")
    for i in range(N_ORG_STATES):
        row = " ".join(f"{result.transition_matrix[i, j]:.2f}"
                       for j in range(N_ORG_STATES))
        print(f"    {ORG_SHORT[i]}: [{row}]")

    print(f"\n  Стационарное распределение:")
    for i in range(N_ORG_STATES):
        bar = "█" * int(result.stationary[i] * 30)
        print(f"    {ORG_SHORT[i]}: π={result.stationary[i]:.3f} {bar}")

    print(f"\n  Среднее время до полной оргструктуры (O1→O4): "
          f"{result.first_passage_to_full:.0f} мин")

    print(f"\n  Параметры Вейбулла (время пребывания):")
    for i in range(N_ORG_STATES):
        if i in result.sojourn_weibull:
            k, lam = result.sojourn_weibull[i]
            print(f"    {ORG_SHORT[i]}: k={k:.2f}, λ={lam:.0f} мин "
                  f"(n={len(result.sojourn_times.get(i, []))})")

    # Визуализации
    paths = {}
    paths["graph"] = plot_transition_graph(result)
    paths["heatmap"] = plot_transition_heatmap(result)
    paths["gantt"] = plot_org_gantt(result, n_show=20)
    paths["phase_org"] = plot_phase_org_relation(result)
    paths["sojourn"] = plot_sojourn_distributions(result)
    paths["stationary"] = plot_stationary_and_passage(result)

    print(f"\n  Визуализации (6):")
    for name, path in paths.items():
        print(f"    {name}: {path}")


if __name__ == "__main__":
    demo()
