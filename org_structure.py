"""
org_structure.py — Анализ организационной структуры управления тушением пожара.
═══════════════════════════════════════════════════════════════════════════════
Извлечение, визуализация и анализ оргструктуры из описаний пожаров:

1. Извлечение оргструктуры из хронологии:
   - РТП (кто, когда принял, когда сменён)
   - НБУ (сколько, какие секторы, когда назначены)
   - НШ, НТ, ОТ (когда сформированы)
   - Оперативный штаб (когда создан)

2. Динамика изменения оргструктуры во времени:
   - Момент создания/расформирования каждого звена
   - Рост числа уровней и элементов
   - Смены РТП с временны́ми метками

3. Метрики сложности оргструктуры:
   - Число уровней иерархии
   - Число элементов (узлов)
   - Связность (среднее число подчинённых)
   - Индекс Нормана (span of control)
   - Динамический индекс: скорость усложнения

4. Сравнительный анализ по массиву сценариев:
   - Зависимость сложности от ранга пожара
   - Типовые паттерны развития оргструктуры
   - Кластеризация по типу оргструктуры

5. Визуализации:
   - Дерево оргструктуры на момент t
   - Динамика сложности во времени
   - Сравнение оргструктур между сценариями
═══════════════════════════════════════════════════════════════════════════════
"""
from __future__ import annotations

import os
import re
import math
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

_OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "data", "figures")
os.makedirs(_OUT_DIR, exist_ok=True)

def _save(fig, name):
    path = os.path.join(_OUT_DIR, name)
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    return path


# ═══════════════════════════════════════════════════════════════════════════
# СТРУКТУРЫ ДАННЫХ
# ═══════════════════════════════════════════════════════════════════════════
@dataclass
class OrgNode:
    """Один элемент (должностное лицо) оргструктуры."""
    role: str           # "РТП", "НБУ", "НШ", "НТ", "ОТ", "ОШ"
    name: str           # "РТП-1", "НБУ-2 (восток)", "ОШ"
    sector: str = ""    # "юг", "восток", "запад", "" (нет сектора)
    t_start: int = 0    # время назначения (мин)
    t_end: int = -1     # время замены (-1 = до конца)
    parent: str = ""    # ID родительского узла ("РТП-1")
    level: int = 0      # уровень в иерархии (0 = верх)

    @property
    def active_at(self):
        """Проверить, активен ли узел в момент t."""
        def check(t):
            return self.t_start <= t and (self.t_end < 0 or t <= self.t_end)
        return check


@dataclass
class OrgSnapshot:
    """Снимок оргструктуры в момент t."""
    t: int
    nodes: List[OrgNode]
    n_levels: int = 0
    n_nodes: int = 0
    n_bu: int = 0
    has_shtab: bool = False
    has_nt: bool = False
    rtp_name: str = ""
    complexity_index: float = 0.0


@dataclass
class OrgDynamics:
    """Полная динамика оргструктуры за эпизод пожара."""
    snapshots: List[OrgSnapshot] = field(default_factory=list)
    rtp_changes: List[Tuple[int, str, str]] = field(default_factory=list)
    # [(t, old_rtp, new_rtp), ...]
    bu_history: List[Tuple[int, int]] = field(default_factory=list)
    # [(t, n_bu), ...]
    max_complexity: float = 0.0
    final_complexity: float = 0.0
    total_rtp_changes: int = 0
    time_to_full_structure: int = 0  # мин до максимальной оргструктуры


# ═══════════════════════════════════════════════════════════════════════════
# ИЗВЛЕЧЕНИЕ ОРГСТРУКТУРЫ ИЗ ХРОНОЛОГИИ
# ═══════════════════════════════════════════════════════════════════════════
_RE_RTP = re.compile(r'РТП[-\s]*(\d+)', re.IGNORECASE)
_RE_NBU = re.compile(r'НБУ[-\s]*(\d+)', re.IGNORECASE)
_RE_SHTAB = re.compile(r'(?:штаб|ОШ|оперативн\w*\s*штаб)', re.IGNORECASE)
_RE_NT = re.compile(r'(?:НТ|начальник\w*\s*тыл)', re.IGNORECASE)
_RE_NSH = re.compile(r'(?:НШ|начальник\w*\s*штаб)', re.IGNORECASE)
_RE_BU = re.compile(r'(?:БУ[-\s]*(\d+)|боев\w*\s*участ)', re.IGNORECASE)
_RE_SECTOR = re.compile(r'(?:юг|восток|запад|север)', re.IGNORECASE)
_RE_PRINYAT = re.compile(r'принял\s*руководств', re.IGNORECASE)
_RE_NAZNACHEN = re.compile(r'назначен|переназначен|создан|сформирован', re.IGNORECASE)


def extract_org_structure(timeline_events: List[Tuple[int, str, str]],
                          total_min: int = 300
                          ) -> OrgDynamics:
    """Извлечь оргструктуру из хронологии событий.

    timeline_events: [(t_min, color, description), ...]
    Формат из TankFireSim.events или TIMELINE.
    """
    dynamics = OrgDynamics()
    all_nodes: List[OrgNode] = []

    # Начальное состояние: РТП-1 с момента 0
    current_rtp = OrgNode(role="РТП", name="РТП-1", t_start=0, level=0)
    all_nodes.append(current_rtp)

    for t, color, desc in timeline_events:
        lower = desc.lower()

        # Смена РТП
        m_rtp = _RE_RTP.search(desc)
        if m_rtp and _RE_PRINYAT.search(desc):
            old_name = current_rtp.name
            current_rtp.t_end = t
            new_name = f"РТП-{m_rtp.group(1)}"
            current_rtp = OrgNode(role="РТП", name=new_name, t_start=t, level=0)
            all_nodes.append(current_rtp)
            dynamics.rtp_changes.append((t, old_name, new_name))

        # Оперативный штаб
        if _RE_SHTAB.search(desc) and _RE_NAZNACHEN.search(desc):
            exists = any(n.role == "ОШ" and n.t_end < 0 for n in all_nodes)
            if not exists:
                all_nodes.append(OrgNode(
                    role="ОШ", name="Оперативный штаб", t_start=t,
                    parent=current_rtp.name, level=1))

        # НШ
        if _RE_NSH.search(desc) and _RE_NAZNACHEN.search(desc):
            exists = any(n.role == "НШ" and n.t_end < 0 for n in all_nodes)
            if not exists:
                all_nodes.append(OrgNode(
                    role="НШ", name="НШ", t_start=t,
                    parent="ОШ", level=2))

        # НТ
        if _RE_NT.search(desc) and _RE_NAZNACHEN.search(desc):
            exists = any(n.role == "НТ" and n.t_end < 0 for n in all_nodes)
            if not exists:
                all_nodes.append(OrgNode(
                    role="НТ", name="НТ", t_start=t,
                    parent="ОШ", level=2))

        # НБУ
        m_nbu = _RE_NBU.findall(desc)
        if m_nbu and _RE_NAZNACHEN.search(desc):
            for num in m_nbu:
                name = f"НБУ-{num}"
                sector = ""
                m_sec = _RE_SECTOR.search(desc)
                if m_sec:
                    sector = m_sec.group(0).lower()
                exists = any(n.name == name and n.t_end < 0 for n in all_nodes)
                if not exists:
                    all_nodes.append(OrgNode(
                        role="НБУ", name=name, sector=sector,
                        t_start=t, parent=current_rtp.name, level=2))

        # БУ (боевые участки)
        m_bu = _RE_BU.findall(desc)
        if m_bu:
            n_bu = len([n for n in all_nodes
                        if n.role == "НБУ" and n.t_end < 0])
            dynamics.bu_history.append((t, max(n_bu, len(m_bu))))

    # Построить снимки
    step = max(1, total_min // 100)
    for t in range(0, total_min + step, step):
        active = [n for n in all_nodes if n.active_at(t)]
        n_bu = sum(1 for n in active if n.role == "НБУ")
        has_shtab = any(n.role == "ОШ" for n in active)
        has_nt = any(n.role == "НТ" for n in active)
        rtp = next((n for n in active if n.role == "РТП"), None)

        # Сложность = число уровней × log2(число узлов + 1)
        levels = set(n.level for n in active)
        n_levels = len(levels)
        n_nodes = len(active)
        complexity = n_levels * math.log2(n_nodes + 1) if n_nodes > 0 else 0

        snapshot = OrgSnapshot(
            t=t, nodes=list(active),
            n_levels=n_levels, n_nodes=n_nodes, n_bu=n_bu,
            has_shtab=has_shtab, has_nt=has_nt,
            rtp_name=rtp.name if rtp else "",
            complexity_index=round(complexity, 2),
        )
        dynamics.snapshots.append(snapshot)

    # Метрики
    if dynamics.snapshots:
        complexities = [s.complexity_index for s in dynamics.snapshots]
        dynamics.max_complexity = max(complexities)
        dynamics.final_complexity = complexities[-1]
        dynamics.total_rtp_changes = len(dynamics.rtp_changes)

        # Время до максимальной оргструктуры
        max_nodes = max(s.n_nodes for s in dynamics.snapshots)
        for s in dynamics.snapshots:
            if s.n_nodes == max_nodes:
                dynamics.time_to_full_structure = s.t
                break

    return dynamics


# ═══════════════════════════════════════════════════════════════════════════
# АНАЛИЗ ПО МАССИВУ СЦЕНАРИЕВ
# ═══════════════════════════════════════════════════════════════════════════
@dataclass
class OrgComparisonResult:
    """Результат сравнительного анализа оргструктур."""
    n_scenarios: int = 0
    complexity_by_rank: Dict[int, List[float]] = field(default_factory=dict)
    rtp_changes_by_rank: Dict[int, List[int]] = field(default_factory=dict)
    time_to_full_by_rank: Dict[int, List[int]] = field(default_factory=dict)
    # Корреляции
    rank_complexity_corr: float = 0.0
    rank_rtp_changes_corr: float = 0.0


def analyze_org_across_scenarios(
    scenarios: List[Dict],
) -> OrgComparisonResult:
    """Анализ оргструктуры по всему массиву сценариев.

    scenarios: список словарей с ключами:
        fire_rank, total_duration_min, timeline (или events)
    """
    result = OrgComparisonResult(n_scenarios=len(scenarios))

    ranks = []
    complexities = []
    rtp_counts = []

    for scen in scenarios:
        rank = scen.get("fire_rank", 2)
        total_min = scen.get("total_duration_min", 300)

        # Извлечь хронологию
        events = scen.get("timeline", scen.get("events", []))
        if not events:
            continue

        # Привести к формату (t, color, desc)
        normalized = []
        for ev in events:
            if isinstance(ev, (list, tuple)) and len(ev) >= 3:
                normalized.append((ev[0], str(ev[1]), str(ev[2])))
            elif isinstance(ev, dict):
                normalized.append((
                    ev.get("t_min", 0),
                    ev.get("category", "info"),
                    ev.get("description", ""),
                ))

        dynamics = extract_org_structure(normalized, total_min)

        # Собрать метрики
        ranks.append(rank)
        complexities.append(dynamics.max_complexity)
        rtp_counts.append(dynamics.total_rtp_changes)

        result.complexity_by_rank.setdefault(rank, []).append(
            dynamics.max_complexity)
        result.rtp_changes_by_rank.setdefault(rank, []).append(
            dynamics.total_rtp_changes)
        result.time_to_full_by_rank.setdefault(rank, []).append(
            dynamics.time_to_full_structure)

    # Корреляции
    if len(ranks) >= 3:
        from scipy import stats as sp_stats
        result.rank_complexity_corr = float(
            sp_stats.spearmanr(ranks, complexities)[0])
        result.rank_rtp_changes_corr = float(
            sp_stats.spearmanr(ranks, rtp_counts)[0])

    return result


# ═══════════════════════════════════════════════════════════════════════════
# ВИЗУАЛИЗАЦИИ
# ═══════════════════════════════════════════════════════════════════════════
ROLE_COLORS = {
    "РТП": "#c0392b", "ОШ": "#8e44ad", "НШ": "#2980b9",
    "НТ": "#27ae60", "НБУ": "#e67e22", "ОТ": "#1abc9c",
}

def plot_org_tree(snapshot: OrgSnapshot,
                  title: str = "",
                  filename: str = "org_tree.png") -> str:
    """Дерево оргструктуры на момент t."""
    fig, ax = plt.subplots(figsize=(12, 6), facecolor="white")
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 6)
    ax.axis("off")

    if not title:
        title = f"Организационная структура (t = Ч+{snapshot.t} мин)"
    ax.text(6, 5.7, title, ha="center", fontsize=12, fontweight="bold",
            color="#2c3e50")

    # Группировка по уровням
    levels = {}
    for node in snapshot.nodes:
        levels.setdefault(node.level, []).append(node)

    # Позиционирование
    positions = {}
    for level, nodes in sorted(levels.items()):
        n = len(nodes)
        for i, node in enumerate(nodes):
            x = (i + 1) * 12 / (n + 1)
            y = 5.0 - level * 1.5
            positions[node.name] = (x, y)

            color = ROLE_COLORS.get(node.role, "#95a5a6")
            rect = plt.Rectangle((x - 0.8, y - 0.3), 1.6, 0.6,
                                  facecolor=color, edgecolor="white",
                                  linewidth=1.5, alpha=0.85)
            ax.add_patch(rect)

            label = node.name
            if node.sector:
                label += f"\n({node.sector})"
            ax.text(x, y, label, ha="center", va="center",
                    fontsize=8, fontweight="bold", color="white")

    # Связи
    for node in snapshot.nodes:
        if node.parent and node.parent in positions and node.name in positions:
            x1, y1 = positions[node.parent]
            x2, y2 = positions[node.name]
            ax.plot([x1, x2], [y1 - 0.3, y2 + 0.3], color="#bdc3c7",
                    linewidth=1.5, zorder=0)

    # Метрики
    ax.text(0.5, 0.3,
            f"Узлов: {snapshot.n_nodes}  |  Уровней: {snapshot.n_levels}  |  "
            f"БУ: {snapshot.n_bu}  |  Штаб: {'Да' if snapshot.has_shtab else 'Нет'}  |  "
            f"Сложность: {snapshot.complexity_index:.1f}",
            fontsize=9, color="#7f8c8d")

    # Легенда
    for i, (role, color) in enumerate(ROLE_COLORS.items()):
        ax.add_patch(plt.Rectangle((0.5 + i * 1.8, 0.7), 0.3, 0.3,
                                    facecolor=color))
        ax.text(0.9 + i * 1.8, 0.85, role, fontsize=7, color="#2c3e50",
                va="center")

    fig.tight_layout()
    return _save(fig, filename)


def plot_org_dynamics(dynamics: OrgDynamics,
                      title: str = "Динамика организационной структуры",
                      filename: str = "org_dynamics.png") -> str:
    """Графики динамики оргструктуры во времени."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 8), facecolor="white")

    times = [s.t for s in dynamics.snapshots]

    # 1. Число узлов
    ax = axes[0, 0]
    nodes = [s.n_nodes for s in dynamics.snapshots]
    ax.plot(times, nodes, color="#c0392b", linewidth=2)
    ax.fill_between(times, nodes, alpha=0.15, color="#c0392b")
    ax.set_ylabel("Число элементов")
    ax.set_title("Рост организационной структуры", fontweight="bold")
    ax.grid(True, alpha=0.3)
    # Отметить смены РТП
    for t, old, new in dynamics.rtp_changes:
        ax.axvline(t, color="#8e44ad", linestyle="--", alpha=0.5)
        ax.text(t, max(nodes) * 0.9, f"{new}", fontsize=6,
                color="#8e44ad", rotation=90, va="top")

    # 2. Число уровней иерархии
    ax = axes[0, 1]
    levels = [s.n_levels for s in dynamics.snapshots]
    ax.step(times, levels, color="#2980b9", linewidth=2, where="post")
    ax.fill_between(times, levels, alpha=0.15, color="#2980b9", step="post")
    ax.set_ylabel("Число уровней")
    ax.set_title("Глубина иерархии", fontweight="bold")
    ax.grid(True, alpha=0.3)

    # 3. Индекс сложности
    ax = axes[1, 0]
    complexity = [s.complexity_index for s in dynamics.snapshots]
    ax.plot(times, complexity, color="#e67e22", linewidth=2)
    ax.fill_between(times, complexity, alpha=0.15, color="#e67e22")
    ax.set_xlabel("Время (мин)")
    ax.set_ylabel("Индекс сложности")
    ax.set_title("Сложность = уровни × log₂(узлы + 1)", fontweight="bold")
    ax.grid(True, alpha=0.3)
    ax.axhline(dynamics.max_complexity, color="#c0392b", linestyle=":",
               alpha=0.5, label=f"Макс: {dynamics.max_complexity:.1f}")
    ax.legend(fontsize=8)

    # 4. Число БУ
    ax = axes[1, 1]
    bu = [s.n_bu for s in dynamics.snapshots]
    ax.step(times, bu, color="#27ae60", linewidth=2, where="post")
    ax.fill_between(times, bu, alpha=0.15, color="#27ae60", step="post")
    ax.set_xlabel("Время (мин)")
    ax.set_ylabel("Число БУ")
    ax.set_title("Боевые участки", fontweight="bold")
    ax.grid(True, alpha=0.3)

    fig.suptitle(title, fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    return _save(fig, filename)


def plot_org_comparison(result: OrgComparisonResult,
                        filename: str = "org_comparison.png") -> str:
    """Сравнение оргструктур по рангам пожара."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), facecolor="white")
    colors = ["#3498db", "#27ae60", "#e67e22", "#c0392b", "#8e44ad"]

    # 1. Сложность по рангам
    ax = axes[0]
    ranks = sorted(result.complexity_by_rank.keys())
    data = [result.complexity_by_rank[r] for r in ranks]
    bp = ax.boxplot(data, labels=[f"Ранг {r}" for r in ranks],
                    patch_artist=True)
    for patch, c in zip(bp["boxes"], colors):
        patch.set_facecolor(c)
        patch.set_alpha(0.6)
    ax.set_ylabel("Макс. индекс сложности")
    ax.set_title(f"Сложность по рангам\n(ρ={result.rank_complexity_corr:.2f})",
                 fontweight="bold")
    ax.grid(True, axis="y", alpha=0.3)

    # 2. Смены РТП
    ax = axes[1]
    data2 = [result.rtp_changes_by_rank.get(r, [0]) for r in ranks]
    bp2 = ax.boxplot(data2, labels=[f"Ранг {r}" for r in ranks],
                     patch_artist=True)
    for patch, c in zip(bp2["boxes"], colors):
        patch.set_facecolor(c)
        patch.set_alpha(0.6)
    ax.set_ylabel("Число смен РТП")
    ax.set_title(f"Смены руководства\n(ρ={result.rank_rtp_changes_corr:.2f})",
                 fontweight="bold")
    ax.grid(True, axis="y", alpha=0.3)

    # 3. Время до полной оргструктуры
    ax = axes[2]
    data3 = [result.time_to_full_by_rank.get(r, [0]) for r in ranks]
    means = [np.mean(d) for d in data3]
    ax.bar(range(len(ranks)), means, color=colors[:len(ranks)],
           edgecolor="white", alpha=0.8)
    ax.set_xticks(range(len(ranks)))
    ax.set_xticklabels([f"Ранг {r}" for r in ranks])
    ax.set_ylabel("Время (мин)")
    ax.set_title("Время до полной оргструктуры", fontweight="bold")
    ax.grid(True, axis="y", alpha=0.3)
    for i, m in enumerate(means):
        ax.text(i, m + max(means) * 0.02, f"{m:.0f}", ha="center",
                fontsize=9, fontweight="bold")

    fig.suptitle(f"Сравнение оргструктур ({result.n_scenarios} сценариев)",
                 fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    return _save(fig, filename)


# ═══════════════════════════════════════════════════════════════════════════
# ДЕМОНСТРАЦИЯ
# ═══════════════════════════════════════════════════════════════════════════
def demo():
    """Демо с данными из встроенных сценариев."""
    import sys, io
    if sys.platform == "win32":
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8',
                                      errors='replace')

    # Загрузить хронологию сценария А
    from tank_fire_sim import TIMELINE, TIMELINE_SERP, TOTAL_MIN, TOTAL_MIN_SERP

    print("Анализ оргструктуры: Сценарий А (ранг №4)")
    events_a = [(t, c, d) for t, _, d, c in TIMELINE]
    dyn_a = extract_org_structure(events_a, TOTAL_MIN)

    print(f"  Макс. сложность: {dyn_a.max_complexity:.1f}")
    print(f"  Смен РТП: {dyn_a.total_rtp_changes}")
    print(f"  Время до полной структуры: {dyn_a.time_to_full_structure} мин")

    # Снимок на пике
    peak = max(dyn_a.snapshots, key=lambda s: s.complexity_index)
    print(f"  Пиковый момент: t={peak.t} мин")
    print(f"    Узлов: {peak.n_nodes}, Уровней: {peak.n_levels}, "
          f"БУ: {peak.n_bu}")
    for n in peak.nodes:
        print(f"    {n.role} {n.name} [{n.sector or '—'}] "
              f"(L{n.level}, t={n.t_start})")

    p1 = plot_org_tree(peak, filename="org_tree_a.png")
    print(f"  Дерево: {p1}")
    p2 = plot_org_dynamics(dyn_a, title="Оргструктура: Сценарий А (ранг №4)",
                           filename="org_dynamics_a.png")
    print(f"  Динамика: {p2}")

    # Сценарий Б
    print("\nАнализ оргструктуры: Сценарий Б (ранг №2)")
    events_b = [(t, c, d) for t, _, d, c in TIMELINE_SERP]
    dyn_b = extract_org_structure(events_b, TOTAL_MIN_SERP)
    print(f"  Макс. сложность: {dyn_b.max_complexity:.1f}")

    # Сравнение по синтетическим данным
    print("\nСравнительный анализ по массиву сценариев")
    rng = np.random.RandomState(42)
    fake_scenarios = []
    for _ in range(50):
        rank = rng.choice([1, 2, 3, 4, 5])
        duration = int(rank * rng.uniform(40, 300))
        events = []
        t = 0
        events.append((t, "info", "Сообщение о загорании"))
        t += rng.randint(5, 15)
        events.append((t, "info", f"РТП-1 прибыл, принял руководство"))
        if rank >= 2:
            t += rng.randint(5, 20)
            events.append((t, "info", "Создан оперативный штаб, назначен НШ, НТ"))
        if rank >= 3:
            t += rng.randint(10, 40)
            events.append((t, "info", "Назначены НБУ-1 (юг), НБУ-2 (восток)"))
        if rank >= 4:
            t += rng.randint(20, 60)
            events.append((t, "info", "Назначен НБУ-3 (запад), 3 БУ"))
            t += rng.randint(50, 200)
            events.append((t, "info", "РТП-2 принял руководство"))
        if rank >= 5:
            t += rng.randint(100, 300)
            events.append((t, "info", "РТП-3 принял руководство"))
        fake_scenarios.append({
            "fire_rank": rank,
            "total_duration_min": duration,
            "timeline": events,
        })

    comp = analyze_org_across_scenarios(fake_scenarios)
    print(f"  Сценариев: {comp.n_scenarios}")
    print(f"  Корреляция ранг↔сложность: ρ={comp.rank_complexity_corr:.2f}")
    print(f"  Корреляция ранг↔смены РТП: ρ={comp.rank_rtp_changes_corr:.2f}")
    p3 = plot_org_comparison(comp, filename="org_comparison.png")
    print(f"  Сравнение: {p3}")


if __name__ == "__main__":
    demo()
