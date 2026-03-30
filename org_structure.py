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
# ═══════════════════════════════════════════════════════════════════════════
# СЕТЕВОЙ АНАЛИЗ ОРГСТРУКТУРЫ (Social Network Analysis)
# ═══════════════════════════════════════════════════════════════════════════
@dataclass
class NetworkMetrics:
    """Метрики сетевого анализа оргструктуры."""
    # Основные
    n_nodes: int = 0                    # число вершин
    n_edges: int = 0                    # число рёбер (связей подчинения)
    density: float = 0.0                # плотность = E / (N*(N-1)/2)
    # Центральность
    degree_centrality: Dict[str, float] = field(default_factory=dict)
    betweenness_centrality: Dict[str, float] = field(default_factory=dict)
    closeness_centrality: Dict[str, float] = field(default_factory=dict)
    # Структурные
    diameter: int = 0                   # диаметр графа (макс. кратчайший путь)
    avg_path_length: float = 0.0        # средняя длина кратчайшего пути
    clustering_coeff: float = 0.0       # коэффициент кластеризации
    # Иерархические
    span_of_control: Dict[str, int] = field(default_factory=dict)  # число подчинённых
    avg_span: float = 0.0              # средний span of control
    max_span: int = 0                   # максимальный span of control
    hierarchy_index: float = 0.0        # Крэкхардт (0=плоская, 1=дерево)
    # Информационные
    freeman_centralization: float = 0.0 # централизация Фримена [0; 1]
    information_entropy: float = 0.0    # энтропия распределения связей


def compute_network_metrics(snapshot: OrgSnapshot) -> NetworkMetrics:
    """Вычислить все метрики сетевого анализа для снимка оргструктуры."""
    nodes = snapshot.nodes
    n = len(nodes)
    if n <= 1:
        return NetworkMetrics(n_nodes=n)

    metrics = NetworkMetrics(n_nodes=n)
    name_to_idx = {node.name: i for i, node in enumerate(nodes)}

    # Построить матрицу смежности (неориентированную)
    adj = np.zeros((n, n), dtype=int)
    edges = []
    for node in nodes:
        if node.parent and node.parent in name_to_idx:
            i = name_to_idx[node.name]
            j = name_to_idx[node.parent]
            adj[i, j] = 1
            adj[j, i] = 1
            edges.append((i, j))

    metrics.n_edges = len(edges)

    # ── Плотность ────────────────────────────────────────────────
    max_edges = n * (n - 1) / 2
    metrics.density = metrics.n_edges / max_edges if max_edges > 0 else 0

    # ── Степень вершин (degree) ──────────────────────────────────
    degrees = adj.sum(axis=1)
    for node in nodes:
        i = name_to_idx[node.name]
        metrics.degree_centrality[node.name] = float(degrees[i]) / max(1, n - 1)

    # ── Span of control (число непосредственных подчинённых) ─────
    for node in nodes:
        subordinates = sum(1 for other in nodes if other.parent == node.name)
        metrics.span_of_control[node.name] = subordinates
    spans = list(metrics.span_of_control.values())
    metrics.avg_span = float(np.mean(spans)) if spans else 0
    metrics.max_span = max(spans) if spans else 0

    # ── Кратчайшие пути (BFS) ────────────────────────────────────
    dist = np.full((n, n), np.inf)
    np.fill_diagonal(dist, 0)
    for i, j in edges:
        dist[i, j] = 1
        dist[j, i] = 1
    # Флойд-Уоршелл
    for k in range(n):
        for i in range(n):
            for j in range(n):
                if dist[i, k] + dist[k, j] < dist[i, j]:
                    dist[i, j] = dist[i, k] + dist[k, j]

    finite = dist[dist < np.inf]
    nonzero = finite[finite > 0]
    if len(nonzero) > 0:
        metrics.avg_path_length = float(np.mean(nonzero))
        metrics.diameter = int(np.max(nonzero))

    # ── Closeness centrality ─────────────────────────────────────
    for node in nodes:
        i = name_to_idx[node.name]
        row = dist[i, :]
        reachable = row[row < np.inf]
        total_dist = reachable.sum()
        metrics.closeness_centrality[node.name] = \
            float((len(reachable) - 1) / total_dist) if total_dist > 0 else 0

    # ── Betweenness centrality (приближённая) ────────────────────
    between = np.zeros(n)
    for s in range(n):
        for t in range(s + 1, n):
            if dist[s, t] == np.inf:
                continue
            # Число кратчайших путей через каждый узел
            for v in range(n):
                if v == s or v == t:
                    continue
                if abs(dist[s, v] + dist[v, t] - dist[s, t]) < 0.5:
                    between[v] += 1
    norm = max(1, (n - 1) * (n - 2) / 2)
    for node in nodes:
        i = name_to_idx[node.name]
        metrics.betweenness_centrality[node.name] = float(between[i] / norm)

    # ── Коэффициент кластеризации ────────────────────────────────
    cc_list = []
    for i in range(n):
        neighbors = [j for j in range(n) if adj[i, j] > 0]
        k = len(neighbors)
        if k < 2:
            cc_list.append(0.0)
            continue
        links = sum(1 for a in neighbors for b in neighbors
                    if a < b and adj[a, b] > 0)
        cc_list.append(2.0 * links / (k * (k - 1)))
    metrics.clustering_coeff = float(np.mean(cc_list))

    # ── Централизация Фримена ────────────────────────────────────
    dc_vals = list(metrics.degree_centrality.values())
    if dc_vals:
        max_dc = max(dc_vals)
        sum_diff = sum(max_dc - d for d in dc_vals)
        max_possible = (n - 1) * (n - 2) / max(1, n * (n - 1) / 2)
        metrics.freeman_centralization = \
            float(sum_diff / max_possible) if max_possible > 0 else 0

    # ── Индекс иерархии Крэкхардта ──────────────────────────────
    # = 1 - (число избыточных рёбер) / (N-1)
    # Для дерева: edges = N-1, индекс = 1
    # Для полного графа: индекс → 0
    if n > 1:
        metrics.hierarchy_index = \
            1.0 - max(0, metrics.n_edges - (n - 1)) / max(1, n - 1)

    # ── Информационная энтропия ──────────────────────────────────
    if degrees.sum() > 0:
        probs = degrees / degrees.sum()
        probs = probs[probs > 0]
        metrics.information_entropy = float(-np.sum(probs * np.log2(probs)))

    return metrics


def plot_network_analysis(snapshot: OrgSnapshot,
                          metrics: NetworkMetrics,
                          filename: str = "org_network.png") -> str:
    """Визуализация результатов сетевого анализа."""
    fig, axes = plt.subplots(2, 3, figsize=(16, 9), facecolor="white")

    nodes = snapshot.nodes
    names = [n.name for n in nodes]

    # 1. Degree centrality
    ax = axes[0, 0]
    dc = [metrics.degree_centrality.get(n, 0) for n in names]
    colors = [ROLE_COLORS.get(node.role, "#95a5a6") for node in nodes]
    ax.barh(range(len(names)), dc, color=colors, edgecolor="white")
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=8)
    ax.set_xlabel("Центральность по степени")
    ax.set_title("Степенная центральность", fontweight="bold", fontsize=10)
    ax.grid(True, axis="x", alpha=0.3)

    # 2. Betweenness centrality
    ax = axes[0, 1]
    bc = [metrics.betweenness_centrality.get(n, 0) for n in names]
    ax.barh(range(len(names)), bc, color=colors, edgecolor="white")
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=8)
    ax.set_xlabel("Центральность по посредничеству")
    ax.set_title("Посредническая центральность", fontweight="bold", fontsize=10)
    ax.grid(True, axis="x", alpha=0.3)

    # 3. Span of control
    ax = axes[0, 2]
    spans = [metrics.span_of_control.get(n, 0) for n in names]
    ax.barh(range(len(names)), spans, color=colors, edgecolor="white")
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=8)
    ax.set_xlabel("Число подчинённых")
    ax.set_title("Диапазон управления (Span of Control)", fontweight="bold",
                 fontsize=10)
    ax.grid(True, axis="x", alpha=0.3)

    # 4. Сводные метрики (текст)
    ax = axes[1, 0]
    ax.axis("off")
    text_lines = [
        f"Вершин:              {metrics.n_nodes}",
        f"Рёбер:               {metrics.n_edges}",
        f"Плотность:           {metrics.density:.3f}",
        f"Диаметр:             {metrics.diameter}",
        f"Средний путь:        {metrics.avg_path_length:.2f}",
        f"Кластеризация:       {metrics.clustering_coeff:.3f}",
        f"Централизация:       {metrics.freeman_centralization:.3f}",
        f"Индекс иерархии:     {metrics.hierarchy_index:.3f}",
        f"Энтропия:            {metrics.information_entropy:.3f}",
        f"Средний span:        {metrics.avg_span:.1f}",
        f"Макс. span:          {metrics.max_span}",
    ]
    ax.text(0.05, 0.95, "\n".join(text_lines), transform=ax.transAxes,
            fontsize=10, fontfamily="monospace", va="top",
            bbox=dict(boxstyle="round", facecolor="#f5f6fa"))
    ax.set_title("Сводные метрики сети", fontweight="bold", fontsize=10)

    # 5. Closeness centrality
    ax = axes[1, 1]
    cc = [metrics.closeness_centrality.get(n, 0) for n in names]
    ax.barh(range(len(names)), cc, color=colors, edgecolor="white")
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=8)
    ax.set_xlabel("Центральность по близости")
    ax.set_title("Центральность по близости", fontweight="bold", fontsize=10)
    ax.grid(True, axis="x", alpha=0.3)

    # 6. Граф (простая визуализация)
    ax = axes[1, 2]
    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(-1.5, 1.5)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title("Граф оргструктуры", fontweight="bold", fontsize=10)

    n = len(nodes)
    pos = {}
    for i, node in enumerate(nodes):
        angle = 2 * math.pi * i / n
        r = 0.8 + 0.3 * node.level
        pos[node.name] = (r * math.cos(angle), r * math.sin(angle))

    # Рёбра
    for node in nodes:
        if node.parent and node.parent in pos and node.name in pos:
            x1, y1 = pos[node.parent]
            x2, y2 = pos[node.name]
            ax.plot([x1, x2], [y1, y2], color="#bdc3c7", linewidth=1.5,
                    zorder=1)

    # Вершины
    for node in nodes:
        if node.name in pos:
            x, y = pos[node.name]
            color = ROLE_COLORS.get(node.role, "#95a5a6")
            size = 200 + 300 * metrics.degree_centrality.get(node.name, 0)
            ax.scatter(x, y, s=size, c=color, edgecolors="white",
                       linewidth=1.5, zorder=2)
            ax.text(x, y - 0.15, node.name, ha="center", fontsize=7,
                    fontweight="bold", zorder=3)

    fig.suptitle(f"Сетевой анализ оргструктуры (t = Ч+{snapshot.t} мин)",
                 fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    return _save(fig, filename)


def plot_network_dynamics(dynamics: OrgDynamics,
                          step: int = 0,
                          filename: str = "org_net_dynamics.png") -> str:
    """Динамика ВСЕХ сетевых метрик во времени."""
    snapshots = dynamics.snapshots
    if not snapshots:
        return ""

    # Вычислить метрики для каждого снимка
    times = []
    m_density = []
    m_diameter = []
    m_avg_path = []
    m_clustering = []
    m_centralization = []
    m_hierarchy = []
    m_entropy = []
    m_avg_span = []

    sample_step = max(1, len(snapshots) // 60)
    for i, snap in enumerate(snapshots):
        if i % sample_step != 0 and i != len(snapshots) - 1:
            continue
        net = compute_network_metrics(snap)
        times.append(snap.t)
        m_density.append(net.density)
        m_diameter.append(net.diameter)
        m_avg_path.append(net.avg_path_length)
        m_clustering.append(net.clustering_coeff)
        m_centralization.append(net.freeman_centralization)
        m_hierarchy.append(net.hierarchy_index)
        m_entropy.append(net.information_entropy)
        m_avg_span.append(net.avg_span)

    fig, axes = plt.subplots(2, 4, figsize=(18, 8), facecolor="white")
    metrics_data = [
        (axes[0, 0], m_density, "Плотность графа", "#3498db"),
        (axes[0, 1], m_diameter, "Диаметр графа", "#e74c3c"),
        (axes[0, 2], m_avg_path, "Средний кратчайший путь", "#27ae60"),
        (axes[0, 3], m_clustering, "Коэф. кластеризации", "#e67e22"),
        (axes[1, 0], m_centralization, "Централизация Фримена", "#8e44ad"),
        (axes[1, 1], m_hierarchy, "Индекс иерархии Крэкхардта", "#c0392b"),
        (axes[1, 2], m_entropy, "Энтропия связей (Шеннон)", "#1abc9c"),
        (axes[1, 3], m_avg_span, "Средний Span of Control", "#f39c12"),
    ]

    for ax, data, title, color in metrics_data:
        ax.plot(times, data, color=color, linewidth=2)
        ax.fill_between(times, data, alpha=0.15, color=color)
        ax.set_title(title, fontsize=9, fontweight="bold")
        ax.set_xlabel("Время (мин)", fontsize=7)
        ax.grid(True, alpha=0.3)
        ax.tick_params(labelsize=7)
        # Отметить смены РТП
        for t_ch, _, _ in dynamics.rtp_changes:
            ax.axvline(t_ch, color="#bdc3c7", linestyle="--", alpha=0.5)

    fig.suptitle("Динамика сетевых метрик оргструктуры",
                 fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    return _save(fig, filename)


def plot_adjacency_matrix(snapshot: OrgSnapshot,
                          filename: str = "org_adjacency.png") -> str:
    """Матрица смежности оргструктуры."""
    nodes = snapshot.nodes
    n = len(nodes)
    if n < 2:
        return ""

    names = [nd.name for nd in nodes]
    name_to_idx = {nd.name: i for i, nd in enumerate(nodes)}
    adj = np.zeros((n, n))
    for nd in nodes:
        if nd.parent and nd.parent in name_to_idx:
            i, j = name_to_idx[nd.name], name_to_idx[nd.parent]
            adj[i, j] = 1
            adj[j, i] = 1

    fig, ax = plt.subplots(figsize=(8, 7), facecolor="white")
    im = ax.imshow(adj, cmap="YlOrRd", vmin=0, vmax=1)
    ax.set_xticks(range(n))
    ax.set_xticklabels(names, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(n))
    ax.set_yticklabels(names, fontsize=8)
    for i in range(n):
        for j in range(n):
            if adj[i, j] > 0:
                ax.text(j, i, "1", ha="center", va="center",
                        fontsize=10, fontweight="bold", color="white")
    fig.colorbar(im, ax=ax, shrink=0.7, label="Связь")
    ax.set_title(f"Матрица смежности оргструктуры (t=Ч+{snapshot.t} мин)",
                 fontsize=11, fontweight="bold")
    fig.tight_layout()
    return _save(fig, filename)


def plot_centrality_comparison(snapshot: OrgSnapshot,
                               metrics: NetworkMetrics,
                               filename: str = "org_centrality_cmp.png") -> str:
    """Сравнение 3 типов центральности для каждого узла."""
    nodes = snapshot.nodes
    names = [n.name for n in nodes]
    n = len(names)

    fig, ax = plt.subplots(figsize=(10, max(4, n * 0.6)), facecolor="white")
    x = np.arange(n)
    w = 0.25

    dc = [metrics.degree_centrality.get(nm, 0) for nm in names]
    bc = [metrics.betweenness_centrality.get(nm, 0) for nm in names]
    cc = [metrics.closeness_centrality.get(nm, 0) for nm in names]

    ax.barh(x - w, dc, w, label="По степени", color="#3498db", edgecolor="white")
    ax.barh(x, bc, w, label="По посредничеству", color="#e67e22", edgecolor="white")
    ax.barh(x + w, cc, w, label="По близости", color="#27ae60", edgecolor="white")

    ax.set_yticks(x)
    ax.set_yticklabels(names, fontsize=9)
    ax.set_xlabel("Центральность", fontsize=10)
    ax.set_title("Сравнение трёх типов центральности",
                 fontsize=12, fontweight="bold")
    ax.legend(fontsize=9, loc="lower right")
    ax.grid(True, axis="x", alpha=0.3)
    fig.tight_layout()
    return _save(fig, filename)


def plot_span_of_control(snapshot: OrgSnapshot,
                         metrics: NetworkMetrics,
                         filename: str = "org_span.png") -> str:
    """Диапазон управления (span of control) каждого руководителя."""
    nodes = snapshot.nodes
    fig, ax = plt.subplots(figsize=(8, 5), facecolor="white")

    managers = [(n.name, metrics.span_of_control.get(n.name, 0))
                for n in nodes if metrics.span_of_control.get(n.name, 0) > 0]
    if not managers:
        managers = [(nodes[0].name, 0)] if nodes else [("—", 0)]

    names = [m[0] for m in managers]
    spans = [m[1] for m in managers]
    colors = [ROLE_COLORS.get(next((n.role for n in nodes if n.name == nm), ""),
              "#95a5a6") for nm in names]

    ax.bar(range(len(names)), spans, color=colors, edgecolor="white", width=0.6)
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, fontsize=9)
    ax.set_ylabel("Число подчинённых")
    ax.set_title(f"Диапазон управления (средний: {metrics.avg_span:.1f}, "
                 f"макс: {metrics.max_span})", fontsize=11, fontweight="bold")
    ax.grid(True, axis="y", alpha=0.3)

    # Порог Нормана (7±2)
    ax.axhline(7, color="#c0392b", linestyle="--", alpha=0.5,
               label="Порог Нормана (7)")
    ax.legend(fontsize=8)

    for i, s in enumerate(spans):
        ax.text(i, s + 0.1, str(s), ha="center", fontsize=10, fontweight="bold")

    fig.tight_layout()
    return _save(fig, filename)


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

    # Сетевой анализ
    net = compute_network_metrics(peak)
    print(f"\n  Сетевой анализ (t={peak.t} мин):")
    print(f"    Плотность:       {net.density:.3f}")
    print(f"    Диаметр:         {net.diameter}")
    print(f"    Средний путь:    {net.avg_path_length:.2f}")
    print(f"    Кластеризация:   {net.clustering_coeff:.3f}")
    print(f"    Централизация:   {net.freeman_centralization:.3f}")
    print(f"    Индекс иерархии: {net.hierarchy_index:.3f}")
    print(f"    Энтропия:        {net.information_entropy:.3f}")
    print(f"    Средний span:    {net.avg_span:.1f}")
    p_net = plot_network_analysis(peak, net, filename="org_network_a.png")
    print(f"  Сетевой анализ: {p_net}")

    # Дополнительные визуализации
    p_dyn = plot_network_dynamics(dyn_a, filename="org_net_dynamics_a.png")
    print(f"  Динамика метрик: {p_dyn}")
    p_adj = plot_adjacency_matrix(peak, filename="org_adjacency_a.png")
    print(f"  Матрица смежности: {p_adj}")
    p_cc = plot_centrality_comparison(peak, net, filename="org_centrality_a.png")
    print(f"  Сравнение центральностей: {p_cc}")
    p_sp = plot_span_of_control(peak, net, filename="org_span_a.png")
    print(f"  Диапазон управления: {p_sp}")

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
