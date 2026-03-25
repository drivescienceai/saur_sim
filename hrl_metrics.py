"""
hrl_metrics.py
════════════════════════════════════════════════════════════════════════════════
Статистическое сравнение Flat Q-learning vs 3-уровневый иерархический RL.

Метрики:
  success_rate         — доля успешных ликвидаций (0..1)
  steps_to_ext         — шагов до ликвидации (∞ при неуспехе)
  total_reward         — накопленная награда за эпизод
  fire_area_reduction  — (S_нач − S_мин) / S_нач
  foam_efficiency      — 1/foam_attacks при успехе, 0 при неуспехе

Статистика по N оценочным эпизодам:
  mean, std, median, IQR, 95% bootstrap CI

Тесты значимости (Flat vs Hierarchical):
  Критерий Манна-Уитни (непараметрический) — основной
  t-критерий Стьюдента — дополнительный для нормальных распределений
  Cohen's d — размер эффекта

Критерии интерпретации Cohen's d:
  |d| < 0.2  — пренебрежимо малый
  0.2–0.5    — малый
  0.5–0.8    — средний ✓
  > 0.8      — крупный ✓✓
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

try:
    from scipy import stats as _scipy_stats
    _SCIPY_OK = True
except ImportError:
    _SCIPY_OK = False


# ══════════════════════════════════════════════════════════════════════════════
# ИМЕНА МЕТРИК (для GUI и экспорта)
# ══════════════════════════════════════════════════════════════════════════════

METRIC_LABELS: Dict[str, str] = {
    "success_rate":         "Успешная ликвидация (%)",
    "steps_to_ext":         "Шагов до ликвидации",
    "total_reward":         "Накопленная награда",
    "fire_area_reduction":  "Снижение площади пожара (%)",
    "foam_efficiency":      "Эффективность пенной атаки",
}

HIGHER_IS_BETTER = {"success_rate", "total_reward", "fire_area_reduction",
                    "foam_efficiency"}
LOWER_IS_BETTER  = {"steps_to_ext"}

EVAL_METRICS = list(METRIC_LABELS.keys())


# ══════════════════════════════════════════════════════════════════════════════
# МЕТРИКИ ОДНОГО АГЕНТА
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class AgentMetrics:
    """Метрики одного агента по N оценочным эпизодам."""
    name: str

    # Сырые данные: метрика → список значений по эпизодам
    raw: Dict[str, List[float]] = field(default_factory=dict)

    # Агрегированные статистики (заполняются методом compute())
    mean:   Dict[str, float]              = field(default_factory=dict)
    std:    Dict[str, float]              = field(default_factory=dict)
    median: Dict[str, float]              = field(default_factory=dict)
    ci95:   Dict[str, Tuple[float, float]] = field(default_factory=dict)
    iqr:    Dict[str, float]              = field(default_factory=dict)

    def compute(self, n_bootstrap: int = 10000, seed: int = 42):
        """Вычислить все статистики из raw данных."""
        rng = np.random.RandomState(seed)
        for k, vals in self.raw.items():
            a = np.array(vals, dtype=float)
            if len(a) == 0:
                continue
            # Основные моменты
            self.mean[k]   = float(np.mean(a))
            self.std[k]    = float(np.std(a, ddof=1) if len(a) > 1 else 0.0)
            self.median[k] = float(np.median(a))
            q1, q3         = np.percentile(a, [25, 75])
            self.iqr[k]    = float(q3 - q1)
            # Bootstrap 95% CI для среднего
            if len(a) >= 2:
                boots = np.array([
                    np.mean(rng.choice(a, len(a), replace=True))
                    for _ in range(n_bootstrap)
                ])
                self.ci95[k] = (float(np.percentile(boots, 2.5)),
                                float(np.percentile(boots, 97.5)))
            else:
                self.ci95[k] = (self.mean[k], self.mean[k])

    def format_mean_ci(self, metric: str, pct: bool = False) -> str:
        """Форматирование mean [CI_low, CI_high] для таблицы."""
        m  = self.mean.get(metric, 0.0)
        ci = self.ci95.get(metric, (m, m))
        if pct:
            return f"{m*100:.1f}% [{ci[0]*100:.1f}%,{ci[1]*100:.1f}%]"
        return f"{m:.3f} [{ci[0]:.3f},{ci[1]:.3f}]"


# ══════════════════════════════════════════════════════════════════════════════
# СРАВНЕНИЕ ДВУХ АГЕНТОВ
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class ComparisonResult:
    """Результат статистического сравнения Flat vs Hierarchical."""
    flat: AgentMetrics
    hier: AgentMetrics

    # Тесты значимости
    mannwhitney_p: Dict[str, float] = field(default_factory=dict)
    ttest_p:       Dict[str, float] = field(default_factory=dict)
    cohens_d:      Dict[str, float] = field(default_factory=dict)
    significant:   Dict[str, bool]  = field(default_factory=dict)

    def run_tests(self, alpha: float = 0.05):
        """Провести статистические тесты по всем общим метрикам."""
        for k in EVAL_METRICS:
            if k not in self.flat.raw or k not in self.hier.raw:
                continue
            a = np.array(self.flat.raw[k], dtype=float)
            b = np.array(self.hier.raw[k], dtype=float)
            if len(a) < 2 or len(b) < 2:
                continue

            # ── Манна-Уитни (непараметрический) ──────────────────────────
            if _SCIPY_OK:
                try:
                    _, p_mw = _scipy_stats.mannwhitneyu(
                        a, b, alternative="two-sided")
                    self.mannwhitney_p[k] = float(p_mw)
                except Exception:
                    self.mannwhitney_p[k] = 1.0
                try:
                    _, p_t = _scipy_stats.ttest_ind(a, b, equal_var=False)
                    self.ttest_p[k] = float(p_t)
                except Exception:
                    self.ttest_p[k] = 1.0
            else:
                # Ручная реализация U-статистики Манна-Уитни
                self.mannwhitney_p[k] = _mannwhitney_p(a, b)
                self.ttest_p[k]       = _welch_t_p(a, b)

            # ── Cohen's d ─────────────────────────────────────────────────
            s_a = float(np.std(a, ddof=1))
            s_b = float(np.std(b, ddof=1))
            pooled = np.sqrt((s_a ** 2 + s_b ** 2) / 2)
            self.cohens_d[k] = float(
                (np.mean(b) - np.mean(a)) / pooled if pooled > 0 else 0.0
            )
            self.significant[k] = self.mannwhitney_p.get(k, 1.0) < alpha

    def effect_label(self, metric: str) -> str:
        """Словесная интерпретация размера эффекта Cohen's d."""
        d = abs(self.cohens_d.get(metric, 0.0))
        if d < 0.2: return "незначим"
        if d < 0.5: return "малый"
        if d < 0.8: return "средний"
        return "крупный"

    def improvement_pct(self, metric: str) -> float:
        """Относительное улучшение иерарх. агента vs flat (в %)."""
        fm = self.flat.mean.get(metric, 0.0)
        hm = self.hier.mean.get(metric, 0.0)
        if fm == 0:
            return 0.0
        return (hm - fm) / abs(fm) * 100.0

    def summary_table(self) -> List[List[str]]:
        """Таблица результатов для GUI и экспорта в научную статью."""
        headers = [
            "Метрика",
            "Flat mean [95% CI]",
            "Hier mean [95% CI]",
            "Δ%",
            "Cohen's d",
            "Эффект",
            "p (M-W)",
            "Значимо",
        ]
        rows = [headers]
        for k in EVAL_METRICS:
            if k not in self.flat.raw:
                continue
            label = METRIC_LABELS.get(k, k)
            pct   = k in ("success_rate", "fire_area_reduction")
            fm_str = self.flat.format_mean_ci(k, pct=pct)
            hm_str = self.hier.format_mean_ci(k, pct=pct)
            delta  = self.improvement_pct(k)
            d_val  = self.cohens_d.get(k, 0.0)
            p_val  = self.mannwhitney_p.get(k, 1.0)
            sig    = "✓" if self.significant.get(k) else "—"
            sign   = "+" if delta >= 0 else ""
            rows.append([
                label,
                fm_str,
                hm_str,
                f"{sign}{delta:.1f}%",
                f"{d_val:+.3f}",
                self.effect_label(k),
                f"{p_val:.4f}",
                sig,
            ])
        return rows

    def to_dict(self) -> dict:
        """Сериализация для JSON-экспорта."""
        return {
            "flat": {
                "name": self.flat.name,
                "mean": self.flat.mean,
                "std":  self.flat.std,
                "ci95": {k: list(v) for k, v in self.flat.ci95.items()},
            },
            "hierarchical": {
                "name": self.hier.name,
                "mean": self.hier.mean,
                "std":  self.hier.std,
                "ci95": {k: list(v) for k, v in self.hier.ci95.items()},
            },
            "tests": {
                "mannwhitney_p": self.mannwhitney_p,
                "ttest_p":       self.ttest_p,
                "cohens_d":      self.cohens_d,
                "significant":   self.significant,
            },
        }


# ══════════════════════════════════════════════════════════════════════════════
# ОЦЕНОЧНЫЕ ПРОГОНЫ
# ══════════════════════════════════════════════════════════════════════════════

def evaluate_flat(trained_sim,
                  n_episodes: int = 100,
                  seed_offset: int = 1000) -> AgentMetrics:
    """Оценить обученный flat Q-learning агент (TankFireSim).

    trained_sim: TankFireSim с уже обученным self.agent.
    seed_offset: начальный seed для оценочных прогонов (не пересекается с обучением).
    """
    try:
        from tank_fire_sim import TankFireSim
    except ImportError:
        from .tank_fire_sim import TankFireSim

    metrics = AgentMetrics(name="Flat Q-learning")
    for k in EVAL_METRICS:
        metrics.raw[k] = []

    # Сохранить Q-таблицу и ε из обученного агента
    Q_trained  = trained_sim.agent.Q.copy()
    scenario   = trained_sim.scenario

    rng = np.random.RandomState(seed_offset)
    for ep in range(n_episodes):
        seed = int(rng.randint(0, 99999))
        sim  = TankFireSim(seed=seed, training=False, scenario=scenario)
        # Восстановить обученные Q-значения, ε=0 (жадная политика)
        sim.agent.Q       = Q_trained.copy()
        sim.agent.epsilon = 0.0

        steps = 0
        while not sim.extinguished and sim.t < sim._cfg["total_min"]:
            sim.step()
            steps += 1
            if steps >= 2000:
                break

        init = sim._cfg["initial_fire_area"]
        attacks = sim.foam_attacks
        metrics.raw["success_rate"].append(float(sim.extinguished))
        metrics.raw["steps_to_ext"].append(
            float(steps) if sim.extinguished else 9999.0)
        metrics.raw["total_reward"].append(
            float(sum(sim.agent.episode_rewards[-1:])))
        metrics.raw["fire_area_reduction"].append(
            max(0.0, (init - sim.fire_area) / init))
        metrics.raw["foam_efficiency"].append(
            1.0 / max(1, attacks) if sim.extinguished else 0.0)

    metrics.compute()
    return metrics


def evaluate_hierarchical(trained_hsim,
                           n_episodes: int = 100,
                           seed_offset: int = 1000) -> AgentMetrics:
    """Оценить обученный иерархический агент (HierarchicalTankFireSim).

    trained_hsim: HierarchicalTankFireSim с обученными агентами.
    """
    try:
        from hrl_sim import HierarchicalTankFireSim
    except ImportError:
        from .hrl_sim import HierarchicalTankFireSim

    metrics = AgentMetrics(name="Hierarchical RL (3 уровня)")
    for k in EVAL_METRICS:
        metrics.raw[k] = []

    # Сохранить Q-таблицы
    Q1 = trained_hsim.l1.Q.copy()
    Q2 = trained_hsim.l2.Q.copy()
    Q3 = trained_hsim.l3.Q.copy()
    cfg      = trained_hsim.cfg
    scenario = trained_hsim.scenario

    rng = np.random.RandomState(seed_offset + 1)
    for ep in range(n_episodes):
        seed  = int(rng.randint(0, 99999))
        hsim  = HierarchicalTankFireSim(cfg=cfg, seed=seed, scenario=scenario)
        # Восстановить обученные Q-таблицы, выключить исследование
        hsim.l1.Q = Q1.copy(); hsim.l2.Q = Q2.copy(); hsim.l3.Q = Q3.copy()
        hsim.set_eval_mode()

        result = hsim.run_episode(training=False)

        metrics.raw["success_rate"].append(float(result["extinguished"]))
        metrics.raw["steps_to_ext"].append(
            float(result["total_steps"]) if result["extinguished"] else 9999.0)
        metrics.raw["total_reward"].append(float(result["total_reward"]))
        metrics.raw["fire_area_reduction"].append(
            float(result["fire_area_reduction"]))
        metrics.raw["foam_efficiency"].append(float(result["foam_efficiency"]))

    metrics.compute()
    return metrics


def run_full_comparison(trained_flat_sim, trained_hier_sim,
                        n_eval: int = 100,
                        seed_offset: int = 1000,
                        alpha: float = 0.05) -> ComparisonResult:
    """Полное сравнение: оценочные прогоны + все тесты значимости."""
    flat_m = evaluate_flat(trained_flat_sim, n_eval, seed_offset)
    hier_m = evaluate_hierarchical(trained_hier_sim, n_eval, seed_offset)
    result = ComparisonResult(flat=flat_m, hier=hier_m)
    result.run_tests(alpha=alpha)
    return result


# ══════════════════════════════════════════════════════════════════════════════
# КРИВЫЕ ОБУЧЕНИЯ
# ══════════════════════════════════════════════════════════════════════════════

def smooth_rewards(rewards: List[float], window: int = 20) -> np.ndarray:
    """Скользящее среднее кривой обучения."""
    if len(rewards) < window:
        return np.array(rewards)
    kernel = np.ones(window) / window
    return np.convolve(rewards, kernel, mode="valid")


def convergence_episode(rewards: List[float], window: int = 20,
                         tol: float = 0.01) -> int:
    """Эпизод первой сходимости: δ среднего < tol·|среднего|."""
    sm = smooth_rewards(rewards, window)
    if len(sm) < 2:
        return len(rewards)
    for i in range(1, len(sm)):
        if abs(sm[i] - sm[i-1]) < tol * (abs(sm[i]) + 1e-9):
            return i + window
    return len(rewards)


# ══════════════════════════════════════════════════════════════════════════════
# РУЧНЫЕ РЕАЛИЗАЦИИ ТЕСТОВ (fallback без scipy)
# ══════════════════════════════════════════════════════════════════════════════

def _mannwhitney_p(a: np.ndarray, b: np.ndarray) -> float:
    """Приближённый p-value критерия Манна-Уитни (нормальное приближение)."""
    n1, n2 = len(a), len(b)
    combined = np.concatenate([a, b])
    ranks    = _rankdata(combined)
    R1 = float(np.sum(ranks[:n1]))
    U1 = R1 - n1 * (n1 + 1) / 2.0
    mu_U = n1 * n2 / 2.0
    sigma_U = np.sqrt(n1 * n2 * (n1 + n2 + 1) / 12.0)
    if sigma_U == 0:
        return 1.0
    z = (U1 - mu_U) / sigma_U
    # Двусторонний p-value через стандартное нормальное
    p = 2.0 * _norm_sf(abs(z))
    return float(np.clip(p, 0.0, 1.0))


def _welch_t_p(a: np.ndarray, b: np.ndarray) -> float:
    """Welch t-test p-value."""
    n1, n2 = len(a), len(b)
    m1, m2 = np.mean(a), np.mean(b)
    v1, v2 = np.var(a, ddof=1), np.var(b, ddof=1)
    se = np.sqrt(v1 / n1 + v2 / n2)
    if se == 0:
        return 1.0
    t = (m1 - m2) / se
    df = (v1/n1 + v2/n2)**2 / ((v1/n1)**2/(n1-1) + (v2/n2)**2/(n2-1) + 1e-12)
    # Используем нормальное приближение при df > 30
    if df > 30:
        p = 2.0 * _norm_sf(abs(t))
    else:
        p = 1.0  # консервативная оценка
    return float(np.clip(p, 0.0, 1.0))


def _rankdata(a: np.ndarray) -> np.ndarray:
    """Ранжирование с усреднением при совпадениях."""
    n = len(a)
    order = np.argsort(a)
    ranks = np.empty(n)
    i = 0
    while i < n:
        j = i
        while j < n - 1 and a[order[j+1]] == a[order[j]]:
            j += 1
        avg_rank = (i + j) / 2.0 + 1
        for k in range(i, j + 1):
            ranks[order[k]] = avg_rank
        i = j + 1
    return ranks


def _norm_sf(z: float) -> float:
    """P(Z > z) для стандартного нормального (приближение Абрамовица)."""
    import math
    if z < 0:
        return 1.0 - _norm_sf(-z)
    t = 1.0 / (1.0 + 0.2316419 * z)
    poly = t * (0.319381530
                + t * (-0.356563782
                       + t * (1.781477937
                              + t * (-1.821255978
                                     + t * 1.330274429))))
    return float(math.exp(-0.5 * z * z) / math.sqrt(2 * math.pi) * poly)
