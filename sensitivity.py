"""
sensitivity.py — Анализ чувствительности и валидация модели.
═══════════════════════════════════════════════════════════════════════════════
Систематическое исследование: как параметры модели влияют на результат?
Какие параметры наиболее значимы? Совпадает ли модель с реальными данными?

Методы:
  1. Однофакторный анализ чувствительности (OAT)
  2. Глобальный анализ (метод Морриса)
  3. Перекрёстная валидация (k-fold)
  4. Валидация на прецедентах (модель vs реальный исход)
═══════════════════════════════════════════════════════════════════════════════
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Callable

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "data", "figures")
os.makedirs(_OUT_DIR, exist_ok=True)


# ═══════════════════════════════════════════════════════════════════════════
# 1. ОДНОФАКТОРНЫЙ АНАЛИЗ ЧУВСТВИТЕЛЬНОСТИ (OAT)
# ═══════════════════════════════════════════════════════════════════════════
@dataclass
class SensitivityResult:
    """Результат OAT для одного параметра."""
    param_name: str
    param_values: List[float]
    metric_values: List[float]  # значения отклика при вариации параметра
    metric_name: str
    elasticity: float  # (ΔY/Y) / (ΔX/X) — относительная чувствительность
    rank: int = 0  # ранг по влиянию (1 = наибольшее)


def one_at_a_time(
    run_fn: Callable[[Dict[str, float]], float],
    base_params: Dict[str, float],
    param_ranges: Dict[str, Tuple[float, float, int]],
    metric_name: str = "L7",
) -> List[SensitivityResult]:
    """Однофакторный анализ: варьировать каждый параметр при фиксации остальных.

    run_fn: функция (params_dict) → metric_value
    base_params: базовые значения всех параметров
    param_ranges: {param_name: (min, max, n_points)}
    """
    base_value = run_fn(base_params)
    results = []

    for param_name, (p_min, p_max, n_pts) in param_ranges.items():
        values = np.linspace(p_min, p_max, n_pts).tolist()
        metrics = []
        for v in values:
            params = dict(base_params)
            params[param_name] = v
            metrics.append(run_fn(params))

        # Эластичность: (ΔY/Y_base) / (ΔX/X_base)
        base_x = base_params[param_name]
        if abs(base_x) > 1e-8 and abs(base_value) > 1e-8:
            dx = p_max - p_min
            dy = max(metrics) - min(metrics)
            elasticity = (dy / abs(base_value)) / (dx / abs(base_x))
        else:
            elasticity = 0.0

        results.append(SensitivityResult(
            param_name=param_name,
            param_values=values,
            metric_values=metrics,
            metric_name=metric_name,
            elasticity=abs(elasticity),
        ))

    # Ранжирование по эластичности
    results.sort(key=lambda r: -r.elasticity)
    for i, r in enumerate(results):
        r.rank = i + 1

    return results


# ═══════════════════════════════════════════════════════════════════════════
# 2. ГЛОБАЛЬНЫЙ АНАЛИЗ ЧУВСТВИТЕЛЬНОСТИ (МЕТОД МОРРИСА)
# ═══════════════════════════════════════════════════════════════════════════
@dataclass
class MorrisResult:
    """Результат метода Морриса для одного параметра."""
    param_name: str
    mu_star: float      # |μ*| — среднее абс. элементарного эффекта
    sigma: float         # σ — стандартное отклонение эл. эффекта
    # μ* высокий → параметр влиятельный
    # σ высокий → эффект нелинейный или взаимодействует с другими


def morris_screening(
    run_fn: Callable[[Dict[str, float]], float],
    param_ranges: Dict[str, Tuple[float, float]],
    n_trajectories: int = 20,
    n_levels: int = 4,
    seed: int = 42,
) -> List[MorrisResult]:
    """Метод Морриса (элементарные эффекты).

    Эффективнее OAT для большого числа параметров: O(k·r) вычислений
    вместо O(k·n) для OAT.
    """
    rng = np.random.RandomState(seed)
    param_names = list(param_ranges.keys())
    k = len(param_names)

    if k == 0:
        return []

    bounds = np.array([(lo, hi) for lo, hi in param_ranges.values()])
    delta = 1.0 / (n_levels - 1)

    effects = {name: [] for name in param_names}

    for _ in range(n_trajectories):
        # Случайная начальная точка на решётке
        x0 = rng.randint(0, n_levels, size=k) / (n_levels - 1)

        # Масштабировать в реальный диапазон
        def _to_real(x_norm):
            return {name: bounds[i, 0] + x_norm[i] * (bounds[i, 1] - bounds[i, 0])
                    for i, name in enumerate(param_names)}

        y0 = run_fn(_to_real(x0))

        # Для каждого параметра — элементарный эффект
        order = rng.permutation(k)
        x_cur = x0.copy()
        y_cur = y0

        for idx in order:
            x_new = x_cur.copy()
            x_new[idx] = min(1.0, x_cur[idx] + delta)
            if x_new[idx] == x_cur[idx]:
                x_new[idx] = max(0.0, x_cur[idx] - delta)

            y_new = run_fn(_to_real(x_new))
            ee = (y_new - y_cur) / delta
            effects[param_names[idx]].append(ee)

            x_cur = x_new
            y_cur = y_new

    results = []
    for name in param_names:
        ee_arr = np.array(effects[name])
        if len(ee_arr) == 0:
            continue
        results.append(MorrisResult(
            param_name=name,
            mu_star=float(np.mean(np.abs(ee_arr))),
            sigma=float(np.std(ee_arr)),
        ))

    results.sort(key=lambda r: -r.mu_star)
    return results


# ═══════════════════════════════════════════════════════════════════════════
# 3. ПЕРЕКРЁСТНАЯ ВАЛИДАЦИЯ
# ═══════════════════════════════════════════════════════════════════════════
@dataclass
class ValidationResult:
    """Результат валидации модели."""
    method: str
    n_total: int
    n_correct: int
    accuracy: float
    mae: float           # средняя абсолютная ошибка
    rmse: float           # среднеквадратическая ошибка
    r_squared: float      # коэффициент детерминации
    details: List[Dict] = field(default_factory=list)


def cross_validate(
    predict_fn: Callable[[Dict], float],
    cases: List[Dict],
    target_key: str = "duration_min",
    k_folds: int = 5,
    seed: int = 42,
) -> ValidationResult:
    """K-fold перекрёстная валидация.

    predict_fn: (case_dict) → predicted_value
    cases: список словарей прецедентов
    target_key: ключ целевой переменной
    """
    rng = np.random.RandomState(seed)
    n = len(cases)
    if n < k_folds:
        k_folds = max(2, n)

    indices = rng.permutation(n)
    fold_size = n // k_folds

    all_true = []
    all_pred = []
    details = []

    for fold in range(k_folds):
        start = fold * fold_size
        end = start + fold_size if fold < k_folds - 1 else n
        test_idx = indices[start:end]

        for idx in test_idx:
            case = cases[idx]
            true_val = case.get(target_key, 0)
            try:
                pred_val = predict_fn(case)
            except Exception:
                pred_val = 0
            all_true.append(true_val)
            all_pred.append(pred_val)
            details.append({
                "case": case.get("case_id", str(idx)),
                "true": true_val,
                "predicted": pred_val,
                "error": pred_val - true_val,
            })

    true_arr = np.array(all_true)
    pred_arr = np.array(all_pred)

    mae = float(np.mean(np.abs(true_arr - pred_arr)))
    rmse = float(np.sqrt(np.mean((true_arr - pred_arr) ** 2)))
    ss_res = np.sum((true_arr - pred_arr) ** 2)
    ss_tot = np.sum((true_arr - true_arr.mean()) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0

    # Accuracy: доля предсказаний в пределах ±20%
    n_correct = sum(1 for t, p in zip(all_true, all_pred)
                    if abs(t) > 0 and abs(p - t) / abs(t) <= 0.20)

    return ValidationResult(
        method=f"{k_folds}-fold перекрёстная валидация",
        n_total=n,
        n_correct=n_correct,
        accuracy=n_correct / max(n, 1),
        mae=mae,
        rmse=rmse,
        r_squared=float(r2),
        details=details,
    )


# ═══════════════════════════════════════════════════════════════════════════
# ВИЗУАЛИЗАЦИИ
# ═══════════════════════════════════════════════════════════════════════════
def plot_sensitivity_tornado(results: List[SensitivityResult],
                             filename: str = "sens_tornado.png") -> str:
    """Торнадо-диаграмма чувствительности."""
    fig, ax = plt.subplots(figsize=(10, max(4, len(results) * 0.6)),
                           facecolor="#f5f6fa")

    names = [r.param_name for r in results]
    elasticities = [r.elasticity for r in results]

    colors = ["#c0392b" if e > 0.5 else "#e67e22" if e > 0.2 else "#27ae60"
              for e in elasticities]

    y_pos = range(len(results))
    ax.barh(y_pos, elasticities, color=colors, edgecolor="white", height=0.6)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(names, fontsize=9)
    ax.set_xlabel("Эластичность |ΔY/Y| / |ΔX/X|", fontsize=10)
    ax.set_title("Анализ чувствительности: ранжирование параметров",
                 fontsize=12, fontweight="bold")
    ax.axvline(0.5, color="#bdc3c7", linestyle="--", alpha=0.7,
               label="Порог высокой чувств.")
    ax.legend(fontsize=8)
    ax.grid(True, axis="x", alpha=0.3)

    for i, e in enumerate(elasticities):
        ax.text(e + 0.02, i, f"{e:.3f}", va="center", fontsize=8)

    fig.tight_layout()
    path = os.path.join(_OUT_DIR, filename)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_morris(results: List[MorrisResult],
                filename: str = "sens_morris.png") -> str:
    """Диаграмма μ* vs σ (метод Морриса)."""
    fig, ax = plt.subplots(figsize=(8, 6), facecolor="#f5f6fa")
    ax.set_facecolor("#fafafa")

    for r in results:
        ax.scatter(r.mu_star, r.sigma, s=80, zorder=5)
        ax.annotate(r.param_name, (r.mu_star, r.sigma),
                    textcoords="offset points", xytext=(5, 5),
                    fontsize=8)

    ax.set_xlabel("|μ*| — среднее абсолютное влияние", fontsize=10)
    ax.set_ylabel("σ — нелинейность / взаимодействие", fontsize=10)
    ax.set_title("Глобальный анализ чувствительности (метод Морриса)",
                 fontsize=12, fontweight="bold")

    # Квадранты
    if results:
        x_mid = np.median([r.mu_star for r in results])
        y_mid = np.median([r.sigma for r in results])
        ax.axvline(x_mid, color="#bdc3c7", linestyle=":", alpha=0.5)
        ax.axhline(y_mid, color="#bdc3c7", linestyle=":", alpha=0.5)

        ax.text(0.98, 0.02, "Линейное\nвлияние", transform=ax.transAxes,
                ha="right", va="bottom", fontsize=7, color="#7f8c8d")
        ax.text(0.98, 0.98, "Нелинейное\nвлияние", transform=ax.transAxes,
                ha="right", va="top", fontsize=7, color="#7f8c8d")
        ax.text(0.02, 0.02, "Незначимые\nпараметры", transform=ax.transAxes,
                ha="left", va="bottom", fontsize=7, color="#7f8c8d")

    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    path = os.path.join(_OUT_DIR, filename)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_validation_scatter(result: ValidationResult,
                            filename: str = "sens_validation.png") -> str:
    """Диаграмма рассеяния: предсказание vs реальность."""
    fig, ax = plt.subplots(figsize=(7, 7), facecolor="#f5f6fa")

    true_vals = [d["true"] for d in result.details]
    pred_vals = [d["predicted"] for d in result.details]

    ax.scatter(true_vals, pred_vals, s=30, alpha=0.6, color="#3498db",
               edgecolors="white", linewidth=0.5)

    # Линия идеального совпадения
    lims = [min(min(true_vals), min(pred_vals)),
            max(max(true_vals), max(pred_vals))]
    ax.plot(lims, lims, "--", color="#c0392b", linewidth=1.5,
            label="Идеальное совпадение")

    ax.set_xlabel("Реальное значение", fontsize=10)
    ax.set_ylabel("Предсказание модели", fontsize=10)
    ax.set_title(f"Валидация модели (R²={result.r_squared:.3f}, "
                 f"MAE={result.mae:.1f})",
                 fontsize=12, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_aspect("equal")

    fig.tight_layout()
    path = os.path.join(_OUT_DIR, filename)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


# ═══════════════════════════════════════════════════════════════════════════
# ДЕМОНСТРАЦИЯ
# ═══════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    rng = np.random.RandomState(42)

    # Модельная функция: L7 = f(alpha, epsilon, gamma, lambda, foam_intensity)
    def model_fn(params):
        alpha = params.get("alpha", 0.15)
        eps = params.get("epsilon", 0.90)
        gamma = params.get("gamma", 0.95)
        lam = params.get("lambda", 0.30)
        foam = params.get("foam_intensity", 0.065)
        noise = rng.normal(0, 0.02)
        return (0.5 + 0.3 * alpha + 0.1 * (1 - eps) + 0.15 * gamma
                - 0.2 * lam + 2.0 * foam + noise)

    base = {"alpha": 0.15, "epsilon": 0.90, "gamma": 0.95,
            "lambda": 0.30, "foam_intensity": 0.065}
    ranges = {
        "alpha": (0.01, 0.50, 10),
        "epsilon": (0.10, 1.00, 10),
        "gamma": (0.80, 0.99, 10),
        "lambda": (0.00, 1.00, 10),
        "foam_intensity": (0.03, 0.10, 10),
    }

    # OAT
    oat = one_at_a_time(model_fn, base, ranges)
    print("OAT — ранжирование по эластичности:")
    for r in oat:
        print(f"  #{r.rank} {r.param_name}: эластичность={r.elasticity:.3f}")

    path_tornado = plot_sensitivity_tornado(oat)
    print(f"\nТорнадо: {path_tornado}")

    # Моррис
    morris_ranges = {k: (v[0], v[1]) for k, v in ranges.items()}
    morris = morris_screening(model_fn, morris_ranges, n_trajectories=30)
    print("\nМоррис — μ* vs σ:")
    for m in morris:
        print(f"  {m.param_name}: μ*={m.mu_star:.4f}, σ={m.sigma:.4f}")

    path_morris = plot_morris(morris)
    print(f"\nМоррис: {path_morris}")
