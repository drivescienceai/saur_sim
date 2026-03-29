"""
stat_analysis.py — Статистический анализ базы пожаров САУР-ПСП.
═══════════════════════════════════════════════════════════════════════════════
Полный набор методов для исследования закономерностей в данных о пожарах:

  1. Описательная статистика (среднее, медиана, ДИ, квантили)
  2. Корреляционный анализ (Пирсон, Спирмен, матрица корреляций)
  3. Дисперсионный анализ (ANOVA, Краскела-Уоллиса)
  4. Регрессионный анализ (линейная, множественная)
  5. Анализ распределений (Шапиро-Уилк, Колмогоров-Смирнов, Вейбулл)
  6. Анализ выживаемости (Каплан-Мейер для времени ликвидации)
  7. Визуализация результатов
═══════════════════════════════════════════════════════════════════════════════
"""
from __future__ import annotations

import os
import json
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple

import numpy as np
from scipy import stats as sp_stats

try:
    from .cbr_engine import CaseBase, FireCase, FEATURE_NAMES, N_FEATURES
except ImportError:
    from cbr_engine import CaseBase, FireCase, FEATURE_NAMES, N_FEATURES

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "data", "figures")
os.makedirs(_OUT_DIR, exist_ok=True)


def _save(fig, name: str) -> str:
    path = os.path.join(_OUT_DIR, name)
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    return path


# ═══════════════════════════════════════════════════════════════════════════
# 1. ОПИСАТЕЛЬНАЯ СТАТИСТИКА
# ═══════════════════════════════════════════════════════════════════════════
@dataclass
class DescriptiveStats:
    """Описательная статистика для одного признака."""
    name: str
    n: int
    mean: float
    std: float
    median: float
    q25: float
    q75: float
    min_val: float
    max_val: float
    ci95_low: float
    ci95_high: float
    skewness: float
    kurtosis: float


def descriptive_statistics(cb: CaseBase) -> List[DescriptiveStats]:
    """Описательная статистика по всем признакам."""
    if cb._feature_matrix is None:
        cb._build_matrix()
    X = cb._feature_matrix
    results = []
    for j in range(N_FEATURES):
        col = X[:, j]
        n = len(col)
        if n == 0:
            continue
        mean = float(np.mean(col))
        std = float(np.std(col, ddof=1)) if n > 1 else 0.0
        se = std / np.sqrt(n) if n > 0 else 0.0
        t_crit = sp_stats.t.ppf(0.975, max(1, n - 1))
        results.append(DescriptiveStats(
            name=FEATURE_NAMES[j],
            n=n,
            mean=mean,
            std=std,
            median=float(np.median(col)),
            q25=float(np.percentile(col, 25)),
            q75=float(np.percentile(col, 75)),
            min_val=float(np.min(col)),
            max_val=float(np.max(col)),
            ci95_low=mean - t_crit * se,
            ci95_high=mean + t_crit * se,
            skewness=float(sp_stats.skew(col)) if n > 2 else 0.0,
            kurtosis=float(sp_stats.kurtosis(col)) if n > 3 else 0.0,
        ))
    return results


# ═══════════════════════════════════════════════════════════════════════════
# 2. КОРРЕЛЯЦИОННЫЙ АНАЛИЗ
# ═══════════════════════════════════════════════════════════════════════════
@dataclass
class CorrelationResult:
    """Результат корреляции между двумя признаками."""
    feature_a: str
    feature_b: str
    pearson_r: float
    pearson_p: float
    spearman_rho: float
    spearman_p: float
    significant: bool  # p < 0.05


def correlation_matrix(cb: CaseBase) -> Tuple[np.ndarray, np.ndarray,
                                               List[CorrelationResult]]:
    """Матрица корреляций Пирсона + Спирмена + значимые пары.

    Возвращает: (pearson_matrix, spearman_matrix, significant_pairs)
    """
    if cb._feature_matrix is None:
        cb._build_matrix()
    X = cb._feature_matrix
    n_feat = X.shape[1]
    n_obs = X.shape[0]

    pearson_m = np.corrcoef(X.T)
    spearman_m = np.zeros((n_feat, n_feat))
    pairs = []

    for i in range(n_feat):
        for j in range(i + 1, n_feat):
            if n_obs < 3:
                continue
            r_p, p_p = sp_stats.pearsonr(X[:, i], X[:, j])
            r_s, p_s = sp_stats.spearmanr(X[:, i], X[:, j])
            spearman_m[i, j] = r_s
            spearman_m[j, i] = r_s
            sig = p_p < 0.05 or p_s < 0.05
            pairs.append(CorrelationResult(
                feature_a=FEATURE_NAMES[i],
                feature_b=FEATURE_NAMES[j],
                pearson_r=float(r_p),
                pearson_p=float(p_p),
                spearman_rho=float(r_s),
                spearman_p=float(p_s),
                significant=sig,
            ))

    np.fill_diagonal(spearman_m, 1.0)
    return pearson_m, spearman_m, pairs


# ═══════════════════════════════════════════════════════════════════════════
# 3. ДИСПЕРСИОННЫЙ АНАЛИЗ
# ═══════════════════════════════════════════════════════════════════════════
@dataclass
class ANOVAResult:
    """Результат дисперсионного анализа."""
    factor_name: str
    response_name: str
    method: str            # "ANOVA" или "Краскел-Уоллис"
    statistic: float
    p_value: float
    significant: bool
    group_means: Dict[str, float]
    eta_squared: float     # размер эффекта


def anova_by_factor(cb: CaseBase, factor_idx: int,
                    response_idx: int) -> ANOVAResult:
    """Дисперсионный анализ: влияние фактора на отклик.

    factor_idx: индекс категориального признака (ранг, топливо, кровля)
    response_idx: индекс непрерывного признака (длительность, площадь)
    """
    if cb._feature_matrix is None:
        cb._build_matrix()
    X = cb._feature_matrix
    factor = X[:, factor_idx]
    response = X[:, response_idx]

    groups = {}
    for i, f in enumerate(factor):
        key = f"{f:.0f}"
        groups.setdefault(key, []).append(response[i])

    group_arrays = [np.array(v) for v in groups.values() if len(v) >= 2]
    group_means = {k: float(np.mean(v)) for k, v in groups.items()}

    if len(group_arrays) < 2:
        return ANOVAResult(
            factor_name=FEATURE_NAMES[factor_idx],
            response_name=FEATURE_NAMES[response_idx],
            method="—", statistic=0, p_value=1, significant=False,
            group_means=group_means, eta_squared=0)

    # Проверка нормальности (для выбора метода)
    all_normal = all(
        sp_stats.shapiro(g)[1] > 0.05 if len(g) >= 3 else True
        for g in group_arrays
    )

    if all_normal and len(group_arrays) >= 2:
        stat, p = sp_stats.f_oneway(*group_arrays)
        method = "Однофакторный дисперсионный анализ"
    else:
        stat, p = sp_stats.kruskal(*group_arrays)
        method = "Критерий Краскела-Уоллиса"

    # Eta-squared (размер эффекта)
    ss_between = sum(len(g) * (np.mean(g) - response.mean()) ** 2
                     for g in group_arrays)
    ss_total = np.sum((response - response.mean()) ** 2)
    eta_sq = ss_between / ss_total if ss_total > 0 else 0

    return ANOVAResult(
        factor_name=FEATURE_NAMES[factor_idx],
        response_name=FEATURE_NAMES[response_idx],
        method=method,
        statistic=float(stat),
        p_value=float(p),
        significant=p < 0.05,
        group_means=group_means,
        eta_squared=float(eta_sq),
    )


# ═══════════════════════════════════════════════════════════════════════════
# 4. РЕГРЕССИОННЫЙ АНАЛИЗ
# ═══════════════════════════════════════════════════════════════════════════
@dataclass
class RegressionResult:
    """Результат линейной регрессии."""
    predictors: List[str]
    response: str
    coefficients: Dict[str, float]  # включая intercept
    r_squared: float
    adj_r_squared: float
    f_statistic: float
    f_p_value: float
    residual_std: float


def multiple_regression(cb: CaseBase,
                        predictor_indices: List[int],
                        response_idx: int) -> RegressionResult:
    """Множественная линейная регрессия."""
    if cb._feature_matrix is None:
        cb._build_matrix()
    X = cb._feature_matrix
    n = X.shape[0]
    p = len(predictor_indices)

    if n <= p + 1:
        return RegressionResult(
            predictors=[FEATURE_NAMES[i] for i in predictor_indices],
            response=FEATURE_NAMES[response_idx],
            coefficients={}, r_squared=0, adj_r_squared=0,
            f_statistic=0, f_p_value=1, residual_std=0)

    Xp = np.column_stack([np.ones(n)] + [X[:, i] for i in predictor_indices])
    y = X[:, response_idx]

    # OLS: β = (X'X)^{-1} X'y
    try:
        beta = np.linalg.lstsq(Xp, y, rcond=None)[0]
    except np.linalg.LinAlgError:
        beta = np.zeros(p + 1)

    y_pred = Xp @ beta
    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0
    adj_r2 = 1 - (1 - r2) * (n - 1) / max(1, n - p - 1)

    # F-статистика
    ms_reg = (ss_tot - ss_res) / max(1, p)
    ms_res = ss_res / max(1, n - p - 1)
    f_stat = ms_reg / ms_res if ms_res > 0 else 0
    f_p = 1 - sp_stats.f.cdf(f_stat, p, max(1, n - p - 1))

    coefficients = {"Свободный член": float(beta[0])}
    for i, idx in enumerate(predictor_indices):
        coefficients[FEATURE_NAMES[idx]] = float(beta[i + 1])

    return RegressionResult(
        predictors=[FEATURE_NAMES[i] for i in predictor_indices],
        response=FEATURE_NAMES[response_idx],
        coefficients=coefficients,
        r_squared=float(r2),
        adj_r_squared=float(adj_r2),
        f_statistic=float(f_stat),
        f_p_value=float(f_p),
        residual_std=float(np.sqrt(ms_res)),
    )


# ═══════════════════════════════════════════════════════════════════════════
# 5. АНАЛИЗ РАСПРЕДЕЛЕНИЙ
# ═══════════════════════════════════════════════════════════════════════════
@dataclass
class DistributionFit:
    """Результат подгонки распределения."""
    feature_name: str
    # Нормальность
    shapiro_stat: float
    shapiro_p: float
    is_normal: bool
    # Вейбулл
    weibull_shape: float    # k
    weibull_scale: float    # λ
    weibull_ks_stat: float
    weibull_ks_p: float


def fit_distributions(cb: CaseBase,
                      feature_indices: Optional[List[int]] = None
                      ) -> List[DistributionFit]:
    """Подгонка распределений для указанных признаков."""
    if cb._feature_matrix is None:
        cb._build_matrix()
    X = cb._feature_matrix
    if feature_indices is None:
        feature_indices = [2, 3, 5, 9]  # площадь, ранг, пенные атаки, длит.

    results = []
    for j in feature_indices:
        col = X[:, j]
        col_pos = col[col > 0]  # Вейбулл — только положительные
        n = len(col)
        if n < 3:
            continue

        # Шапиро-Уилк (нормальность)
        if n <= 5000:
            sw_stat, sw_p = sp_stats.shapiro(col)
        else:
            sw_stat, sw_p = 0, 0

        # Вейбулл (для положительных данных)
        w_shape, w_loc, w_scale = 1.0, 0.0, 1.0
        ks_stat, ks_p = 0.0, 0.0
        if len(col_pos) >= 3:
            try:
                w_shape, w_loc, w_scale = sp_stats.weibull_min.fit(
                    col_pos, floc=0)
                ks_stat, ks_p = sp_stats.kstest(
                    col_pos, 'weibull_min', args=(w_shape, 0, w_scale))
            except Exception:
                pass

        results.append(DistributionFit(
            feature_name=FEATURE_NAMES[j],
            shapiro_stat=float(sw_stat),
            shapiro_p=float(sw_p),
            is_normal=sw_p > 0.05,
            weibull_shape=float(w_shape),
            weibull_scale=float(w_scale),
            weibull_ks_stat=float(ks_stat),
            weibull_ks_p=float(ks_p),
        ))
    return results


# ═══════════════════════════════════════════════════════════════════════════
# 6. ВИЗУАЛИЗАЦИИ
# ═══════════════════════════════════════════════════════════════════════════
def plot_correlation_matrix(pearson_m: np.ndarray,
                            filename: str = "stat_correlation.png") -> str:
    """Тепловая карта корреляций."""
    n = min(pearson_m.shape[0], N_FEATURES)
    fig, ax = plt.subplots(figsize=(10, 8), facecolor="#f5f6fa")
    im = ax.imshow(pearson_m[:n, :n], cmap="RdBu_r", vmin=-1, vmax=1)

    short_names = [name.split("(")[0].strip()[:15] for name in FEATURE_NAMES[:n]]
    ax.set_xticks(range(n))
    ax.set_xticklabels(short_names, rotation=45, ha="right", fontsize=7)
    ax.set_yticks(range(n))
    ax.set_yticklabels(short_names, fontsize=7)

    for i in range(n):
        for j in range(n):
            v = pearson_m[i, j]
            ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                    fontsize=6, color="white" if abs(v) > 0.5 else "black")

    fig.colorbar(im, ax=ax, shrink=0.8, label="Коэффициент Пирсона")
    ax.set_title("Матрица корреляций признаков пожаров",
                 fontsize=12, fontweight="bold")
    fig.tight_layout()
    return _save(fig, filename)


def plot_distributions(cb: CaseBase,
                       feature_indices: List[int] = None,
                       filename: str = "stat_distributions.png") -> str:
    """Гистограммы распределений ключевых признаков."""
    if cb._feature_matrix is None:
        cb._build_matrix()
    if feature_indices is None:
        feature_indices = [0, 2, 3, 5, 7, 9]
    X = cb._feature_matrix

    n_plots = len(feature_indices)
    ncols = 3
    nrows = (n_plots + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(12, 3.5 * nrows),
                             facecolor="#f5f6fa")
    axes = np.array(axes).flatten()

    for idx, (ax, fi) in enumerate(zip(axes, feature_indices)):
        col = X[:, fi]
        ax.hist(col, bins=min(20, len(col) // 3 + 1), color="#3498db",
                alpha=0.7, edgecolor="white")
        ax.set_title(FEATURE_NAMES[fi], fontsize=9, fontweight="bold")
        ax.set_ylabel("Частота", fontsize=8)
        ax.axvline(np.mean(col), color="#c0392b", linewidth=1.5,
                   linestyle="--", label=f"M={np.mean(col):.1f}")
        ax.axvline(np.median(col), color="#27ae60", linewidth=1.5,
                   linestyle=":", label=f"Me={np.median(col):.1f}")
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

    for ax in axes[len(feature_indices):]:
        ax.set_visible(False)

    fig.suptitle("Распределения признаков пожаров", fontsize=12,
                 fontweight="bold")
    fig.tight_layout()
    return _save(fig, filename)


def plot_scatter_matrix(cb: CaseBase,
                        feature_indices: List[int] = None,
                        filename: str = "stat_scatter.png") -> str:
    """Матрица диаграмм рассеяния."""
    if cb._feature_matrix is None:
        cb._build_matrix()
    if feature_indices is None:
        feature_indices = [0, 2, 3, 9]  # объём, площадь, ранг, длительность
    X = cb._feature_matrix
    n = len(feature_indices)

    fig, axes = plt.subplots(n, n, figsize=(3 * n, 3 * n), facecolor="#f5f6fa")
    short_names = [FEATURE_NAMES[i].split("(")[0].strip()[:12]
                   for i in feature_indices]

    for i in range(n):
        for j in range(n):
            ax = axes[i][j]
            xi = X[:, feature_indices[j]]
            yi = X[:, feature_indices[i]]
            if i == j:
                ax.hist(xi, bins=15, color="#3498db", alpha=0.7, edgecolor="white")
            else:
                ax.scatter(xi, yi, s=10, alpha=0.5, color="#2c3e50")
                # Линия тренда
                if len(xi) > 2:
                    z = np.polyfit(xi, yi, 1)
                    x_line = np.linspace(xi.min(), xi.max(), 50)
                    ax.plot(x_line, np.polyval(z, x_line), color="#c0392b",
                            linewidth=1, alpha=0.7)
            if i == n - 1:
                ax.set_xlabel(short_names[j], fontsize=8)
            if j == 0:
                ax.set_ylabel(short_names[i], fontsize=8)
            ax.tick_params(labelsize=6)

    fig.suptitle("Матрица рассеяния ключевых признаков",
                 fontsize=12, fontweight="bold")
    fig.tight_layout()
    return _save(fig, filename)


def plot_anova_boxplot(cb: CaseBase,
                       factor_idx: int = 3,   # ранг
                       response_idx: int = 9,  # длительность
                       filename: str = "stat_anova.png") -> str:
    """Ящик с усами для дисперсионного анализа."""
    if cb._feature_matrix is None:
        cb._build_matrix()
    X = cb._feature_matrix
    factor = X[:, factor_idx]
    response = X[:, response_idx]

    groups = {}
    for i, f in enumerate(factor):
        key = f"{f:.0f}"
        groups.setdefault(key, []).append(response[i])

    fig, ax = plt.subplots(figsize=(8, 5), facecolor="#f5f6fa")
    labels = sorted(groups.keys())
    data = [groups[k] for k in labels]

    bp = ax.boxplot(data, labels=labels, patch_artist=True,
                    medianprops=dict(color="#c0392b", linewidth=2))
    colors = ["#3498db", "#27ae60", "#e67e22", "#9b59b6", "#1abc9c"]
    for patch, color in zip(bp["boxes"], colors * 3):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)

    ax.set_xlabel(FEATURE_NAMES[factor_idx], fontsize=10)
    ax.set_ylabel(FEATURE_NAMES[response_idx], fontsize=10)
    ax.set_title(f"Дисперсионный анализ: {FEATURE_NAMES[factor_idx]} → "
                 f"{FEATURE_NAMES[response_idx]}",
                 fontsize=11, fontweight="bold")
    ax.grid(True, axis="y", alpha=0.3)

    # Добавить результат ANOVA
    result = anova_by_factor(cb, factor_idx, response_idx)
    ax.text(0.98, 0.98,
            f"{result.method}\n"
            f"H={result.statistic:.2f}, p={result.p_value:.4f}\n"
            f"\u03b7\u00b2={result.eta_squared:.3f} "
            f"({'значимо' if result.significant else 'не значимо'})",
            transform=ax.transAxes, fontsize=8, va="top", ha="right",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))

    fig.tight_layout()
    return _save(fig, filename)


# ═══════════════════════════════════════════════════════════════════════════
# ПОЛНЫЙ ОТЧЁТ
# ═══════════════════════════════════════════════════════════════════════════
def full_analysis(cb: CaseBase) -> Dict:
    """Провести полный статистический анализ базы прецедентов.

    Возвращает: словарь со всеми результатами + пути к визуализациям.
    """
    results = {"n_cases": len(cb)}

    # 1. Описательная
    desc = descriptive_statistics(cb)
    results["descriptive"] = [
        {"name": d.name, "n": d.n, "M": round(d.mean, 2),
         "SD": round(d.std, 2), "Me": round(d.median, 2),
         "CI95": f"[{d.ci95_low:.2f}; {d.ci95_high:.2f}]",
         "Асимметрия": round(d.skewness, 2),
         "Эксцесс": round(d.kurtosis, 2)}
        for d in desc
    ]

    # 2. Корреляции
    pearson_m, spearman_m, pairs = correlation_matrix(cb)
    sig_pairs = [p for p in pairs if p.significant]
    results["significant_correlations"] = [
        {"a": p.feature_a, "b": p.feature_b,
         "r": round(p.pearson_r, 3), "p": round(p.pearson_p, 4),
         "rho": round(p.spearman_rho, 3)}
        for p in sorted(sig_pairs, key=lambda x: -abs(x.pearson_r))[:10]
    ]

    # 3. ANOVA: ранг → длительность, ранг → площадь
    results["anova"] = []
    for factor, response in [(3, 9), (3, 2), (10, 9)]:
        a = anova_by_factor(cb, factor, response)
        results["anova"].append({
            "фактор": a.factor_name, "отклик": a.response_name,
            "метод": a.method, "H/F": round(a.statistic, 2),
            "p": round(a.p_value, 4), "η²": round(a.eta_squared, 3),
            "значимо": a.significant,
        })

    # 4. Регрессия: объём + ранг + препятствие → длительность
    reg = multiple_regression(cb, [0, 3, 4], 9)
    results["regression"] = {
        "отклик": reg.response,
        "предикторы": reg.predictors,
        "R²": round(reg.r_squared, 3),
        "R²_adj": round(reg.adj_r_squared, 3),
        "F": round(reg.f_statistic, 2),
        "p(F)": round(reg.f_p_value, 4),
        "коэффициенты": {k: round(v, 4) for k, v in reg.coefficients.items()},
    }

    # 5. Распределения
    dist_fits = fit_distributions(cb)
    results["distributions"] = [
        {"признак": d.feature_name,
         "нормальное": d.is_normal,
         "Шапиро p": round(d.shapiro_p, 4),
         "Вейбулл k": round(d.weibull_shape, 2),
         "Вейбулл λ": round(d.weibull_scale, 2)}
        for d in dist_fits
    ]

    # 6. Визуализации
    results["figures"] = {}
    results["figures"]["correlation"] = plot_correlation_matrix(pearson_m)
    results["figures"]["distributions"] = plot_distributions(cb)
    results["figures"]["scatter"] = plot_scatter_matrix(cb)
    results["figures"]["anova"] = plot_anova_boxplot(cb)

    return results


if __name__ == "__main__":
    from cbr_engine import generate_demo_casebase

    cb = generate_demo_casebase(100)
    print(f"Анализ базы: {len(cb)} случаев\n")

    results = full_analysis(cb)

    print("Описательная статистика (топ-5):")
    for d in results["descriptive"][:5]:
        print(f"  {d['name'][:30]:<30s}  M={d['M']:>10.1f}  SD={d['SD']:>8.1f}  {d['CI95']}")

    print(f"\nЗначимые корреляции (топ-5):")
    for c in results["significant_correlations"][:5]:
        print(f"  {c['a'][:20]} — {c['b'][:20]}: r={c['r']:+.3f} (p={c['p']:.4f})")

    print(f"\nДисперсионный анализ:")
    for a in results["anova"]:
        sig = "✓" if a["значимо"] else "✗"
        print(f"  {a['фактор'][:15]} → {a['отклик'][:15]}: "
              f"p={a['p']:.4f} η²={a['η²']:.3f} {sig}")

    print(f"\nРегрессия: {results['regression']['отклик']}")
    print(f"  R²={results['regression']['R²']:.3f}, "
          f"F={results['regression']['F']:.2f}, "
          f"p={results['regression']['p(F)']:.4f}")
    for k, v in results["regression"]["коэффициенты"].items():
        print(f"  {k}: β={v:+.4f}")

    print(f"\nВизуализации:")
    for name, path in results["figures"].items():
        print(f"  {name}: {path}")
