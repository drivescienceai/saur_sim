"""
timeseries.py — Анализ временных рядов и обнаружение точек разладки.
═══════════════════════════════════════════════════════════════════════════════
Методы:
  1. Автокорреляционный анализ (ACF, PACF)
  2. Скользящие статистики (среднее, дисперсия, тренд)
  3. Обнаружение точек разладки (CUSUM, Байесовский)
  4. Спектральный анализ (периодограмма)
  5. Байесовская оценка неопределённости параметров

Применение к данным пожара:
  - h_risk  → точки разладки = моменты необходимости адаптации
  - h_fire  → тренд площади = скорость распространения
  - h_water → стабильность водоснабжения
═══════════════════════════════════════════════════════════════════════════════
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional

import numpy as np
from scipy import stats as sp_stats

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "data", "figures")
os.makedirs(_OUT_DIR, exist_ok=True)


def _save(fig, name):
    path = os.path.join(_OUT_DIR, name)
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    return path


# ═══════════════════════════════════════════════════════════════════════════
# 1. АВТОКОРРЕЛЯЦИЯ
# ═══════════════════════════════════════════════════════════════════════════
def acf(x: np.ndarray, max_lag: int = 40) -> np.ndarray:
    """Автокорреляционная функция (ACF)."""
    n = len(x)
    x_centered = x - x.mean()
    var = np.sum(x_centered ** 2)
    if var < 1e-12:
        return np.zeros(min(max_lag, n - 1))
    lags = min(max_lag, n - 1)
    result = np.zeros(lags)
    for k in range(lags):
        result[k] = np.sum(x_centered[:n - k] * x_centered[k:]) / var
    return result


def pacf(x: np.ndarray, max_lag: int = 20) -> np.ndarray:
    """Частная автокорреляция (PACF) через рекуррентное соотношение."""
    acf_vals = acf(x, max_lag)
    n_lags = len(acf_vals)
    result = np.zeros(n_lags)
    result[0] = acf_vals[0]
    phi = np.zeros((n_lags, n_lags))
    phi[0, 0] = acf_vals[0]
    for k in range(1, n_lags):
        num = acf_vals[k] - sum(phi[k - 1, j] * acf_vals[k - 1 - j]
                                 for j in range(k))
        den = 1.0 - sum(phi[k - 1, j] * acf_vals[j] for j in range(k))
        if abs(den) < 1e-12:
            break
        phi[k, k] = num / den
        for j in range(k):
            phi[k, j] = phi[k - 1, j] - phi[k, k] * phi[k - 1, k - 1 - j]
        result[k] = phi[k, k]
    return result


# ═══════════════════════════════════════════════════════════════════════════
# 2. СКОЛЬЗЯЩИЕ СТАТИСТИКИ
# ═══════════════════════════════════════════════════════════════════════════
def rolling_stats(x: np.ndarray, window: int = 10
                  ) -> Dict[str, np.ndarray]:
    """Скользящее среднее, дисперсия, линейный тренд."""
    n = len(x)
    if n < window:
        return {"mean": x.copy(), "var": np.zeros(n), "trend": np.zeros(n)}
    ma = np.convolve(x, np.ones(window) / window, mode="valid")
    mv = np.array([np.var(x[max(0, i - window):i + 1])
                   for i in range(n)])
    # Линейный тренд (наклон OLS в скользящем окне)
    trend = np.zeros(n)
    for i in range(window, n):
        chunk = x[i - window:i + 1]
        t = np.arange(len(chunk))
        if np.std(chunk) > 1e-12:
            slope = np.polyfit(t, chunk, 1)[0]
        else:
            slope = 0.0
        trend[i] = slope
    return {"mean": ma, "var": mv, "trend": trend}


# ═══════════════════════════════════════════════════════════════════════════
# 3. ОБНАРУЖЕНИЕ ТОЧЕК РАЗЛАДКИ
# ═══════════════════════════════════════════════════════════════════════════
@dataclass
class ChangePoint:
    """Обнаруженная точка разладки."""
    index: int
    time: float          # время (мин), если известно
    method: str          # "CUSUM" / "Bayesian" / "Variance"
    score: float         # значимость
    direction: str       # "increase" / "decrease"


def cusum_detect(x: np.ndarray, threshold: float = 5.0,
                 drift: float = 0.0,
                 times: Optional[np.ndarray] = None
                 ) -> List[ChangePoint]:
    """Обнаружение точек разладки методом CUSUM.

    CUSUM (Cumulative Sum): S_t = max(0, S_{t-1} + (x_t - μ - drift))
    Разладка: S_t > threshold.
    """
    n = len(x)
    if n < 3:
        return []
    mu = np.mean(x)
    S_pos = np.zeros(n)  # для увеличения
    S_neg = np.zeros(n)  # для уменьшения
    changes = []

    for t in range(1, n):
        S_pos[t] = max(0, S_pos[t - 1] + (x[t] - mu) - drift)
        S_neg[t] = max(0, S_neg[t - 1] - (x[t] - mu) - drift)

        if S_pos[t] > threshold:
            time_val = float(times[t]) if times is not None else float(t)
            changes.append(ChangePoint(
                index=t, time=time_val, method="CUSUM",
                score=float(S_pos[t]), direction="increase"))
            S_pos[t] = 0  # сбросить после обнаружения

        if S_neg[t] > threshold:
            time_val = float(times[t]) if times is not None else float(t)
            changes.append(ChangePoint(
                index=t, time=time_val, method="CUSUM",
                score=float(S_neg[t]), direction="decrease"))
            S_neg[t] = 0

    return changes


def bayesian_changepoint(x: np.ndarray, prior_prob: float = 0.01,
                         times: Optional[np.ndarray] = None
                         ) -> List[ChangePoint]:
    """Байесовское обнаружение точек разладки (online).

    Модель: P(разладка в t) = prior_prob (a priori).
    Вычисляет апостериорную вероятность разладки для каждого t.
    """
    n = len(x)
    if n < 5:
        return []

    # Упрощённая модель: разладка = значительное изменение среднего
    window = max(3, n // 20)
    posterior = np.zeros(n)
    changes = []

    for t in range(window, n - window):
        left = x[t - window:t]
        right = x[t:t + window]
        # Байес-фактор: отношение правдоподобия (разные средние vs одно среднее)
        mu_l, mu_r = np.mean(left), np.mean(right)
        sigma = max(np.std(x), 1e-8)
        log_bf = (window / (2 * sigma ** 2)) * (mu_l - mu_r) ** 2
        # Апостериорная вероятность
        posterior[t] = 1.0 / (1.0 + (1 - prior_prob) / prior_prob
                              * np.exp(-log_bf))

    # Найти пики апостериорной вероятности
    for t in range(1, n - 1):
        if (posterior[t] > 0.5 and
                posterior[t] > posterior[t - 1] and
                posterior[t] >= posterior[t + 1]):
            time_val = float(times[t]) if times is not None else float(t)
            direction = ("increase" if t < n - 1 and x[t + 1] > x[t - 1]
                         else "decrease")
            changes.append(ChangePoint(
                index=t, time=time_val, method="Bayesian",
                score=float(posterior[t]), direction=direction))

    return changes


def variance_changepoint(x: np.ndarray, window: int = 10,
                         threshold: float = 3.0,
                         times: Optional[np.ndarray] = None
                         ) -> List[ChangePoint]:
    """Обнаружение изменения дисперсии (F-тест скользящих окон)."""
    n = len(x)
    if n < 2 * window:
        return []
    changes = []
    for t in range(window, n - window):
        var_left = np.var(x[t - window:t], ddof=1)
        var_right = np.var(x[t:t + window], ddof=1)
        if var_left < 1e-12 or var_right < 1e-12:
            continue
        f_ratio = max(var_left, var_right) / min(var_left, var_right)
        if f_ratio > threshold:
            time_val = float(times[t]) if times is not None else float(t)
            changes.append(ChangePoint(
                index=t, time=time_val, method="Variance",
                score=float(f_ratio), direction="increase"))
    return changes


# ═══════════════════════════════════════════════════════════════════════════
# 4. СПЕКТРАЛЬНЫЙ АНАЛИЗ
# ═══════════════════════════════════════════════════════════════════════════
def periodogram(x: np.ndarray, dt: float = 5.0) -> Tuple[np.ndarray, np.ndarray]:
    """Периодограмма (оценка спектральной плотности мощности).

    dt: шаг дискретизации (мин).
    Возвращает: (frequencies_Hz, power)
    """
    n = len(x)
    x_centered = x - x.mean()
    fft_vals = np.fft.rfft(x_centered)
    power = (np.abs(fft_vals) ** 2) / n
    freqs = np.fft.rfftfreq(n, d=dt)
    return freqs, power


# ═══════════════════════════════════════════════════════════════════════════
# 5. БАЙЕСОВСКАЯ ОЦЕНКА НЕОПРЕДЕЛЁННОСТИ
# ═══════════════════════════════════════════════════════════════════════════
@dataclass
class BayesianEstimate:
    """Байесовская оценка параметра."""
    param_name: str
    posterior_mean: float
    posterior_std: float
    ci95: Tuple[float, float]
    prior_mean: float
    prior_std: float
    n_observations: int


def bayesian_parameter_estimate(
    observations: np.ndarray,
    prior_mean: float = 0.0,
    prior_std: float = 1.0,
    param_name: str = "параметр",
) -> BayesianEstimate:
    """Байесовская оценка среднего с нормальным сопряжённым приором.

    Posterior: N(μ_post, σ²_post)
      μ_post = (prior_mean/prior_std² + n·x̄/σ²_obs) / (1/prior_std² + n/σ²_obs)
      σ²_post = 1 / (1/prior_std² + n/σ²_obs)
    """
    n = len(observations)
    if n == 0:
        return BayesianEstimate(param_name=param_name,
                                posterior_mean=prior_mean,
                                posterior_std=prior_std,
                                ci95=(prior_mean - 1.96 * prior_std,
                                      prior_mean + 1.96 * prior_std),
                                prior_mean=prior_mean,
                                prior_std=prior_std,
                                n_observations=0)

    x_bar = float(np.mean(observations))
    sigma_obs = float(np.std(observations, ddof=1)) if n > 1 else prior_std

    prior_prec = 1.0 / (prior_std ** 2)
    obs_prec = n / (sigma_obs ** 2) if sigma_obs > 1e-12 else 0.0
    post_prec = prior_prec + obs_prec
    post_var = 1.0 / post_prec
    post_mean = (prior_mean * prior_prec + x_bar * obs_prec) / post_prec
    post_std = float(np.sqrt(post_var))

    return BayesianEstimate(
        param_name=param_name,
        posterior_mean=post_mean,
        posterior_std=post_std,
        ci95=(post_mean - 1.96 * post_std, post_mean + 1.96 * post_std),
        prior_mean=prior_mean,
        prior_std=prior_std,
        n_observations=n,
    )


# ═══════════════════════════════════════════════════════════════════════════
# 6. КОМПЛЕКСНЫЙ АНАЛИЗ ВРЕМЕННОГО РЯДА ПОЖАРА
# ═══════════════════════════════════════════════════════════════════════════
def analyze_fire_timeseries(
    h_risk: List[Tuple[int, float]],
    h_fire: List[Tuple[int, float]],
    h_water: List[Tuple[int, float]],
) -> Dict:
    """Полный анализ временных рядов одного эпизода пожара.

    Входные данные из TankFireSim: h_risk, h_fire, h_water.
    """
    results = {}

    for name, data in [("risk", h_risk), ("fire", h_fire), ("water", h_water)]:
        if not data:
            continue
        times = np.array([t for t, v in data], dtype=float)
        values = np.array([v for t, v in data], dtype=float)

        # Скользящие статистики
        window = max(3, len(values) // 15)
        rs = rolling_stats(values, window)

        # Точки разладки
        cusum_th = 3.0 * np.std(values) if np.std(values) > 0 else 5.0
        cp_cusum = cusum_detect(values, threshold=cusum_th, times=times)
        cp_bayes = bayesian_changepoint(values, times=times)

        # ACF
        acf_vals = acf(values, max_lag=min(30, len(values) // 3))

        # Байесовская оценка среднего
        bayes_est = bayesian_parameter_estimate(
            values, prior_mean=float(np.mean(values)),
            prior_std=float(np.std(values) + 0.01),
            param_name=f"среднее {name}")

        results[name] = {
            "n": len(values),
            "mean": float(np.mean(values)),
            "std": float(np.std(values)),
            "trend_slope": float(rs["trend"][-1]) if len(rs["trend"]) > 0 else 0,
            "changepoints_cusum": len(cp_cusum),
            "changepoints_bayes": len(cp_bayes),
            "changepoints": [{"t": cp.time, "method": cp.method,
                              "score": cp.score, "direction": cp.direction}
                             for cp in cp_cusum + cp_bayes],
            "acf_lag1": float(acf_vals[0]) if len(acf_vals) > 0 else 0,
            "bayesian_mean": bayes_est.posterior_mean,
            "bayesian_ci95": bayes_est.ci95,
        }

    return results


# ═══════════════════════════════════════════════════════════════════════════
# ВИЗУАЛИЗАЦИИ
# ═══════════════════════════════════════════════════════════════════════════
def plot_changepoint_analysis(
    times: np.ndarray, values: np.ndarray,
    changepoints: List[ChangePoint],
    series_name: str = "Индекс риска",
    filename: str = "ts_changepoints.png",
) -> str:
    """Временной ряд с отмеченными точками разладки."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 6),
                                    facecolor="#f5f6fa", sharex=True,
                                    gridspec_kw={"height_ratios": [3, 1]})

    ax1.plot(times, values, color="#2c3e50", linewidth=1, label=series_name)

    # Скользящее среднее
    w = max(3, len(values) // 15)
    if len(values) > w:
        ma = np.convolve(values, np.ones(w) / w, mode="valid")
        ax1.plot(times[w - 1:], ma, color="#3498db", linewidth=2,
                 alpha=0.7, label=f"Скользящее среднее (окно {w})")

    # Точки разладки
    colors_cp = {"CUSUM": "#c0392b", "Bayesian": "#e67e22", "Variance": "#8e44ad"}
    for cp in changepoints:
        color = colors_cp.get(cp.method, "#c0392b")
        ax1.axvline(cp.time, color=color, linestyle="--", alpha=0.7, linewidth=1.5)
        label = f"{cp.method} (t={cp.time:.0f})"
        ax1.annotate(label, xy=(cp.time, values[min(cp.index, len(values) - 1)]),
                     xytext=(10, 15), textcoords="offset points",
                     fontsize=7, color=color,
                     arrowprops=dict(arrowstyle="->", color=color, alpha=0.5))

    ax1.set_ylabel(series_name, fontsize=10)
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)
    ax1.set_title(f"Обнаружение точек разладки: {series_name}",
                  fontsize=12, fontweight="bold")

    # Нижний график: скользящая дисперсия
    rs = rolling_stats(values, w)
    ax2.fill_between(range(len(rs["var"])), rs["var"],
                     color="#e74c3c", alpha=0.3)
    ax2.plot(rs["var"], color="#e74c3c", linewidth=1)
    ax2.set_ylabel("Дисперсия", fontsize=9)
    ax2.set_xlabel("Время (мин)", fontsize=10)
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    return _save(fig, filename)


def plot_acf_pacf(values: np.ndarray, series_name: str = "Ряд",
                  filename: str = "ts_acf.png") -> str:
    """ACF и PACF."""
    max_lag = min(30, len(values) // 3)
    acf_vals = acf(values, max_lag)
    pacf_vals = pacf(values, min(max_lag, 20))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4), facecolor="#f5f6fa")

    ci = 1.96 / np.sqrt(len(values))

    ax1.bar(range(len(acf_vals)), acf_vals, color="#3498db", width=0.4)
    ax1.axhline(ci, color="#c0392b", linestyle="--", linewidth=0.8)
    ax1.axhline(-ci, color="#c0392b", linestyle="--", linewidth=0.8)
    ax1.set_title(f"ACF: {series_name}", fontweight="bold")
    ax1.set_xlabel("Лаг")
    ax1.grid(True, alpha=0.3)

    ax2.bar(range(len(pacf_vals)), pacf_vals, color="#27ae60", width=0.4)
    ax2.axhline(ci, color="#c0392b", linestyle="--", linewidth=0.8)
    ax2.axhline(-ci, color="#c0392b", linestyle="--", linewidth=0.8)
    ax2.set_title(f"PACF: {series_name}", fontweight="bold")
    ax2.set_xlabel("Лаг")
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    return _save(fig, filename)


def plot_bayesian_posterior(estimate: BayesianEstimate,
                           filename: str = "ts_bayesian.png") -> str:
    """Визуализация апостериорного распределения."""
    fig, ax = plt.subplots(figsize=(8, 5), facecolor="#f5f6fa")

    # Приор
    x = np.linspace(estimate.prior_mean - 4 * estimate.prior_std,
                    estimate.prior_mean + 4 * estimate.prior_std, 200)
    prior_pdf = sp_stats.norm.pdf(x, estimate.prior_mean, estimate.prior_std)
    ax.plot(x, prior_pdf, "--", color="#3498db", linewidth=1.5,
            label=f"Априорное: N({estimate.prior_mean:.2f}, {estimate.prior_std:.2f})")
    ax.fill_between(x, prior_pdf, alpha=0.1, color="#3498db")

    # Апостериорное
    x2 = np.linspace(estimate.posterior_mean - 4 * estimate.posterior_std,
                     estimate.posterior_mean + 4 * estimate.posterior_std, 200)
    post_pdf = sp_stats.norm.pdf(x2, estimate.posterior_mean,
                                  estimate.posterior_std)
    ax.plot(x2, post_pdf, "-", color="#c0392b", linewidth=2,
            label=f"Апостериорное: N({estimate.posterior_mean:.2f}, "
                  f"{estimate.posterior_std:.2f})")
    ax.fill_between(x2, post_pdf, alpha=0.2, color="#c0392b")

    # 95% ДИ
    ax.axvline(estimate.ci95[0], color="#e67e22", linestyle=":", linewidth=1,
               label=f"95% ДИ: [{estimate.ci95[0]:.2f}; {estimate.ci95[1]:.2f}]")
    ax.axvline(estimate.ci95[1], color="#e67e22", linestyle=":", linewidth=1)

    ax.set_xlabel(estimate.param_name, fontsize=10)
    ax.set_ylabel("Плотность вероятности", fontsize=10)
    ax.set_title(f"Байесовская оценка: {estimate.param_name} "
                 f"(n={estimate.n_observations})",
                 fontsize=12, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return _save(fig, filename)
