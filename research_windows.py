"""
research_windows.py — Полноценные интерактивные окна исследовательских блоков.
═══════════════════════════════════════════════════════════════════════════════
4 окна с вкладками, настройками, графиками и таблицами:
  1. StatWindow       — Статистика и анализ данных
  2. TimeseriesWindow — Анализ временных рядов
  3. MarkovWindow     — Марковские модели
  4. ExpertCBRWindow  — Экспертная система и прецеденты
═══════════════════════════════════════════════════════════════════════════════
"""
from __future__ import annotations

import os
import threading
import tkinter as tk
from tkinter import ttk, scrolledtext

import numpy as np

try:
    import matplotlib
    matplotlib.use("TkAgg")
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
    from matplotlib.figure import Figure
    _MPL = True
except Exception:
    _MPL = False

# Цвета (светлая тема)
BG = "#f5f6fa"
PNL = "#ffffff"
PNL2 = "#eef0f4"
TXT = "#2c3e50"
TXT2 = "#7f8c8d"
ACC = "#c0392b"
GRD = "#d5d8dc"


def _embed_fig(parent, fig) -> FigureCanvasTkAgg:
    canvas = FigureCanvasTkAgg(fig, master=parent)
    canvas.draw()
    canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
    return canvas


def _status_bar(parent) -> tk.StringVar:
    var = tk.StringVar(value="Готов")
    tk.Label(parent, textvariable=var, bg=PNL2, fg=TXT2,
             font=("Consolas", 8), anchor="w").pack(fill="x", side="bottom")
    return var


def _results_table(parent, headers, rows):
    """Создать таблицу результатов."""
    tree = ttk.Treeview(parent, columns=headers, show="headings", height=12)
    style = ttk.Style()
    style.configure("Treeview", background=PNL, foreground=TXT,
                    fieldbackground=PNL, rowheight=24, font=("Arial", 8))
    style.configure("Treeview.Heading", background=PNL2, foreground=ACC,
                    font=("Arial", 8, "bold"))
    for h in headers:
        tree.heading(h, text=h)
        tree.column(h, width=100, anchor="center")
    for row in rows:
        tree.insert("", "end", values=row)
    sb = ttk.Scrollbar(parent, orient="vertical", command=tree.yview)
    tree.configure(yscrollcommand=sb.set)
    tree.pack(side="left", fill="both", expand=True, padx=4, pady=4)
    sb.pack(side="right", fill="y")
    return tree


# ═══════════════════════════════════════════════════════════════════════════
# 1. СТАТИСТИКА И АНАЛИЗ ДАННЫХ
# ═══════════════════════════════════════════════════════════════════════════
class StatWindow(tk.Toplevel):
    """Окно статистического анализа."""

    def __init__(self, master=None):
        super().__init__(master)
        self.title("САУР-ПСП — Статистика и анализ данных")
        self.geometry("1100x700")
        self.configure(bg=BG)

        self._status = _status_bar(self)
        self._cb = None
        self._build()
        self._load_data()

    def _build(self):
        nb = ttk.Notebook(self)
        nb.pack(fill="both", expand=True, padx=6, pady=6)

        style = ttk.Style()
        style.configure("TNotebook", background=BG)
        style.configure("TNotebook.Tab", padding=(10, 4), font=("Arial", 9))

        # Вкладки
        self._tab_desc = tk.Frame(nb, bg=BG)
        self._tab_corr = tk.Frame(nb, bg=BG)
        self._tab_anova = tk.Frame(nb, bg=BG)
        self._tab_reg = tk.Frame(nb, bg=BG)
        self._tab_dist = tk.Frame(nb, bg=BG)
        self._tab_cfg = tk.Frame(nb, bg=BG)

        nb.add(self._tab_desc, text="  Описательная  ")
        nb.add(self._tab_corr, text="  Корреляции  ")
        nb.add(self._tab_anova, text="  Дисперсионный  ")
        nb.add(self._tab_reg, text="  Регрессия  ")
        nb.add(self._tab_dist, text="  Распределения  ")
        nb.add(self._tab_cfg, text="  Настройки  ")

        # Настройки
        cfg = self._tab_cfg
        tk.Label(cfg, text="Размер выборки (демо):", bg=BG, fg=TXT,
                 font=("Arial", 9)).pack(anchor="w", padx=10, pady=(10, 2))
        self._n_var = tk.IntVar(value=100)
        tk.Scale(cfg, from_=20, to=500, orient="horizontal",
                 variable=self._n_var, bg=BG, fg=TXT,
                 troughcolor=GRD, length=300).pack(padx=10)
        tk.Button(cfg, text="Пересчитать анализ",
                  command=self._load_data, bg=ACC, fg="white",
                  font=("Arial", 10, "bold"), relief="flat",
                  padx=16, pady=6).pack(pady=10)

    def _load_data(self):
        self._status.set("Загрузка данных...")
        self.update_idletasks()

        def _run():
            try:
                from cbr_engine import generate_demo_casebase
                from stat_analysis import (descriptive_statistics,
                    correlation_matrix, anova_by_factor, multiple_regression,
                    fit_distributions)

                n = self._n_var.get()
                self._cb = generate_demo_casebase(n)
                cb = self._cb

                desc = descriptive_statistics(cb)
                pearson_m, spearman_m, pairs = correlation_matrix(cb)
                anova_res = anova_by_factor(cb, 3, 9)
                reg_res = multiple_regression(cb, [0, 3, 4], 9)
                dist_res = fit_distributions(cb)

                self.after(0, lambda: self._show_descriptive(desc))
                self.after(0, lambda: self._show_correlations(pearson_m, pairs))
                self.after(0, lambda: self._show_anova(anova_res))
                self.after(0, lambda: self._show_regression(reg_res))
                self.after(0, lambda: self._show_distributions(cb, dist_res))
                self.after(0, lambda: self._status.set(
                    f"Анализ завершён ({n} случаев)"))
            except Exception as e:
                self.after(0, lambda: self._status.set(f"Ошибка: {e}"))

        threading.Thread(target=_run, daemon=True).start()

    def _show_descriptive(self, desc):
        for w in self._tab_desc.winfo_children():
            w.destroy()
        headers = ["Признак", "n", "M", "SD", "Me", "Q25", "Q75",
                   "95% ДИ нижн.", "95% ДИ верхн.", "Асимметрия"]
        rows = [(d.name[:25], d.n, f"{d.mean:.1f}", f"{d.std:.1f}",
                 f"{d.median:.1f}", f"{d.q25:.1f}", f"{d.q75:.1f}",
                 f"{d.ci95_low:.1f}", f"{d.ci95_high:.1f}",
                 f"{d.skewness:.2f}")
                for d in desc]
        _results_table(self._tab_desc, headers, rows)

    def _show_correlations(self, pearson_m, pairs):
        for w in self._tab_corr.winfo_children():
            w.destroy()
        if not _MPL:
            return
        fig = Figure(figsize=(8, 6), facecolor=BG)
        ax = fig.add_subplot(111)
        n = min(pearson_m.shape[0], 12)
        from cbr_engine import FEATURE_NAMES
        short = [name.split("(")[0].strip()[:12] for name in FEATURE_NAMES[:n]]
        im = ax.imshow(pearson_m[:n, :n], cmap="RdBu_r", vmin=-1, vmax=1)
        ax.set_xticks(range(n))
        ax.set_xticklabels(short, rotation=45, ha="right", fontsize=7)
        ax.set_yticks(range(n))
        ax.set_yticklabels(short, fontsize=7)
        for i in range(n):
            for j in range(n):
                ax.text(j, i, f"{pearson_m[i,j]:.2f}", ha="center",
                        va="center", fontsize=5,
                        color="white" if abs(pearson_m[i,j]) > 0.5 else "black")
        fig.colorbar(im, ax=ax, shrink=0.7)
        ax.set_title("Матрица корреляций Пирсона", fontsize=10, fontweight="bold")
        fig.tight_layout()
        _embed_fig(self._tab_corr, fig)

    def _show_anova(self, res):
        for w in self._tab_anova.winfo_children():
            w.destroy()
        if not _MPL or not self._cb:
            return
        cb = self._cb
        if cb._feature_matrix is None:
            cb._build_matrix()
        X = cb._feature_matrix
        factor = X[:, 3]
        response = X[:, 9]
        groups = {}
        for i, f in enumerate(factor):
            groups.setdefault(f"{f:.0f}", []).append(response[i])

        fig = Figure(figsize=(8, 5), facecolor=BG)
        ax = fig.add_subplot(111)
        labels = sorted(groups.keys())
        data = [groups[k] for k in labels]
        bp = ax.boxplot(data, labels=labels, patch_artist=True,
                        medianprops=dict(color=ACC, linewidth=2))
        colors = ["#3498db", "#27ae60", "#e67e22", "#9b59b6", "#1abc9c"]
        for patch, c in zip(bp["boxes"], colors * 3):
            patch.set_facecolor(c)
            patch.set_alpha(0.6)
        ax.set_xlabel("Ранг пожара", fontsize=10)
        ax.set_ylabel("Длительность (мин)", fontsize=10)
        sig = "значимо" if res.significant else "не значимо"
        ax.set_title(f"Дисперсионный анализ: {res.method}\n"
                     f"H={res.statistic:.2f}, p={res.p_value:.4f}, "
                     f"η²={res.eta_squared:.3f} ({sig})",
                     fontsize=9, fontweight="bold")
        ax.grid(True, axis="y", alpha=0.3)
        fig.tight_layout()
        _embed_fig(self._tab_anova, fig)

    def _show_regression(self, res):
        for w in self._tab_reg.winfo_children():
            w.destroy()
        frm = tk.Frame(self._tab_reg, bg=BG)
        frm.pack(fill="both", expand=True, padx=10, pady=10)
        text = scrolledtext.ScrolledText(frm, bg=PNL, fg=TXT,
                                         font=("Consolas", 10), wrap="word")
        text.pack(fill="both", expand=True)
        lines = [
            f"Множественная регрессия",
            f"{'='*50}",
            f"Отклик:     {res.response}",
            f"Предикторы: {', '.join(res.predictors)}",
            f"",
            f"R²       = {res.r_squared:.4f}",
            f"R²_adj   = {res.adj_r_squared:.4f}",
            f"F        = {res.f_statistic:.2f}",
            f"p(F)     = {res.f_p_value:.6f}",
            f"σ_остат. = {res.residual_std:.2f}",
            f"",
            f"Коэффициенты:",
        ]
        for name, beta in res.coefficients.items():
            lines.append(f"  {name:<35s} β = {beta:+.4f}")
        text.insert("1.0", "\n".join(lines))
        text.config(state="disabled")

    def _show_distributions(self, cb, dist_res):
        for w in self._tab_dist.winfo_children():
            w.destroy()
        if not _MPL:
            return
        if cb._feature_matrix is None:
            cb._build_matrix()
        X = cb._feature_matrix
        indices = [0, 2, 3, 5, 7, 9]
        from cbr_engine import FEATURE_NAMES

        fig = Figure(figsize=(10, 6), facecolor=BG)
        for idx, fi in enumerate(indices):
            ax = fig.add_subplot(2, 3, idx + 1)
            col = X[:, fi]
            ax.hist(col, bins=min(20, len(col)//3 + 1), color="#3498db",
                    alpha=0.7, edgecolor="white")
            ax.axvline(np.mean(col), color=ACC, linewidth=1.5, linestyle="--")
            ax.set_title(FEATURE_NAMES[fi].split("(")[0][:18], fontsize=8,
                         fontweight="bold")
            ax.tick_params(labelsize=6)
            ax.grid(True, alpha=0.3)
        fig.suptitle("Распределения ключевых признаков", fontsize=10,
                     fontweight="bold")
        fig.tight_layout()
        _embed_fig(self._tab_dist, fig)


# ═══════════════════════════════════════════════════════════════════════════
# 2. АНАЛИЗ ВРЕМЕННЫХ РЯДОВ
# ═══════════════════════════════════════════════════════════════════════════
class TimeseriesWindow(tk.Toplevel):
    """Окно анализа временных рядов."""

    def __init__(self, master=None):
        super().__init__(master)
        self.title("САУР-ПСП — Анализ временных рядов")
        self.geometry("1100x700")
        self.configure(bg=BG)

        self._status = _status_bar(self)
        self._build()
        self._generate_data()

    def _build(self):
        nb = ttk.Notebook(self)
        nb.pack(fill="both", expand=True, padx=6, pady=6)

        self._tab_cp = tk.Frame(nb, bg=BG)
        self._tab_acf = tk.Frame(nb, bg=BG)
        self._tab_spec = tk.Frame(nb, bg=BG)
        self._tab_bayes = tk.Frame(nb, bg=BG)
        self._tab_cfg = tk.Frame(nb, bg=BG)

        nb.add(self._tab_cp, text="  Точки разладки  ")
        nb.add(self._tab_acf, text="  ACF / PACF  ")
        nb.add(self._tab_spec, text="  Спектр  ")
        nb.add(self._tab_bayes, text="  Байесовская оценка  ")
        nb.add(self._tab_cfg, text="  Настройки  ")

        # Настройки
        cfg = self._tab_cfg
        tk.Label(cfg, text="Длина ряда:", bg=BG, fg=TXT,
                 font=("Arial", 9)).pack(anchor="w", padx=10, pady=(10, 2))
        self._len_var = tk.IntVar(value=200)
        tk.Scale(cfg, from_=50, to=1000, orient="horizontal",
                 variable=self._len_var, bg=BG, fg=TXT,
                 troughcolor=GRD, length=300).pack(padx=10)

        tk.Label(cfg, text="Порог CUSUM:", bg=BG, fg=TXT,
                 font=("Arial", 9)).pack(anchor="w", padx=10, pady=(10, 2))
        self._thresh_var = tk.DoubleVar(value=3.0)
        tk.Scale(cfg, from_=0.5, to=10.0, resolution=0.5, orient="horizontal",
                 variable=self._thresh_var, bg=BG, fg=TXT,
                 troughcolor=GRD, length=300).pack(padx=10)

        tk.Button(cfg, text="Пересчитать", command=self._generate_data,
                  bg=ACC, fg="white", font=("Arial", 10, "bold"),
                  relief="flat", padx=16, pady=6).pack(pady=10)

    def _generate_data(self):
        self._status.set("Вычисление...")
        self.update_idletasks()

        def _run():
            try:
                from timeseries import (acf, pacf, cusum_detect,
                    bayesian_changepoint, bayesian_parameter_estimate,
                    periodogram, rolling_stats)

                n = self._len_var.get()
                thresh = self._thresh_var.get()
                rng = np.random.RandomState(42)
                # Синтетический ряд с разладкой в середине
                half = n // 2
                risk = np.concatenate([
                    rng.normal(0.3, 0.05, half),
                    rng.normal(0.7, 0.08, n - half)
                ])
                times = np.arange(n) * 5.0

                cp_c = cusum_detect(risk, threshold=thresh * np.std(risk),
                                    times=times)
                cp_b = bayesian_changepoint(risk, times=times)
                acf_v = acf(risk, max_lag=min(40, n // 3))
                pacf_v = pacf(risk, max_lag=min(20, n // 4))
                freqs, power = periodogram(risk)
                bayes = bayesian_parameter_estimate(
                    risk[:half], prior_mean=0.5, prior_std=0.2,
                    param_name="Индекс риска (фаза S2)")
                rs = rolling_stats(risk, max(3, n // 15))

                self.after(0, lambda: self._show_changepoints(
                    times, risk, cp_c + cp_b, rs))
                self.after(0, lambda: self._show_acf(acf_v, pacf_v, n))
                self.after(0, lambda: self._show_spectrum(freqs, power))
                self.after(0, lambda: self._show_bayesian(bayes))
                self.after(0, lambda: self._status.set(
                    f"Готово: {n} точек, CUSUM: {len(cp_c)}, "
                    f"Байес: {len(cp_b)} разладок"))
            except Exception as e:
                self.after(0, lambda: self._status.set(f"Ошибка: {e}"))

        threading.Thread(target=_run, daemon=True).start()

    def _show_changepoints(self, times, values, cps, rs):
        for w in self._tab_cp.winfo_children():
            w.destroy()
        if not _MPL:
            return
        fig = Figure(figsize=(10, 5), facecolor=BG)
        ax1 = fig.add_subplot(211)
        ax1.plot(times, values, color=TXT, linewidth=0.8, label="Индекс риска")
        w = max(3, len(values) // 15)
        if len(rs["mean"]) > 0:
            ax1.plot(times[w-1:w-1+len(rs["mean"])], rs["mean"],
                     color="#3498db", linewidth=2, label="Скольз. среднее")
        for cp in cps:
            ax1.axvline(cp.time, color=ACC, linestyle="--", alpha=0.7)
        ax1.set_ylabel("Риск")
        ax1.legend(fontsize=7)
        ax1.set_title("Обнаружение точек разладки", fontweight="bold", fontsize=10)
        ax1.grid(True, alpha=0.3)

        ax2 = fig.add_subplot(212)
        ax2.fill_between(range(len(rs["var"])), rs["var"],
                         color=ACC, alpha=0.3)
        ax2.plot(rs["var"], color=ACC, linewidth=1)
        ax2.set_ylabel("Дисперсия")
        ax2.set_xlabel("Время (мин)")
        ax2.grid(True, alpha=0.3)
        fig.tight_layout()
        _embed_fig(self._tab_cp, fig)

    def _show_acf(self, acf_v, pacf_v, n):
        for w in self._tab_acf.winfo_children():
            w.destroy()
        if not _MPL:
            return
        fig = Figure(figsize=(10, 4), facecolor=BG)
        ci = 1.96 / np.sqrt(n)
        ax1 = fig.add_subplot(121)
        ax1.bar(range(len(acf_v)), acf_v, color="#3498db", width=0.5)
        ax1.axhline(ci, color=ACC, linestyle="--", linewidth=0.8)
        ax1.axhline(-ci, color=ACC, linestyle="--", linewidth=0.8)
        ax1.set_title("Автокорреляция (ACF)", fontweight="bold")
        ax1.set_xlabel("Лаг")
        ax1.grid(True, alpha=0.3)

        ax2 = fig.add_subplot(122)
        ax2.bar(range(len(pacf_v)), pacf_v, color="#27ae60", width=0.5)
        ax2.axhline(ci, color=ACC, linestyle="--", linewidth=0.8)
        ax2.axhline(-ci, color=ACC, linestyle="--", linewidth=0.8)
        ax2.set_title("Частная автокорреляция (PACF)", fontweight="bold")
        ax2.set_xlabel("Лаг")
        ax2.grid(True, alpha=0.3)
        fig.tight_layout()
        _embed_fig(self._tab_acf, fig)

    def _show_spectrum(self, freqs, power):
        for w in self._tab_spec.winfo_children():
            w.destroy()
        if not _MPL:
            return
        fig = Figure(figsize=(8, 4), facecolor=BG)
        ax = fig.add_subplot(111)
        ax.semilogy(freqs[1:], power[1:], color="#e67e22", linewidth=1.5)
        ax.set_xlabel("Частота (1/мин)")
        ax.set_ylabel("Мощность")
        ax.set_title("Периодограмма", fontweight="bold")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        _embed_fig(self._tab_spec, fig)

    def _show_bayesian(self, est):
        for w in self._tab_bayes.winfo_children():
            w.destroy()
        if not _MPL:
            return
        from scipy import stats as sp_stats
        fig = Figure(figsize=(8, 5), facecolor=BG)
        ax = fig.add_subplot(111)
        x = np.linspace(est.prior_mean - 4*est.prior_std,
                       est.prior_mean + 4*est.prior_std, 200)
        ax.plot(x, sp_stats.norm.pdf(x, est.prior_mean, est.prior_std),
                "--", color="#3498db", linewidth=1.5,
                label=f"Априорное N({est.prior_mean:.2f}, {est.prior_std:.2f})")
        ax.fill_between(x, sp_stats.norm.pdf(x, est.prior_mean, est.prior_std),
                        alpha=0.1, color="#3498db")
        x2 = np.linspace(est.posterior_mean - 4*est.posterior_std,
                        est.posterior_mean + 4*est.posterior_std, 200)
        ax.plot(x2, sp_stats.norm.pdf(x2, est.posterior_mean, est.posterior_std),
                "-", color=ACC, linewidth=2,
                label=f"Апостериорное N({est.posterior_mean:.2f}, {est.posterior_std:.2f})")
        ax.fill_between(x2, sp_stats.norm.pdf(x2, est.posterior_mean,
                        est.posterior_std), alpha=0.2, color=ACC)
        ax.axvline(est.ci95[0], color="#e67e22", linestyle=":", linewidth=1)
        ax.axvline(est.ci95[1], color="#e67e22", linestyle=":", linewidth=1,
                   label=f"95% ДИ [{est.ci95[0]:.3f}; {est.ci95[1]:.3f}]")
        ax.set_xlabel(est.param_name)
        ax.set_ylabel("Плотность")
        ax.set_title(f"Байесовская оценка (n={est.n_observations})",
                     fontweight="bold")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        _embed_fig(self._tab_bayes, fig)


# ═══════════════════════════════════════════════════════════════════════════
# 3. МАРКОВСКИЕ МОДЕЛИ
# ═══════════════════════════════════════════════════════════════════════════
class MarkovWindow(tk.Toplevel):
    """Окно калибровки и анализа марковских моделей."""

    def __init__(self, master=None):
        super().__init__(master)
        self.title("САУР-ПСП — Марковские модели")
        self.geometry("1100x700")
        self.configure(bg=BG)

        self._status = _status_bar(self)
        self._build()
        self._calibrate()

    def _build(self):
        nb = ttk.Notebook(self)
        nb.pack(fill="both", expand=True, padx=6, pady=6)

        self._tab_weibull = tk.Frame(nb, bg=BG)
        self._tab_trans = tk.Frame(nb, bg=BG)
        self._tab_compare = tk.Frame(nb, bg=BG)
        self._tab_stat = tk.Frame(nb, bg=BG)
        self._tab_cfg = tk.Frame(nb, bg=BG)

        nb.add(self._tab_weibull, text="  Параметры Вейбулла  ")
        nb.add(self._tab_trans, text="  Матрица переходов  ")
        nb.add(self._tab_compare, text="  Сравнение  ")
        nb.add(self._tab_stat, text="  Стационарное распр.  ")
        nb.add(self._tab_cfg, text="  Настройки  ")

        cfg = self._tab_cfg
        tk.Label(cfg, text="Размер выборки:", bg=BG, fg=TXT,
                 font=("Arial", 9)).pack(anchor="w", padx=10, pady=(10, 2))
        self._n_var = tk.IntVar(value=100)
        tk.Scale(cfg, from_=20, to=500, orient="horizontal",
                 variable=self._n_var, bg=BG, fg=TXT,
                 troughcolor=GRD, length=300).pack(padx=10)
        tk.Button(cfg, text="Перекалибровать",
                  command=self._calibrate, bg=ACC, fg="white",
                  font=("Arial", 10, "bold"), relief="flat",
                  padx=16, pady=6).pack(pady=10)

    def _calibrate(self):
        self._status.set("Калибровка...")
        self.update_idletasks()

        def _run():
            try:
                from cbr_engine import generate_demo_casebase
                from calibration import SemiMarkovCalibrator
                cb = generate_demo_casebase(self._n_var.get())
                cal = SemiMarkovCalibrator()
                result = cal.calibrate(cb)

                self.after(0, lambda: self._show_weibull(result))
                self.after(0, lambda: self._show_transitions(result))
                self.after(0, lambda: self._show_comparison(result))
                self.after(0, lambda: self._show_stationary(result))
                self.after(0, lambda: self._status.set(
                    f"Калибровка завершена ({result.n_cases} случаев, "
                    f"качество: {result.quality_metrics.get('ks_pass_rate', 0):.0%})"))
            except Exception as e:
                self.after(0, lambda: self._status.set(f"Ошибка: {e}"))

        threading.Thread(target=_run, daemon=True).start()

    def _show_weibull(self, result):
        for w in self._tab_weibull.winfo_children():
            w.destroy()
        headers = ["Фаза", "k (форма)", "λ (масштаб)", "M (среднее)",
                   "n", "KS p-знач.", "95% ДИ k", "95% ДИ λ"]
        rows = []
        for p in ["S1", "S2", "S3", "S4", "S5"]:
            wb = result.weibull_params[p]
            rows.append((p, f"{wb.k:.2f}", f"{wb.lam:.1f}", f"{wb.mean:.1f}",
                         wb.n_samples, f"{wb.ks_p:.4f}",
                         f"[{wb.ci95_k[0]:.2f}; {wb.ci95_k[1]:.2f}]",
                         f"[{wb.ci95_lam[0]:.1f}; {wb.ci95_lam[1]:.1f}]"))
        _results_table(self._tab_weibull, headers, rows)

    def _show_transitions(self, result):
        for w in self._tab_trans.winfo_children():
            w.destroy()
        if not _MPL:
            return
        labels = ["S1", "S2", "S3", "S4", "S5", "RESOLVED"]
        M = np.zeros((5, 6))
        for i, p in enumerate(["S1", "S2", "S3", "S4", "S5"]):
            for j, q in enumerate(labels):
                M[i, j] = result.transition_matrix.get(p, {}).get(q, 0)

        fig = Figure(figsize=(8, 5), facecolor=BG)
        ax = fig.add_subplot(111)
        im = ax.imshow(M, cmap="YlOrRd", vmin=0, vmax=1)
        ax.set_xticks(range(6))
        ax.set_xticklabels(labels, fontsize=9)
        ax.set_yticks(range(5))
        ax.set_yticklabels(["S1", "S2", "S3", "S4", "S5"], fontsize=9)
        for i in range(5):
            for j in range(6):
                if M[i, j] > 0.01:
                    ax.text(j, i, f"{M[i,j]:.2f}", ha="center", va="center",
                            fontsize=9, fontweight="bold",
                            color="white" if M[i, j] > 0.5 else "black")
        fig.colorbar(im, ax=ax, shrink=0.8, label="P(переход)")
        ax.set_title("Калиброванная матрица переходов", fontweight="bold")
        ax.set_xlabel("Следующая фаза")
        ax.set_ylabel("Текущая фаза")
        fig.tight_layout()
        _embed_fig(self._tab_trans, fig)

    def _show_comparison(self, result):
        for w in self._tab_compare.winfo_children():
            w.destroy()
        if not _MPL:
            return
        defaults_k = [1.5, 2.0, 2.5, 2.0, 1.5]
        defaults_lam = [12.0, 25.0, 75.0, 50.0, 90.0]
        phases = ["S1", "S2", "S3", "S4", "S5"]
        calib_k = [result.weibull_params[p].k for p in phases]
        calib_lam = [result.weibull_params[p].lam for p in phases]

        fig = Figure(figsize=(10, 4), facecolor=BG)
        x = np.arange(5)
        w = 0.35

        ax1 = fig.add_subplot(121)
        ax1.bar(x - w/2, defaults_k, w, label="Исходные", color="#3498db")
        ax1.bar(x + w/2, calib_k, w, label="Калиброванные", color=ACC)
        ax1.set_xticks(x)
        ax1.set_xticklabels(phases)
        ax1.set_ylabel("k (форма)")
        ax1.set_title("Параметр формы k", fontweight="bold")
        ax1.legend(fontsize=8)
        ax1.grid(True, axis="y", alpha=0.3)

        ax2 = fig.add_subplot(122)
        ax2.bar(x - w/2, defaults_lam, w, label="Исходные", color="#3498db")
        ax2.bar(x + w/2, calib_lam, w, label="Калиброванные", color=ACC)
        ax2.set_xticks(x)
        ax2.set_xticklabels(phases)
        ax2.set_ylabel("λ (масштаб, мин)")
        ax2.set_title("Параметр масштаба λ", fontweight="bold")
        ax2.legend(fontsize=8)
        ax2.grid(True, axis="y", alpha=0.3)
        fig.tight_layout()
        _embed_fig(self._tab_compare, fig)

    def _show_stationary(self, result):
        for w in self._tab_stat.winfo_children():
            w.destroy()
        text = scrolledtext.ScrolledText(self._tab_stat, bg=PNL, fg=TXT,
                                         font=("Consolas", 10), wrap="word")
        text.pack(fill="both", expand=True, padx=6, pady=6)
        lines = ["Стационарное распределение полумарковской цепи", "=" * 50, ""]
        phases = ["S1", "S2", "S3", "S4", "S5"]
        total_mean = sum(result.weibull_params[p].mean for p in phases)
        for p in phases:
            wb = result.weibull_params[p]
            frac = wb.mean / total_mean if total_mean > 0 else 0
            bar = "█" * int(frac * 40)
            lines.append(f"  {p}:  M={wb.mean:>7.1f} мин  "
                         f"({frac:>5.1%})  {bar}")
        lines.append(f"\n  Суммарное среднее время: {total_mean:.1f} мин "
                     f"({total_mean/60:.1f} ч)")
        text.insert("1.0", "\n".join(lines))
        text.config(state="disabled")


# ═══════════════════════════════════════════════════════════════════════════
# 4. ЭКСПЕРТНАЯ СИСТЕМА И ПРЕЦЕДЕНТЫ
# ═══════════════════════════════════════════════════════════════════════════
class ExpertCBRWindow(tk.Toplevel):
    """Окно экспертной системы и прецедентного анализа."""

    def __init__(self, master=None):
        super().__init__(master)
        self.title("САУР-ПСП — Экспертная система и прецеденты")
        self.geometry("1100x700")
        self.configure(bg=BG)

        self._status = _status_bar(self)
        self._build()
        self._load_data()

    def _build(self):
        nb = ttk.Notebook(self)
        nb.pack(fill="both", expand=True, padx=6, pady=6)

        self._tab_rules = tk.Frame(nb, bg=BG)
        self._tab_search = tk.Frame(nb, bg=BG)
        self._tab_clusters = tk.Frame(nb, bg=BG)
        self._tab_classify = tk.Frame(nb, bg=BG)
        self._tab_cfg = tk.Frame(nb, bg=BG)

        nb.add(self._tab_rules, text="  Правила ЭС  ")
        nb.add(self._tab_search, text="  Поиск прецедентов  ")
        nb.add(self._tab_clusters, text="  Кластеры  ")
        nb.add(self._tab_classify, text="  Классификатор  ")
        nb.add(self._tab_cfg, text="  Настройки  ")

        # Настройки поиска
        cfg = self._tab_cfg
        tk.Label(cfg, text="Объём РВС (м³):", bg=BG, fg=TXT,
                 font=("Arial", 9)).pack(anchor="w", padx=10, pady=(10, 2))
        self._vol_var = tk.IntVar(value=20000)
        tk.Scale(cfg, from_=500, to=50000, orient="horizontal",
                 variable=self._vol_var, resolution=500, bg=BG, fg=TXT,
                 troughcolor=GRD, length=300).pack(padx=10)

        tk.Label(cfg, text="Ранг пожара:", bg=BG, fg=TXT,
                 font=("Arial", 9)).pack(anchor="w", padx=10, pady=(10, 2))
        self._rank_var = tk.IntVar(value=4)
        tk.Scale(cfg, from_=1, to=5, orient="horizontal",
                 variable=self._rank_var, bg=BG, fg=TXT,
                 troughcolor=GRD, length=300).pack(padx=10)

        tk.Label(cfg, text="Число кластеров:", bg=BG, fg=TXT,
                 font=("Arial", 9)).pack(anchor="w", padx=10, pady=(10, 2))
        self._k_var = tk.IntVar(value=4)
        tk.Scale(cfg, from_=2, to=8, orient="horizontal",
                 variable=self._k_var, bg=BG, fg=TXT,
                 troughcolor=GRD, length=300).pack(padx=10)

        tk.Button(cfg, text="Пересчитать", command=self._load_data,
                  bg=ACC, fg="white", font=("Arial", 10, "bold"),
                  relief="flat", padx=16, pady=6).pack(pady=10)

    def _load_data(self):
        self._status.set("Загрузка...")
        self.update_idletasks()

        def _run():
            try:
                from expert_system import ExpertSystem, ACTION_NAMES_RU
                from cbr_engine import (generate_demo_casebase,
                    ScenarioClusterer, SituationClassifier, PrecedentSearch)

                es = ExpertSystem()
                cb = generate_demo_casebase(100)
                cl = ScenarioClusterer(n_clusters=self._k_var.get())
                cl_result = cl.fit(cb)
                sc = SituationClassifier()
                sc.fit_from_casebase(cb)
                ps = PrecedentSearch(cb)
                vol = self._vol_var.get()
                rank = self._rank_var.get()
                similar = ps.find_for_situation(vol, vol * 0.06, rank,
                                               "бензин", 0.70, k=10)

                self.after(0, lambda: self._show_rules(es))
                self.after(0, lambda: self._show_search(similar, vol, rank))
                self.after(0, lambda: self._show_clusters(cb, cl_result))
                self.after(0, lambda: self._show_classifier(cb, sc))
                self.after(0, lambda: self._status.set(
                    f"Готово: {es.n_rules} правил, {len(cb)} прецедентов, "
                    f"{cl_result['n_clusters']} кластеров"))
            except Exception as e:
                self.after(0, lambda: self._status.set(f"Ошибка: {e}"))

        threading.Thread(target=_run, daemon=True).start()

    def _show_rules(self, es):
        for w in self._tab_rules.winfo_children():
            w.destroy()
        headers = ["Приоритет", "Фаза", "Название", "Действие", "Обоснование"]
        rows = [(r.priority, r.phase, r.name,
                 f"{r.action}", r.rationale[:50])
                for r in sorted(es.rules, key=lambda x: (-x.priority,))]
        _results_table(self._tab_rules, headers, rows)

    def _show_search(self, similar, vol, rank):
        for w in self._tab_search.winfo_children():
            w.destroy()
        tk.Label(self._tab_search,
                 text=f"Поиск: РВС {vol} м³, ранг №{rank}",
                 font=("Arial", 10, "bold"), bg=BG, fg=TXT).pack(pady=(6, 2))
        headers = ["Прецедент", "Объём", "Ранг", "Топливо", "Кровля",
                   "Длительность", "Ликвидир.", "Расстояние"]
        rows = [(c.case_id, f"{c.rvs_volume:.0f}", c.fire_rank,
                 c.fuel_type, c.roof_type, f"{c.duration_min} мин",
                 "Да" if c.extinguished else "Нет", f"{d:.3f}")
                for c, d in similar]
        _results_table(self._tab_search, headers, rows)

    def _show_clusters(self, cb, cl_result):
        for w in self._tab_clusters.winfo_children():
            w.destroy()
        if not _MPL:
            return
        fig = Figure(figsize=(8, 5), facecolor=BG)
        ax = fig.add_subplot(111)
        colors = ["#3498db", "#e74c3c", "#27ae60", "#e67e22",
                  "#9b59b6", "#1abc9c", "#f39c12", "#566573"]
        for case in cb.cases:
            c = colors[case.cluster_id % len(colors)]
            ax.scatter(case.rvs_volume, case.duration_min,
                       c=c, s=30, alpha=0.6, edgecolors="white", linewidth=0.5)
        # Легенда
        for k, name in cl_result.get("cluster_sizes", {}).items():
            ci = list(cl_result["cluster_sizes"].keys()).index(k)
            ax.scatter([], [], c=colors[ci % len(colors)], s=60, label=k[:25])
        ax.set_xlabel("Объём РВС (м³)", fontsize=10)
        ax.set_ylabel("Длительность (мин)", fontsize=10)
        ax.set_title("Кластеризация сценариев пожаров", fontweight="bold")
        ax.legend(fontsize=7, loc="upper left")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        _embed_fig(self._tab_clusters, fig)

    def _show_classifier(self, cb, sc):
        for w in self._tab_classify.winfo_children():
            w.destroy()
        # Распределение типов ситуаций
        type_counts = {}
        for case in cb.cases:
            t = case.situation_type or "Неизвестно"
            type_counts[t] = type_counts.get(t, 0) + 1

        if not _MPL:
            return
        fig = Figure(figsize=(8, 5), facecolor=BG)
        ax = fig.add_subplot(111)
        labels = list(type_counts.keys())
        values = list(type_counts.values())
        colors = ["#3498db", "#e74c3c", "#27ae60", "#e67e22",
                  "#9b59b6", "#1abc9c", "#f39c12", "#566573"]
        bars = ax.barh(range(len(labels)), values,
                       color=[colors[i % len(colors)] for i in range(len(labels))],
                       edgecolor="white")
        ax.set_yticks(range(len(labels)))
        ax.set_yticklabels([l[:30] for l in labels], fontsize=8)
        ax.set_xlabel("Число случаев")
        ax.set_title("Классификация ситуаций (k-NN)", fontweight="bold")
        for bar, v in zip(bars, values):
            ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2,
                    str(v), va="center", fontsize=8, fontweight="bold")
        ax.grid(True, axis="x", alpha=0.3)
        fig.tight_layout()
        _embed_fig(self._tab_classify, fig)
