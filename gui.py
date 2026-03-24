"""SAUR GUI — multi-tab tkinter interface with real-time matplotlib plots.

Dark theme: bg="#1a1a2e", panel="#16213e", accent="#e94560"
Tabs:
  1. Панель управления  — status indicators + timeline plot
  2. Полумарковская цепь — phase occupancy bar + sojourn PDFs
  3. Обучение RL        — reward curve + Q-value heatmap
  4. Уровни управления  — 5-level status table
  5. Журнал адаптации   — scrollable adaptation log
"""
from __future__ import annotations

import threading
import numpy as np
import tkinter as tk
from tkinter import ttk, scrolledtext

try:
    import matplotlib
    matplotlib.use("TkAgg")
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
    from matplotlib.figure import Figure
    _MPL_OK = True
except Exception:
    _MPL_OK = False

from .simulation import SAURSimulation, SimParams
from .rl_agent import QLearningAgent, ACTION_NAMES, STATE_SIZE, N_ACTIONS
from .semi_markov import SemiMarkovChain, PHASES, SOJOURN_PARAMS
from .state_space import FirePhase

# ---------------------------------------------------------------------------
# Color constants
# ---------------------------------------------------------------------------
BG     = "#1a1a2e"
PANEL  = "#16213e"
ACCENT = "#e94560"
TEXT   = "#e0e0e0"
GREEN  = "#00c853"
YELLOW = "#ffd600"
BLUE   = "#40c4ff"
GRAY   = "#546e7a"

RISK_COLORS = {"LOW": GREEN, "MEDIUM": YELLOW, "HIGH": "#ff6d00", "CRITICAL": "#d50000"}
PHASE_COLORS = {
    "NORMAL": GRAY, "S1": BLUE, "S2": YELLOW,
    "S3": "#ff6d00", "S4": ACCENT, "S5": "#ab47bc", "RESOLVED": GREEN,
}


# ---------------------------------------------------------------------------
# Helper: dark matplotlib figure
# ---------------------------------------------------------------------------
def dark_figure(figsize=(8, 4), nrows=1, ncols=1):
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize,
                             facecolor=PANEL)
    if nrows * ncols == 1:
        axes = [axes]
    else:
        axes = list(np.array(axes).flatten())
    for ax in axes:
        ax.set_facecolor(BG)
        ax.tick_params(colors=TEXT)
        ax.xaxis.label.set_color(TEXT)
        ax.yaxis.label.set_color(TEXT)
        ax.title.set_color(TEXT)
        for spine in ax.spines.values():
            spine.set_edgecolor(GRAY)
    return fig, axes


def embed_figure(fig, parent) -> FigureCanvasTkAgg:
    canvas = FigureCanvasTkAgg(fig, master=parent)
    canvas.draw()
    canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
    return canvas


# ---------------------------------------------------------------------------
# Main application window
# ---------------------------------------------------------------------------
class SAURApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("САУР ПСП — Симуляция")
        self.configure(bg=BG)
        self.geometry("1280x800")
        self.resizable(True, True)

        # Simulation state
        self._sim: SAURSimulation | None = None
        self._results: dict = {}
        self._running = False
        self._rl_agent: QLearningAgent | None = None

        # Build layout
        self._build_left_panel()
        self._build_notebook()

        # Status bar
        self._statusvar = tk.StringVar(value="Готово к запуску")
        tk.Label(self, textvariable=self._statusvar,
                 bg=PANEL, fg=TEXT, anchor="w",
                 font=("Consolas", 9)).pack(fill=tk.X, side=tk.BOTTOM)

    # ------------------------------------------------------------------
    # Left control panel
    # ------------------------------------------------------------------
    def _build_left_panel(self):
        frame = tk.Frame(self, bg=PANEL, width=220)
        frame.pack(side=tk.LEFT, fill=tk.Y, padx=4, pady=4)
        frame.pack_propagate(False)

        tk.Label(frame, text="ПАРАМЕТРЫ", bg=PANEL, fg=ACCENT,
                 font=("Helvetica", 11, "bold")).pack(pady=(10, 4))

        # Sim time
        self._simtime_var = tk.DoubleVar(value=600.0)
        self._make_slider(frame, "Время моделирования (мин)",
                          self._simtime_var, 120, 1440)

        # Fire start
        self._firestart_var = tk.DoubleVar(value=60.0)
        self._make_slider(frame, "Начало пожара (мин)",
                          self._firestart_var, 5, 300)

        # Episodes
        self._episodes_var = tk.IntVar(value=20)
        self._make_slider(frame, "Эпизоды обучения RL",
                          self._episodes_var, 1, 200, integer=True)

        # Seed
        tk.Label(frame, text="Seed", bg=PANEL, fg=TEXT,
                 font=("Consolas", 9)).pack(anchor="w", padx=8)
        self._seed_var = tk.IntVar(value=42)
        tk.Spinbox(frame, from_=0, to=9999, textvariable=self._seed_var,
                   bg=BG, fg=TEXT, insertbackground=TEXT, width=8).pack(
            anchor="w", padx=8, pady=2)

        tk.Frame(frame, bg=GRAY, height=1).pack(fill=tk.X, padx=8, pady=8)

        # Run button
        self._run_btn = tk.Button(
            frame, text="▶ Запустить", bg=ACCENT, fg="white",
            font=("Helvetica", 11, "bold"),
            relief=tk.FLAT, cursor="hand2",
            command=self._on_run)
        self._run_btn.pack(fill=tk.X, padx=8, pady=4)

        # Train button
        self._train_btn = tk.Button(
            frame, text="⚙ Обучить RL", bg="#0d47a1", fg="white",
            font=("Helvetica", 10), relief=tk.FLAT, cursor="hand2",
            command=self._on_train)
        self._train_btn.pack(fill=tk.X, padx=8, pady=4)

        tk.Frame(frame, bg=GRAY, height=1).pack(fill=tk.X, padx=8, pady=8)

        # Big status indicators
        self._phase_var   = tk.StringVar(value="NORMAL")
        self._risk_var    = tk.StringVar(value="LOW")
        self._L7_var      = tk.StringVar(value="L7: —")
        self._eps_var     = tk.StringVar(value="ε: —")

        for var, label in [(self._phase_var, "Фаза"),
                           (self._risk_var,  "Риск"),
                           (self._L7_var,    ""),
                           (self._eps_var,   "")]:
            lbl = tk.Label(frame, textvariable=var, bg=PANEL, fg=ACCENT,
                           font=("Helvetica", 13, "bold"), wraplength=180)
            if label:
                tk.Label(frame, text=label, bg=PANEL, fg=GRAY,
                         font=("Consolas", 8)).pack(anchor="w", padx=10)
            lbl.pack(anchor="w", padx=10, pady=1)

    def _make_slider(self, parent, label, variable, min_v, max_v,
                     integer=False):
        tk.Label(parent, text=label, bg=PANEL, fg=TEXT,
                 font=("Consolas", 8), wraplength=180).pack(
            anchor="w", padx=8, pady=(6, 0))
        frame = tk.Frame(parent, bg=PANEL)
        frame.pack(fill=tk.X, padx=8)
        sl = tk.Scale(frame, from_=min_v, to=max_v,
                      orient=tk.HORIZONTAL,
                      variable=variable, bg=PANEL, fg=TEXT,
                      troughcolor=BG, highlightthickness=0,
                      resolution=1 if integer else 10)
        sl.pack(fill=tk.X)

    # ------------------------------------------------------------------
    # Notebook tabs
    # ------------------------------------------------------------------
    def _build_notebook(self):
        style = ttk.Style()
        style.theme_use("clam")
        style.configure("TNotebook", background=BG, borderwidth=0)
        style.configure("TNotebook.Tab", background=PANEL, foreground=TEXT,
                        padding=[10, 4])
        style.map("TNotebook.Tab",
                  background=[("selected", ACCENT)],
                  foreground=[("selected", "white")])

        nb = ttk.Notebook(self)
        nb.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=4, pady=4)
        self._nb = nb

        self._tab1 = tk.Frame(nb, bg=BG)
        self._tab2 = tk.Frame(nb, bg=BG)
        self._tab3 = tk.Frame(nb, bg=BG)
        self._tab4 = tk.Frame(nb, bg=BG)
        self._tab5 = tk.Frame(nb, bg=BG)

        nb.add(self._tab1, text="Панель управления")
        nb.add(self._tab2, text="Полумарковская цепь")
        nb.add(self._tab3, text="Обучение RL")
        nb.add(self._tab4, text="Уровни управления")
        nb.add(self._tab5, text="Журнал адаптации")

        self._build_tab1()
        self._build_tab2()
        self._build_tab3()
        self._build_tab4()
        self._build_tab5()

    # ------------------------------------------------------------------
    # Tab 1 — Dashboard
    # ------------------------------------------------------------------
    def _build_tab1(self):
        t = self._tab1
        if not _MPL_OK:
            tk.Label(t, text="matplotlib unavailable", bg=BG, fg=ACCENT).pack()
            return

        fig, axes = dark_figure(figsize=(9, 5), nrows=1, ncols=2)
        self._ax_L7   = axes[0]
        self._ax_risk = axes[1]
        self._ax_L7.set_title("L7 Reliability over time")
        self._ax_risk.set_title("Risk Score over time")
        fig.tight_layout(pad=2.0)
        self._canvas_tab1 = embed_figure(fig, t)
        self._fig_tab1 = fig

    def _update_tab1(self):
        if not _MPL_OK or not self.metrics_history:
            return
        mh = self.metrics_history
        t_vals = list(range(len(mh)))
        L7s   = [m.L7_reliability for m in mh]
        risks = [m.risk_score for m in mh]

        ax = self._ax_L7
        ax.cla()
        ax.set_facecolor(BG)
        ax.plot(t_vals, L7s, color=BLUE, linewidth=1.5, label="L7")
        ax.axhline(0.9, color=GREEN, linestyle="--", linewidth=0.8,
                   label="target 0.90")
        ax.set_xlabel("Tick")
        ax.set_ylabel("L7")
        ax.set_ylim(0, 1.05)
        ax.legend(facecolor=PANEL, edgecolor=GRAY, labelcolor=TEXT,
                  fontsize=8)
        ax.set_title("L7 Reliability", color=TEXT)
        ax.tick_params(colors=TEXT)

        ax2 = self._ax_risk
        ax2.cla()
        ax2.set_facecolor(BG)
        phase_colors_mapped = [PHASE_COLORS.get(m.phase.name, GRAY) for m in mh]
        ax2.scatter(t_vals, risks, c=phase_colors_mapped, s=4)
        ax2.plot(t_vals, risks, color=ACCENT, linewidth=1.0, alpha=0.7)
        ax2.axhline(0.5, color=YELLOW, linestyle="--", linewidth=0.8,
                    label="HIGH threshold")
        ax2.set_xlabel("Tick")
        ax2.set_ylabel("Risk score")
        ax2.set_ylim(0, 1.05)
        ax2.legend(facecolor=PANEL, edgecolor=GRAY, labelcolor=TEXT,
                   fontsize=8)
        ax2.set_title("Risk Score", color=TEXT)
        ax2.tick_params(colors=TEXT)

        self._fig_tab1.tight_layout(pad=2.0)
        self._canvas_tab1.draw()

        # Update left panel indicators
        last = mh[-1]
        self._phase_var.set(last.phase.name)
        self._risk_var.set(last.risk_level)
        self._L7_var.set(f"L7: {last.L7_reliability:.2f}")
        if self._sim:
            self._eps_var.set(f"ε: {self._sim._l3.rl_agent.epsilon:.3f}")

    # ------------------------------------------------------------------
    # Tab 2 — Semi-Markov chain
    # ------------------------------------------------------------------
    def _build_tab2(self):
        t = self._tab2
        if not _MPL_OK:
            tk.Label(t, text="matplotlib unavailable", bg=BG, fg=ACCENT).pack()
            return

        fig, axes = dark_figure(figsize=(9, 5), nrows=2, ncols=4)
        self._ax_occ = axes[0]
        self._ax_pdfs = axes[1:]  # 7 subplots for PDFs

        # Occupancy bar chart
        self._ax_occ.set_title("Phase Occupancy (%)")

        # PDF subplots — we have 7 phases, use axes[1..7]
        phase_names = [ph.name for ph in PHASES]
        for i, ax in enumerate(self._ax_pdfs):
            if i < len(PHASES):
                ax.set_title(phase_names[i], fontsize=7)
            ax.set_facecolor(BG)
            ax.tick_params(colors=TEXT, labelsize=6)

        fig.tight_layout(pad=1.5)
        self._canvas_tab2 = embed_figure(fig, t)
        self._fig_tab2 = fig
        # Draw static chain stats immediately
        self._update_tab2()

    def _update_tab2(self):
        if not _MPL_OK:
            return
        chain = SemiMarkovChain(seed=0)
        occ = chain.phase_occupancy()
        names = list(occ.keys())
        vals  = [occ[k] * 100 for k in names]

        ax = self._ax_occ
        ax.cla()
        ax.set_facecolor(BG)
        colors = [PHASE_COLORS.get(n, GRAY) for n in names]
        ax.bar(names, vals, color=colors, edgecolor=GRAY)
        ax.set_ylabel("%", color=TEXT)
        ax.set_title("Phase Occupancy (steady-state)", color=TEXT)
        ax.tick_params(axis="x", rotation=45, colors=TEXT, labelsize=7)
        ax.tick_params(axis="y", colors=TEXT)

        # PDFs
        t_arr = np.linspace(0.1, 300, 200)
        for i, ax_pdf in enumerate(self._ax_pdfs):
            ax_pdf.cla()
            ax_pdf.set_facecolor(BG)
            if i < len(PHASES):
                ph = PHASES[i]
                dist = SOJOURN_PARAMS[ph]
                pdf = dist.pdf(t_arr)
                ax_pdf.plot(t_arr, pdf, color=PHASE_COLORS.get(ph.name, BLUE),
                            linewidth=1.2)
                ax_pdf.set_title(ph.name, color=TEXT, fontsize=7)
                ax_pdf.tick_params(colors=TEXT, labelsize=6)
                ax_pdf.set_facecolor(BG)
                for sp in ax_pdf.spines.values():
                    sp.set_edgecolor(GRAY)

        self._fig_tab2.tight_layout(pad=1.5)
        self._canvas_tab2.draw()

    # ------------------------------------------------------------------
    # Tab 3 — RL Training
    # ------------------------------------------------------------------
    def _build_tab3(self):
        t = self._tab3
        if not _MPL_OK:
            tk.Label(t, text="matplotlib unavailable", bg=BG, fg=ACCENT).pack()
            return

        fig, axes = dark_figure(figsize=(9, 5), nrows=1, ncols=2)
        self._ax_rewards = axes[0]
        self._ax_qheat   = axes[1]
        self._ax_rewards.set_title("Cumulative Reward per Episode")
        self._ax_qheat.set_title("Q-value heatmap (state x action)")
        fig.tight_layout(pad=2.0)
        self._canvas_tab3 = embed_figure(fig, t)
        self._fig_tab3 = fig

    def _update_tab3(self, agent: QLearningAgent | None = None):
        if not _MPL_OK:
            return
        ag = agent or (self._sim._l3.rl_agent if self._sim else None)
        if ag is None:
            return

        ax = self._ax_rewards
        ax.cla()
        ax.set_facecolor(BG)
        rewards = ag.episode_rewards
        if rewards:
            ax.plot(rewards, color=ACCENT, linewidth=1.5)
            # Rolling mean
            window = min(10, len(rewards))
            if window > 1:
                rm = np.convolve(rewards, np.ones(window) / window, mode="valid")
                ax.plot(range(window - 1, len(rewards)), rm,
                        color=GREEN, linewidth=2.0, label=f"RM{window}")
                ax.legend(facecolor=PANEL, edgecolor=GRAY,
                          labelcolor=TEXT, fontsize=8)
        ax.set_xlabel("Episode", color=TEXT)
        ax.set_ylabel("Reward", color=TEXT)
        ax.set_title("RL Cumulative Reward", color=TEXT)
        ax.tick_params(colors=TEXT)

        # Q-value heatmap — show first 60 states
        ax2 = self._ax_qheat
        ax2.cla()
        ax2.set_facecolor(BG)
        n_show = min(60, STATE_SIZE)
        q_sub = ag.Q[:n_show]
        im = ax2.imshow(q_sub, aspect="auto", cmap="coolwarm",
                        origin="upper")
        ax2.set_xlabel("Action", color=TEXT)
        ax2.set_ylabel("State (first 60)", color=TEXT)
        ax2.set_xticks(range(N_ACTIONS))
        ax2.set_xticklabels([a[:4] for a in ACTION_NAMES],
                             fontsize=7, color=TEXT)
        ax2.set_title("Q-values (state × action)", color=TEXT)
        ax2.tick_params(colors=TEXT)
        try:
            self._fig_tab3.colorbar(im, ax=ax2)
        except Exception:
            pass

        self._fig_tab3.tight_layout(pad=2.0)
        self._canvas_tab3.draw()

    # ------------------------------------------------------------------
    # Tab 4 — Level status table
    # ------------------------------------------------------------------
    def _build_tab4(self):
        t = self._tab4
        columns = ("level", "name", "autonomy", "last_action", "metric")
        self._tree = ttk.Treeview(t, columns=columns, show="headings",
                                  height=12)
        style = ttk.Style()
        style.configure("Treeview", background=PANEL, foreground=TEXT,
                        fieldbackground=PANEL, rowheight=28)
        style.configure("Treeview.Heading", background=BG,
                        foreground=ACCENT, font=("Helvetica", 9, "bold"))
        style.map("Treeview", background=[("selected", ACCENT)])

        headers = {"level": "Уровень", "name": "Название",
                   "autonomy": "Автономность (α)",
                   "last_action": "Последнее действие",
                   "metric": "Метрика"}
        widths = {"level": 60, "name": 140, "autonomy": 110,
                  "last_action": 260, "metric": 160}
        for col in columns:
            self._tree.heading(col, text=headers[col])
            self._tree.column(col, width=widths[col], anchor="w")

        self._tree.pack(fill=tk.BOTH, expand=True, padx=8, pady=8)
        self._populate_level_table(None)

        vsb = ttk.Scrollbar(t, orient="vertical",
                            command=self._tree.yview)
        self._tree.configure(yscrollcommand=vsb.set)

    def _populate_level_table(self, sim: SAURSimulation | None):
        for row in self._tree.get_children():
            self._tree.delete(row)

        if sim is None:
            rows = [
                ("L1", "Сенсорно-ситуационный", "α₁ = 0.0",
                 "Ожидание", "P_COP: —"),
                ("L2", "Тактический",           "α₂ ∈ [0.3, 0.6]",
                 "Ожидание", "Потери: —"),
                ("L3", "Оперативный (RL)",       "α₃ ∈ [0.5, 0.8]",
                 "Ожидание", "ε = 1.000"),
                ("L4", "Системный (ЦУКС)",       "α₄ ∈ [0.7, 0.9]",
                 "Ожидание", "Готовность: —"),
                ("L5", "Стратегический",         "α₅ = 1.0",
                 "Ожидание", "Тренд: —"),
            ]
        else:
            mh = sim.metrics_history
            last = mh[-1] if mh else None

            p_cop  = f"P_COP: {last.cop_accuracy:.2f}"  if last else "—"
            ep     = f"ε = {sim._l3.rl_agent.epsilon:.3f}"
            ready  = (f"Готовность: {last.garrison_readiness:.0%}"
                      if last else "—")
            trend  = (f"Тренд: {sim._l5.mortality_trend():.2f}"
                      if sim._l5.operation_records else "Тренд: нет данных")

            last_adapt = ""
            if sim._adaptation_log:
                _, ar = sim._adaptation_log[-1]
                last_adapt = "; ".join(ar.actions[:1])

            l2_tactic = ""
            if sim._l2_agents:
                l2_tactic = sim._l2_agents[0].current_tactic

            rows = [
                ("L1", "Сенсорно-ситуационный", "α₁ = 0.0",
                 "Слияние сенсоров", p_cop),
                ("L2", "Тактический",           "α₂ = 0.45",
                 l2_tactic or "—", "Безопасность: OK"),
                ("L3", "Оперативный (RL)",       "α₃ = 0.65",
                 last_adapt or "—", ep),
                ("L4", "Системный (ЦУКС)",       "α₄ = 0.80",
                 "Мониторинг гарнизона", ready),
                ("L5", "Стратегический",         "α₅ = 1.0",
                 "Анализ операций", trend),
            ]

        for r in rows:
            self._tree.insert("", "end", values=r)

    # ------------------------------------------------------------------
    # Tab 5 — Adaptation log
    # ------------------------------------------------------------------
    def _build_tab5(self):
        t = self._tab5
        self._log_text = scrolledtext.ScrolledText(
            t, bg=BG, fg=TEXT, font=("Consolas", 9),
            insertbackground=TEXT, wrap=tk.WORD)
        self._log_text.pack(fill=tk.BOTH, expand=True, padx=4, pady=4)
        self._log_text.insert(tk.END, "Журнал адаптации появится после запуска...\n")

    def _update_tab5(self):
        if self._sim is None:
            return
        self._log_text.delete("1.0", tk.END)
        self._log_text.insert(tk.END,
                               f"{'='*70}\n"
                               f"  ЖУРНАЛ АДАПТАЦИИ  (всего записей: "
                               f"{len(self._sim._adaptation_log)})\n"
                               f"{'='*70}\n\n")
        for t_val, ar in self._sim._adaptation_log[-200:]:
            mode_str = ar.mode.value.upper()
            self._log_text.insert(
                tk.END,
                f"[t={t_val:6.1f} мин]  [{mode_str:15s}]  "
                f"prio={ar.priority_level}\n")
            for a in ar.actions:
                self._log_text.insert(tk.END, f"   → {a}\n")
        self._log_text.see(tk.END)

    # ------------------------------------------------------------------
    # Simulation runner
    # ------------------------------------------------------------------
    @property
    def metrics_history(self):
        return self._sim.metrics_history if self._sim else []

    def _on_run(self):
        if self._running:
            return
        self._run_btn.config(state=tk.DISABLED)
        self._train_btn.config(state=tk.DISABLED)
        self._statusvar.set("Выполняется симуляция...")
        self._running = True
        thread = threading.Thread(target=self._run_sim_thread, daemon=True)
        thread.start()

    def _run_sim_thread(self):
        try:
            p = SimParams(
                sim_time=float(self._simtime_var.get()),
                fire_start_time=float(self._firestart_var.get()),
                seed=int(self._seed_var.get()),
                training=True,
            )
            self._sim = SAURSimulation(params=p,
                                       rl_agent=self._rl_agent)
            self._results = self._sim.run()
        except Exception as e:
            self.after(0, self._statusvar.set, f"Ошибка: {e}")
        finally:
            self.after(0, self._on_sim_done)

    def _on_sim_done(self):
        self._running = False
        self._run_btn.config(state=tk.NORMAL)
        self._train_btn.config(state=tk.NORMAL)
        self._statusvar.set(
            f"Готово. mean_L7={self._results.get('mean_L7', 0):.3f}  "
            f"mean_risk={self._results.get('mean_risk', 0):.3f}  "
            f"adaptations={self._results.get('n_adaptations', 0)}")
        self._update_tab1()
        self._update_tab2()
        self._update_tab3()
        self._populate_level_table(self._sim)
        self._update_tab5()

    def _on_train(self):
        if self._running:
            return
        self._run_btn.config(state=tk.DISABLED)
        self._train_btn.config(state=tk.DISABLED)
        self._running = True
        n_ep = int(self._episodes_var.get())
        seed = int(self._seed_var.get())
        sim_t = float(self._simtime_var.get())
        self._statusvar.set(f"Обучение RL ({n_ep} эпизодов)...")
        thread = threading.Thread(
            target=self._train_thread,
            args=(n_ep, seed, sim_t),
            daemon=True)
        thread.start()

    def _train_thread(self, n_ep, seed, sim_t):
        try:
            self._rl_agent = SAURSimulation.run_training(
                n_episodes=n_ep, seed=seed, sim_time=sim_t)
        except Exception as e:
            self.after(0, self._statusvar.set, f"Ошибка обучения: {e}")
        finally:
            self.after(0, self._on_train_done)

    def _on_train_done(self):
        self._running = False
        self._run_btn.config(state=tk.NORMAL)
        self._train_btn.config(state=tk.NORMAL)
        eps = self._rl_agent.epsilon if self._rl_agent else "—"
        n_ep = len(self._rl_agent.episode_rewards) if self._rl_agent else 0
        self._statusvar.set(
            f"Обучение завершено. Эпизодов: {n_ep}  ε={eps:.3f}")
        self._update_tab3(self._rl_agent)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def launch_gui():
    app = SAURApp()
    app.mainloop()


if __name__ == "__main__":
    launch_gui()
