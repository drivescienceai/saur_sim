"""
run_experiment_es_vs_rl.py — Полный эксперимент: ЭС vs ОП.
═══════════════════════════════════════════════════════════════════════════════
Запускает моделирование, собирает метрики, проводит статистическую оценку,
генерирует визуализации и научную статью с реальными данными.

Этапы:
  1. Обучение табличного ОП-агента (500 эпизодов)
  2. Сравнительный прогон: ЭС vs обученный ОП (50 эпизодов × 2 сценария)
  3. Статистическая оценка (Манн-Уитни, Коэна, бутстреп 95% ДИ)
  4. Визуализации (тепловые карты, потоки, радар)
  5. Генерация статьи DOCX с реальными данными

Запуск: python run_experiment_es_vs_rl.py
═══════════════════════════════════════════════════════════════════════════════
"""
from __future__ import annotations

import sys
import io
import os
import json
import time
import datetime
import numpy as np
from scipy import stats as sp_stats

# Настройка вывода UTF-8
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

from tank_fire_sim import TankFireSim, ACTIONS, PHASE_VALID, N_ACT
from expert_system import ExpertSystem
from rl_agent import QLearningAgent, STATE_SIZE, N_ACTIONS
from visualizations import (policy_heatmap, policy_from_expert_rules,
                             policy_from_q_table, decision_flow, radar_chart)

_DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
_FIG_DIR  = os.path.join(_DATA_DIR, "figures")
_EXP_DIR  = os.path.join(_DATA_DIR, "experiment_es_vs_rl")
os.makedirs(_FIG_DIR, exist_ok=True)
os.makedirs(_EXP_DIR, exist_ok=True)


# ═══════════════════════════════════════════════════════════════════════════
# КОНФИГУРАЦИЯ ЭКСПЕРИМЕНТА
# ═══════════════════════════════════════════════════════════════════════════
N_TRAIN_EPISODES = 500
N_EVAL_EPISODES  = 50
SCENARIOS        = ["tuapse", "serp"]
SCENARIO_NAMES   = {"tuapse": "Сценарий А (ранг №4)", "serp": "Сценарий Б (ранг №2)"}
SEED             = 42
STEP_DT          = 5   # мин


def print_header(text):
    print(f"\n{'═'*70}")
    print(f"  {text}")
    print(f"{'═'*70}")


# ═══════════════════════════════════════════════════════════════════════════
# ЭТАП 1: ОБУЧЕНИЕ ТАБЛИЧНОГО ОП-АГЕНТА
# ═══════════════════════════════════════════════════════════════════════════
def train_rl_agent(n_episodes=N_TRAIN_EPISODES, scenario="tuapse"):
    """Обучить табличного Q-learning агента."""
    print_header(f"ЭТАП 1: Обучение ОП-агента ({n_episodes} эпизодов, {SCENARIO_NAMES[scenario]})")

    sim = TankFireSim(seed=SEED, training=True, scenario=scenario)
    rewards = []

    for ep in range(n_episodes):
        sim.reset()
        ep_reward = 0
        steps = 0
        while sim.t < sim._cfg["total_min"] and not sim.extinguished:
            snap = sim.step(dt=STEP_DT)
            ep_reward += sim.h_reward[-1] if sim.h_reward else 0
            steps += 1
            if steps > 2000:
                break
        rewards.append(ep_reward)
        sim.agent.end_episode()

        if (ep + 1) % 50 == 0:
            avg = np.mean(rewards[-50:])
            print(f"  Эпизод {ep+1:>4d}/{n_episodes}: "
                  f"ε={sim.agent.epsilon:.3f}  "
                  f"Σr(последние 50)={avg:.1f}  "
                  f"ликвидирован={sim.extinguished}")

    print(f"\n  Обучение завершено:")
    print(f"    Финальный ε: {sim.agent.epsilon:.4f}")
    print(f"    Средняя награда (последние 100): {np.mean(rewards[-100:]):.1f}")
    return sim, rewards


# ═══════════════════════════════════════════════════════════════════════════
# ЭТАП 2: СРАВНИТЕЛЬНЫЙ ПРОГОН
# ═══════════════════════════════════════════════════════════════════════════
def evaluate_agent(agent_name, action_fn, scenario, n_episodes=N_EVAL_EPISODES):
    """Прогнать агента N эпизодов с вариацией начальных условий.

    Стохастика: каждый эпизод использует уникальный seed, который влияет
    на начальный шум параметров пожара (±10% площади, ±15% расхода).
    Это моделирует неопределённость реальных условий.
    """
    results = []
    for ep in range(n_episodes):
        ep_seed = SEED + ep * 137 + 7
        sim = TankFireSim(seed=ep_seed, training=False, scenario=scenario)
        # Вариация начальных условий (±10-15%) для стохастики
        rng = np.random.RandomState(ep_seed)
        sim.fire_area *= rng.uniform(0.85, 1.15)
        sim.water_flow *= rng.uniform(0.85, 1.15)
        sim.foam_conc *= rng.uniform(0.80, 1.20)
        steps = 0
        trace = []
        while sim.t < sim._cfg["total_min"] and not sim.extinguished:
            action = action_fn(sim)
            snap = sim.step(dt=STEP_DT, action=action)
            trace.append((sim.t, action, sim.phase))
            steps += 1
            if steps > 2000:
                break

        total_reward = sum(sim.h_reward)
        risk_max = max((r for _, r in sim.h_risk), default=0)
        fire_final = sim.fire_area
        fire_init = sim._cfg["initial_fire_area"]
        area_reduction = (fire_init - fire_final) / fire_init if fire_init > 0 else 0

        results.append({
            "episode": ep,
            "extinguished": sim.extinguished,
            "localized": sim.localized,
            "t_final": sim.t,
            "steps": steps,
            "total_reward": total_reward,
            "foam_attacks": sim.foam_attacks,
            "risk_max": risk_max,
            "area_reduction": area_reduction,
            "fire_area_final": fire_final,
            "trace": trace,
        })

    return results


def run_comparison(trained_sim):
    """Провести полное сравнение ЭС vs ОП на обоих сценариях."""
    print_header("ЭТАП 2: Сравнительный прогон")

    es = ExpertSystem()
    all_results = {}

    for scenario in SCENARIOS:
        print(f"\n  {SCENARIO_NAMES[scenario]}:")

        # ── ЭС ────────────────────────────────────────────────────────────
        def es_action(sim):
            a, _, _ = es.select_action_from_sim(sim)
            return a

        es_results = evaluate_agent("ЭС", es_action, scenario)
        all_results[f"es_{scenario}"] = es_results

        es_success = np.mean([r["extinguished"] for r in es_results])
        es_reward = np.mean([r["total_reward"] for r in es_results])
        print(f"    ЭС:        успешность={es_success:.0%}, "
              f"Σr={es_reward:.1f}, "
              f"пен.атак={np.mean([r['foam_attacks'] for r in es_results]):.1f}")

        # ── ОП (обученный) ─────────────────────────────────────────────────
        trained_agent = trained_sim.agent

        def rl_action(sim):
            state = sim._state()
            mask = sim._mask()
            return trained_agent.select_action(state, training=False, mask=mask)

        rl_results = evaluate_agent("ОП", rl_action, scenario)
        all_results[f"rl_{scenario}"] = rl_results

        rl_success = np.mean([r["extinguished"] for r in rl_results])
        rl_reward = np.mean([r["total_reward"] for r in rl_results])
        print(f"    ОП:        успешность={rl_success:.0%}, "
              f"Σr={rl_reward:.1f}, "
              f"пен.атак={np.mean([r['foam_attacks'] for r in rl_results]):.1f}")

    return all_results


# ═══════════════════════════════════════════════════════════════════════════
# ЭТАП 3: СТАТИСТИЧЕСКАЯ ОЦЕНКА
# ═══════════════════════════════════════════════════════════════════════════
def statistical_analysis(all_results):
    """Статистическое сравнение ЭС vs ОП."""
    print_header("ЭТАП 3: Статистическая оценка")

    stats = {}
    metrics = ["total_reward", "foam_attacks", "risk_max", "t_final", "area_reduction"]
    metric_names = {
        "total_reward": "Накопленная награда",
        "foam_attacks": "Пенных атак",
        "risk_max": "Макс. риск",
        "t_final": "Время ликвидации (мин)",
        "area_reduction": "Сокращение площади",
    }

    for scenario in SCENARIOS:
        es_data = all_results[f"es_{scenario}"]
        rl_data = all_results[f"rl_{scenario}"]

        print(f"\n  {SCENARIO_NAMES[scenario]}:")
        print(f"  {'Метрика':<25s} {'ЭС: M±SD':>18s} {'ОП: M±SD':>18s} "
              f"{'p-знач.':>10s} {'d Коэна':>10s} {'Знач.':>6s}")
        print(f"  {'─'*95}")

        scenario_stats = {}
        for m in metrics:
            es_vals = np.array([r[m] for r in es_data], dtype=float)
            rl_vals = np.array([r[m] for r in rl_data], dtype=float)

            # Манн-Уитни
            try:
                u_stat, p_val = sp_stats.mannwhitneyu(es_vals, rl_vals,
                                                       alternative="two-sided")
            except Exception:
                u_stat, p_val = 0, 1.0

            # d Коэна
            pooled_std = np.sqrt((np.var(es_vals) + np.var(rl_vals)) / 2)
            cohens_d = (np.mean(rl_vals) - np.mean(es_vals)) / pooled_std \
                if pooled_std > 1e-8 else 0

            # Бутстреп 95% ДИ для разности средних
            n_boot = 10000
            rng = np.random.RandomState(SEED)
            boot_diffs = []
            for _ in range(n_boot):
                es_boot = rng.choice(es_vals, len(es_vals), replace=True)
                rl_boot = rng.choice(rl_vals, len(rl_vals), replace=True)
                boot_diffs.append(np.mean(rl_boot) - np.mean(es_boot))
            ci_lo = np.percentile(boot_diffs, 2.5)
            ci_hi = np.percentile(boot_diffs, 97.5)

            sig = "✓" if p_val < 0.05 else "✗"
            es_str = f"{np.mean(es_vals):.2f}±{np.std(es_vals):.2f}"
            rl_str = f"{np.mean(rl_vals):.2f}±{np.std(rl_vals):.2f}"
            print(f"  {metric_names[m]:<25s} {es_str:>18s} {rl_str:>18s} "
                  f"{p_val:>10.4f} {cohens_d:>+10.3f} {sig:>6s}")

            scenario_stats[m] = {
                "es_mean": float(np.mean(es_vals)),
                "es_std": float(np.std(es_vals)),
                "rl_mean": float(np.mean(rl_vals)),
                "rl_std": float(np.std(rl_vals)),
                "p_value": float(p_val),
                "cohens_d": float(cohens_d),
                "significant": p_val < 0.05,
                "ci95_diff": [float(ci_lo), float(ci_hi)],
                "u_statistic": float(u_stat),
            }

        # Успешность (бинарная)
        es_succ = np.mean([r["extinguished"] for r in es_data])
        rl_succ = np.mean([r["extinguished"] for r in rl_data])
        scenario_stats["success_rate"] = {
            "es": float(es_succ), "rl": float(rl_succ),
        }
        print(f"  {'Успешность':<25s} {es_succ:>18.0%} {rl_succ:>18.0%}")

        stats[scenario] = scenario_stats

    return stats


# ═══════════════════════════════════════════════════════════════════════════
# ЭТАП 4: ВИЗУАЛИЗАЦИИ
# ═══════════════════════════════════════════════════════════════════════════
def generate_visualizations(trained_sim, all_results, stats):
    """Создать все визуализации для статьи."""
    print_header("ЭТАП 4: Визуализации")

    paths = {}

    # 1. Тепловая карта политик
    expert_policy = policy_from_expert_rules()
    rl_policy = policy_from_q_table(trained_sim.agent.Q)
    paths["heatmap"] = policy_heatmap(
        {"Экспертная система": expert_policy,
         "Табличный агент ОП": rl_policy},
        title="Сравнение политик: ЭС vs ОП (фаза × действие)",
        filename="exp_policy_heatmap.png")
    print(f"  Тепловая карта: {paths['heatmap']}")

    # 2. Поток решений (из трасс сценария А)
    es_trace = all_results["es_tuapse"][0]["trace"]
    rl_trace = all_results["rl_tuapse"][0]["trace"]
    paths["flow"] = decision_flow(
        {"Экспертная система": es_trace,
         "Табличный агент ОП": rl_trace},
        title="Поток решений по времени (сценарий А)",
        filename="exp_decision_flow.png")
    print(f"  Поток решений: {paths['flow']}")

    # 3. Радарная диаграмма
    def _norm(vals, reverse=False):
        mn, mx = min(vals), max(vals)
        if mx == mn:
            return [0.5] * len(vals)
        n = [(v - mn) / (mx - mn) for v in vals]
        return [1 - x for x in n] if reverse else n

    es_a = all_results["es_tuapse"]
    rl_a = all_results["rl_tuapse"]

    radar_data = {
        "Экспертная система": {
            "Успешность": np.mean([r["extinguished"] for r in es_a]),
            "Скорость\nликвидации": 1 - min(1, np.mean([r["t_final"] for r in es_a]) / 5000),
            "Экономия\nресурсов": 1 - min(1, np.mean([r["foam_attacks"] for r in es_a]) / 6),
            "Безопасность\nЛС": 1 - np.mean([r["risk_max"] for r in es_a]),
            "Эффективность\nпены": np.mean([r["area_reduction"] for r in es_a]),
        },
        "Табличный агент ОП": {
            "Успешность": np.mean([r["extinguished"] for r in rl_a]),
            "Скорость\nликвидации": 1 - min(1, np.mean([r["t_final"] for r in rl_a]) / 5000),
            "Экономия\nресурсов": 1 - min(1, np.mean([r["foam_attacks"] for r in rl_a]) / 6),
            "Безопасность\nЛС": 1 - np.mean([r["risk_max"] for r in rl_a]),
            "Эффективность\nпены": np.mean([r["area_reduction"] for r in rl_a]),
        },
    }
    paths["radar"] = radar_chart(radar_data,
        title="Профиль агентов по ключевым метрикам (сценарий А)",
        filename="exp_radar.png")
    print(f"  Радарная диаграмма: {paths['radar']}")

    # 4. Кривая обучения
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 4), facecolor="#f5f6fa")
    ax.set_facecolor("#fafafa")
    # Используем rewards из обучения (сохраним их)
    rewards = trained_sim.agent.episode_rewards
    if rewards:
        ax.plot(rewards, color="#bdc3c7", linewidth=0.5, alpha=0.5, label="По эпизодам")
        window = min(50, len(rewards) // 3)
        if window > 1:
            ma = np.convolve(rewards, np.ones(window)/window, mode="valid")
            ax.plot(range(window-1, len(rewards)), ma, color="#c0392b",
                    linewidth=2, label=f"Скользящее среднее ({window})")
    ax.set_xlabel("Эпизод", fontsize=10)
    ax.set_ylabel("Накопленная награда", fontsize=10)
    ax.set_title("Кривая обучения табличного агента ОП", fontsize=11, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    p = os.path.join(_FIG_DIR, "exp_learning_curve.png")
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    paths["learning"] = p
    print(f"  Кривая обучения: {p}")

    # 5. Столбчатая: сравнение метрик
    fig, axes = plt.subplots(1, 3, figsize=(14, 4), facecolor="#f5f6fa")
    for ax_i, (metric, title) in enumerate([
        ("total_reward", "Накопленная награда"),
        ("foam_attacks", "Число пенных атак"),
        ("risk_max", "Максимальный риск"),
    ]):
        ax = axes[ax_i]
        for si, scenario in enumerate(SCENARIOS):
            s = stats[scenario][metric]
            x = si * 3
            ax.bar(x, s["es_mean"], 0.8, yerr=s["es_std"], capsize=3,
                   color="#3498db", alpha=0.8, label="ЭС" if si == 0 else "")
            ax.bar(x + 1, s["rl_mean"], 0.8, yerr=s["rl_std"], capsize=3,
                   color="#c0392b", alpha=0.8, label="ОП" if si == 0 else "")
            # Значимость
            if s["significant"]:
                ax.text(x + 0.5, max(s["es_mean"], s["rl_mean"]) + s["es_std"] * 1.2,
                        f"p={s['p_value']:.3f}*", ha="center", fontsize=7,
                        fontweight="bold", color="#27ae60")

        ax.set_xticks([0.5, 3.5])
        ax.set_xticklabels(["Сценарий А", "Сценарий Б"], fontsize=9)
        ax.set_title(title, fontsize=10, fontweight="bold")
        ax.legend(fontsize=8)
        ax.grid(True, axis="y", alpha=0.3)

    fig.suptitle("Сравнение ЭС vs ОП: ключевые метрики", fontsize=12, fontweight="bold")
    fig.tight_layout()
    p = os.path.join(_FIG_DIR, "exp_comparison_bars.png")
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    paths["bars"] = p
    print(f"  Столбчатая: {p}")

    return paths


# ═══════════════════════════════════════════════════════════════════════════
# ЭТАП 5: СОХРАНЕНИЕ РЕЗУЛЬТАТОВ
# ═══════════════════════════════════════════════════════════════════════════
def save_results(stats, train_rewards, paths):
    """Сохранить все результаты в JSON."""
    print_header("ЭТАП 5: Сохранение результатов")

    result = {
        "timestamp": datetime.datetime.now().isoformat(timespec="seconds"),
        "config": {
            "n_train_episodes": N_TRAIN_EPISODES,
            "n_eval_episodes": N_EVAL_EPISODES,
            "scenarios": SCENARIOS,
            "seed": SEED,
            "step_dt": STEP_DT,
        },
        "training": {
            "final_reward_ma50": float(np.mean(train_rewards[-50:])),
            "total_episodes": len(train_rewards),
        },
        "statistics": stats,
        "figures": paths,
    }

    json_path = os.path.join(_EXP_DIR, "experiment_results.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2, default=str)
    print(f"  Результаты: {json_path}")

    # Сохранить в централизованную БД
    try:
        from results_db import get_db
        db = get_db()
        for scenario in SCENARIOS:
            s = stats[scenario]
            db.log_experiment(
                mode="experiment_es_vs_rl",
                scenario=scenario,
                duration_min=0,
                extinguished=True,
                es_reward=s["total_reward"]["es_mean"],
                rl_reward=s["total_reward"]["rl_mean"],
                p_value=s["total_reward"]["p_value"],
                cohens_d=s["total_reward"]["cohens_d"],
            )
    except Exception:
        pass

    return json_path


# ═══════════════════════════════════════════════════════════════════════════
# ГЛАВНАЯ ФУНКЦИЯ
# ═══════════════════════════════════════════════════════════════════════════
def main():
    t0 = time.time()
    print("╔══════════════════════════════════════════════════════════════════╗")
    print("║  ЭКСПЕРИМЕНТ: ЭКСПЕРТНАЯ СИСТЕМА VS ОБУЧЕНИЕ С ПОДКРЕПЛЕНИЕМ   ║")
    print("║  Платформа САУР-ПСП                                            ║")
    print("╚══════════════════════════════════════════════════════════════════╝")

    # 1. Обучение
    trained_sim, train_rewards = train_rl_agent(N_TRAIN_EPISODES, "tuapse")

    # 2. Сравнительный прогон
    all_results = run_comparison(trained_sim)

    # 3. Статистика
    stats = statistical_analysis(all_results)

    # 4. Визуализации
    paths = generate_visualizations(trained_sim, all_results, stats)

    # 5. Сохранение
    json_path = save_results(stats, train_rewards, paths)

    elapsed = time.time() - t0
    print_header(f"ЭКСПЕРИМЕНТ ЗАВЕРШЁН за {elapsed:.1f} с")
    print(f"\n  Результаты: {json_path}")
    print(f"  Визуализации: {_FIG_DIR}")
    print(f"\n  Для генерации статьи: python generate_imrad_article.py")

    return stats, paths


if __name__ == "__main__":
    main()
