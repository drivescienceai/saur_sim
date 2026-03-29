"""
results_db.py — Централизованное хранилище результатов САУР-ПСП.
═══════════════════════════════════════════════════════════════════════════════
Единая база данных для ВСЕХ результатов моделирования, анализа
и обучения. Решает проблему разрозненного хранения: каждый модуль
сохраняет результаты через единый интерфейс.

Хранилище: data/results_db.json
Структура:
  {
    "experiments": [...],    — прогоны симуляции
    "training":    [...],    — обучение агентов
    "analysis":    [...],    — статистический анализ
    "adaptation":  [...],    — метрики адаптации
    "autonomy":    [...],    — коэффициенты α
    "changepoints":[...],    — точки разладки
    "precedents":  [...],    — прецедентный анализ
    "multiagent":  [...],    — мультиагентные прогоны
  }
═══════════════════════════════════════════════════════════════════════════════
"""
from __future__ import annotations

import os
import json
import datetime
from typing import List, Dict, Optional, Any
from dataclasses import dataclass, asdict

_DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
_DB_PATH = os.path.join(_DATA_DIR, "results_db.json")

# Категории записей
CATEGORIES = [
    "experiments",     # прогоны симуляции (режим, сценарий, исход)
    "training",        # обучение агентов (эпизоды, reward, epsilon)
    "analysis",        # статистический анализ (корреляции, ANOVA, регрессия)
    "adaptation",      # метрики адаптации (J_L7, J_stability, переходы)
    "autonomy",        # коэффициенты α по уровням
    "changepoints",    # точки разладки (CUSUM, Байес)
    "precedents",      # прецедентный анализ (кластеры, поиск)
    "multiagent",      # мультиагентные прогоны (координация, α_НБУ)
    "irl",             # восстановленные веса IRL
    "calibration",     # калибровка полумарковской модели
]


class ResultsDB:
    """Централизованная база результатов."""

    def __init__(self, path: str = _DB_PATH):
        self.path = path
        self.data: Dict[str, List[Dict]] = {cat: [] for cat in CATEGORIES}
        self.load()

    def load(self) -> bool:
        if not os.path.exists(self.path):
            return False
        try:
            with open(self.path, "r", encoding="utf-8") as f:
                loaded = json.load(f)
            for cat in CATEGORIES:
                self.data[cat] = loaded.get(cat, [])
            return True
        except Exception:
            return False

    def save(self):
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        with open(self.path, "w", encoding="utf-8") as f:
            json.dump(self.data, f, ensure_ascii=False, indent=1,
                      default=str)

    def add(self, category: str, record: Dict):
        """Добавить запись в категорию."""
        if category not in self.data:
            self.data[category] = []
        record["_timestamp"] = datetime.datetime.now().isoformat(
            timespec="seconds")
        self.data[category].append(record)
        self.save()

    def get(self, category: str, last_n: int = 0) -> List[Dict]:
        """Получить записи из категории (last_n=0 → все)."""
        records = self.data.get(category, [])
        if last_n > 0:
            return records[-last_n:]
        return records

    def count(self, category: str) -> int:
        return len(self.data.get(category, []))

    def clear(self, category: str):
        self.data[category] = []
        self.save()

    def summary(self) -> Dict[str, int]:
        """Сводка: количество записей по категориям."""
        return {cat: len(records) for cat, records in self.data.items()
                if records}

    # ── Удобные методы для конкретных модулей ─────────────────────────────

    def log_experiment(self, mode: str, scenario: str,
                       duration_min: int, extinguished: bool,
                       **kwargs):
        """Записать результат прогона симуляции."""
        self.add("experiments", {
            "mode": mode, "scenario": scenario,
            "duration_min": duration_min,
            "extinguished": extinguished,
            **kwargs,
        })

    def log_training(self, agent_type: str, n_episodes: int,
                     final_reward: float, final_epsilon: float,
                     **kwargs):
        """Записать результат обучения агента."""
        self.add("training", {
            "agent_type": agent_type,
            "n_episodes": n_episodes,
            "final_reward": round(final_reward, 4),
            "final_epsilon": round(final_epsilon, 4),
            **kwargs,
        })

    def log_analysis(self, method: str, results: Dict, **kwargs):
        """Записать результат статистического анализа."""
        self.add("analysis", {
            "method": method,
            "results": results,
            **kwargs,
        })

    def log_adaptation(self, metrics: Dict, **kwargs):
        """Записать метрики адаптации."""
        self.add("adaptation", {**metrics, **kwargs})

    def log_autonomy(self, alpha_by_level: Dict[str, float],
                     scenario: str = "", **kwargs):
        """Записать коэффициенты автономности."""
        self.add("autonomy", {
            "alpha": alpha_by_level,
            "scenario": scenario,
            **kwargs,
        })

    def log_changepoints(self, series_name: str,
                         changepoints: List[Dict], **kwargs):
        """Записать обнаруженные точки разладки."""
        self.add("changepoints", {
            "series": series_name,
            "n_points": len(changepoints),
            "points": changepoints,
            **kwargs,
        })

    def log_multiagent(self, n_agents: int, autonomy: Dict[str, float],
                       decision_log_size: int, **kwargs):
        """Записать результат мультиагентного прогона."""
        self.add("multiagent", {
            "n_agents": n_agents,
            "autonomy": autonomy,
            "n_decisions": decision_log_size,
            **kwargs,
        })

    def log_irl(self, weights: List[float], feature_names: List[str],
                interpretation: Dict, **kwargs):
        """Записать результат обратного обучения."""
        self.add("irl", {
            "weights": weights,
            "features": feature_names,
            "interpretation": interpretation,
            **kwargs,
        })

    def log_calibration(self, weibull_params: Dict, quality: Dict,
                        **kwargs):
        """Записать результат калибровки."""
        self.add("calibration", {
            "weibull": weibull_params,
            "quality": quality,
            **kwargs,
        })


# Глобальный экземпляр (singleton)
_db: Optional[ResultsDB] = None


def get_db() -> ResultsDB:
    """Получить глобальный экземпляр базы результатов."""
    global _db
    if _db is None:
        _db = ResultsDB()
    return _db
