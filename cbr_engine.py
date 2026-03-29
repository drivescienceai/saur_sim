"""
cbr_engine.py — Прецедентный анализ и классификация ситуаций на пожаре.
═══════════════════════════════════════════════════════════════════════════════
Case-Based Reasoning (CBR): поиск похожих пожаров в базе прецедентов,
классификация ситуаций, кластеризация сценариев.

Компоненты:
  1. CaseBase       — база прецедентов с индексацией
  2. SituationClassifier — классификатор ситуаций (k-NN + деревья)
  3. ScenarioClusterer   — кластеризация сценариев (K-Means + иерарх.)
  4. PrecedentSearch     — поиск ближайших прецедентов для СППР
═══════════════════════════════════════════════════════════════════════════════
"""
from __future__ import annotations

import os
import json
import math
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Optional, Tuple

import numpy as np

try:
    from .rl_agent import N_ACTIONS
except ImportError:
    from rl_agent import N_ACTIONS

_DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")


# ═══════════════════════════════════════════════════════════════════════════
# ВЕКТОР ПРИЗНАКОВ СИТУАЦИИ
# ═══════════════════════════════════════════════════════════════════════════
FEATURE_NAMES = [
    "Объём РВС (м³)",
    "Диаметр РВС (м)",
    "Площадь пожара (м²)",
    "Ранг пожара",
    "Препятствие крыши",
    "Число пенных атак",
    "Доля успешных атак",
    "Число инцидентов",
    "Число решений РТП",
    "Длительность (мин)",
    "Тип топлива (код)",
    "Тип кровли (код)",
]
N_FEATURES = len(FEATURE_NAMES)

# Кодирование категориальных признаков
FUEL_CODES = {"бензин": 0, "нефть": 1, "дизель": 2, "мазут": 3,
              "керосин": 4, "газоконденсат": 5, "нефтепродукт": 6}
ROOF_CODES = {"плавающая": 0, "конусная": 1, "понтонная": 2,
              "стационарная": 3, "неизвестно": 4}


@dataclass
class FireCase:
    """Один прецедент (описание реального пожара)."""
    case_id: str
    source_file: str = ""

    # Вектор признаков
    features: np.ndarray = field(default_factory=lambda: np.zeros(N_FEATURES))

    # Метаданные
    rvs_volume: float = 0.0
    fire_rank: int = 0
    fuel_type: str = ""
    roof_type: str = ""
    duration_min: int = 0
    localized: bool = False
    extinguished: bool = False
    foam_attacks: int = 0
    foam_success: int = 0
    n_incidents: int = 0
    n_decisions: int = 0

    # Решения РТП (для ОП)
    decisions: List[Dict] = field(default_factory=list)

    # Кластер (заполняется после кластеризации)
    cluster_id: int = -1
    cluster_name: str = ""

    # Ситуационный тип (заполняется классификатором)
    situation_type: str = ""

    def to_dict(self) -> dict:
        d = asdict(self)
        d["features"] = self.features.tolist()
        return d


def scenario_to_case(scenario_data: dict, case_id: str = "") -> FireCase:
    """Конвертировать ParsedScenario.to_dict() → FireCase."""
    fuel = scenario_data.get("fuel_type", "нефтепродукт")
    roof = scenario_data.get("roof_type", "неизвестно")
    foam_details = scenario_data.get("foam_attack_details", [])
    foam_total = scenario_data.get("foam_attacks_total",
                                   len(foam_details))
    foam_ok = scenario_data.get("foam_attacks_successful",
                                sum(1 for f in foam_details
                                    if f.get("result") == "успех"))
    incidents = scenario_data.get("incidents", [])
    decisions = scenario_data.get("rtp_decisions", [])
    duration = scenario_data.get("total_duration_min", 0)
    volume = scenario_data.get("rvs_volume_m3", 0)
    diameter = scenario_data.get("rvs_diameter_m", 0)
    area = scenario_data.get("initial_fire_area_m2", 0)
    rank = scenario_data.get("fire_rank", 0)
    obstruction = scenario_data.get("roof_obstruction", 0)

    features = np.array([
        volume,
        diameter,
        area,
        rank,
        obstruction,
        foam_total,
        foam_ok / max(foam_total, 1),
        len(incidents),
        len(decisions),
        duration,
        FUEL_CODES.get(fuel, 6),
        ROOF_CODES.get(roof, 4),
    ], dtype=float)

    return FireCase(
        case_id=case_id or scenario_data.get("source_file", ""),
        source_file=scenario_data.get("source_file", ""),
        features=features,
        rvs_volume=volume,
        fire_rank=rank,
        fuel_type=fuel,
        roof_type=roof,
        duration_min=duration,
        localized=scenario_data.get("localized", False),
        extinguished=scenario_data.get("extinguished", False),
        foam_attacks=foam_total,
        foam_success=foam_ok,
        n_incidents=len(incidents),
        n_decisions=len(decisions),
        decisions=decisions,
    )


# ═══════════════════════════════════════════════════════════════════════════
# 1. БАЗА ПРЕЦЕДЕНТОВ
# ═══════════════════════════════════════════════════════════════════════════
class CaseBase:
    """База прецедентов с индексацией и поиском."""

    def __init__(self):
        self.cases: List[FireCase] = []
        self._feature_matrix: Optional[np.ndarray] = None
        self._norm_params: Optional[Tuple[np.ndarray, np.ndarray]] = None

    def add(self, case: FireCase):
        self.cases.append(case)
        self._feature_matrix = None  # инвалидировать кэш

    def add_from_scenario(self, scenario_data: dict, case_id: str = ""):
        self.add(scenario_to_case(scenario_data, case_id))

    def load_from_folder(self, scenarios_dir: str):
        """Загрузить все JSON-сценарии из папки."""
        if not os.path.isdir(scenarios_dir):
            return
        for filename in sorted(os.listdir(scenarios_dir)):
            if not filename.endswith(".json"):
                continue
            filepath = os.path.join(scenarios_dir, filename)
            try:
                with open(filepath, "r", encoding="utf-8") as f:
                    data = json.load(f)
                case_id = os.path.splitext(filename)[0]
                self.add_from_scenario(data, case_id)
            except Exception:
                continue

    def __len__(self):
        return len(self.cases)

    def _build_matrix(self):
        """Построить матрицу признаков + нормализация."""
        if not self.cases:
            self._feature_matrix = np.zeros((0, N_FEATURES))
            self._norm_params = (np.zeros(N_FEATURES), np.ones(N_FEATURES))
            return
        self._feature_matrix = np.array([c.features for c in self.cases])
        mu = self._feature_matrix.mean(axis=0)
        sigma = self._feature_matrix.std(axis=0)
        sigma[sigma < 1e-8] = 1.0
        self._norm_params = (mu, sigma)

    def _normalized(self) -> np.ndarray:
        """Нормализованная матрица (Z-score)."""
        if self._feature_matrix is None:
            self._build_matrix()
        mu, sigma = self._norm_params
        return (self._feature_matrix - mu) / sigma

    def save(self, path: str = ""):
        if not path:
            path = os.path.join(_DATA_DIR, "case_base.json")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump([c.to_dict() for c in self.cases],
                      f, ensure_ascii=False, indent=1)

    def load(self, path: str = ""):
        if not path:
            path = os.path.join(_DATA_DIR, "case_base.json")
        if not os.path.exists(path):
            return False
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.cases = []
        for d in data:
            c = FireCase(case_id=d["case_id"])
            c.features = np.array(d["features"])
            for k in ["source_file", "rvs_volume", "fire_rank", "fuel_type",
                       "roof_type", "duration_min", "localized", "extinguished",
                       "foam_attacks", "foam_success", "n_incidents",
                       "n_decisions", "decisions", "cluster_id",
                       "cluster_name", "situation_type"]:
                if k in d:
                    setattr(c, k, d[k])
            self.cases.append(c)
        self._feature_matrix = None
        return True


# ═══════════════════════════════════════════════════════════════════════════
# 2. ПОИСК БЛИЖАЙШИХ ПРЕЦЕДЕНТОВ
# ═══════════════════════════════════════════════════════════════════════════
class PrecedentSearch:
    """Поиск k ближайших прецедентов для текущей ситуации."""

    # Весá признаков (какие важнее для сходства)
    WEIGHTS = np.array([
        0.15,  # объём
        0.10,  # диаметр
        0.15,  # площадь
        0.20,  # ранг (самый важный — определяет масштаб)
        0.15,  # препятствие крыши
        0.05,  # пенные атаки
        0.05,  # доля успешных
        0.03,  # инциденты
        0.02,  # решения
        0.05,  # длительность
        0.03,  # топливо
        0.02,  # кровля
    ], dtype=float)

    def __init__(self, case_base: CaseBase):
        self.cb = case_base

    def find_similar(self, query_features: np.ndarray,
                     k: int = 5) -> List[Tuple[FireCase, float]]:
        """Найти k ближайших прецедентов.

        Возвращает: [(case, distance), ...] отсортировано по близости.
        """
        if not self.cb.cases:
            return []

        if self.cb._feature_matrix is None:
            self.cb._build_matrix()

        mu, sigma = self.cb._norm_params
        q_norm = (query_features - mu) / sigma
        X_norm = self.cb._normalized()

        # Взвешенное евклидово расстояние
        diff = X_norm - q_norm
        w_diff = diff * self.WEIGHTS
        distances = np.sqrt((w_diff ** 2).sum(axis=1))

        top_k = np.argsort(distances)[:k]
        return [(self.cb.cases[i], float(distances[i])) for i in top_k]

    def find_for_situation(self, rvs_volume: float, fire_area: float,
                           fire_rank: int, fuel: str = "нефтепродукт",
                           roof_obstruction: float = 0.0,
                           k: int = 5) -> List[Tuple[FireCase, float]]:
        """Поиск по параметрам текущей ситуации."""
        import math
        diameter = 2 * math.sqrt(rvs_volume / (math.pi * 12)) if rvs_volume > 0 else 20.0
        query = np.array([
            rvs_volume, diameter, fire_area, fire_rank,
            roof_obstruction, 0, 0, 0, 0, 0,
            FUEL_CODES.get(fuel, 6),
            0,  # кровля неизвестна
        ], dtype=float)
        return self.find_similar(query, k)

    def recommend_from_precedent(self, case: FireCase) -> Dict:
        """Извлечь рекомендацию из прецедента."""
        if not case.decisions:
            return {"действие": "О4 (разведка)", "обоснование": "Нет данных"}

        # Самое частое действие в прецеденте
        action_counts = {}
        for d in case.decisions:
            code = d.get("action_code", "О4")
            action_counts[code] = action_counts.get(code, 0) + 1
        top_action = max(action_counts, key=action_counts.get)

        return {
            "действие": top_action,
            "прецедент": case.case_id,
            "ранг": case.fire_rank,
            "топливо": case.fuel_type,
            "длительность": f"{case.duration_min} мин",
            "исход": "ликвидирован" if case.extinguished else "не ликвидирован",
            "пенных_атак": case.foam_attacks,
            "успешных": case.foam_success,
            "обоснование": f"В аналогичном случае ({case.case_id}, "
                           f"V={case.rvs_volume:.0f} м³, {case.fuel_type}) "
                           f"действие {top_action} применялось "
                           f"{action_counts[top_action]} раз",
        }


# ═══════════════════════════════════════════════════════════════════════════
# 3. КЛАСТЕРИЗАЦИЯ СЦЕНАРИЕВ
# ═══════════════════════════════════════════════════════════════════════════
class ScenarioClusterer:
    """Кластеризация сценариев пожаров (K-Means)."""

    # Названия кластеров (автоматически по центроидам)
    CLUSTER_TEMPLATES = {
        "small_simple": "Малые РВС, простой сценарий",
        "medium_standard": "Средние РВС, стандартный сценарий",
        "large_complex": "Крупные РВС, сложный сценарий",
        "critical": "Критические сценарии (вскипание, розливы)",
    }

    def __init__(self, n_clusters: int = 4, seed: int = 42):
        self.n_clusters = n_clusters
        self.seed = seed
        self.centroids: Optional[np.ndarray] = None
        self.labels: Optional[np.ndarray] = None
        self.cluster_names: Dict[int, str] = {}

    def fit(self, case_base: CaseBase) -> Dict:
        """Кластеризовать все прецеденты.

        Возвращает: {n_clusters, cluster_sizes, cluster_profiles}
        """
        if len(case_base) < self.n_clusters:
            return {"n_clusters": 0, "error": "Мало данных"}

        X = case_base._normalized()
        n = X.shape[0]

        # K-Means (без sklearn — чистый numpy)
        rng = np.random.RandomState(self.seed)
        indices = rng.choice(n, self.n_clusters, replace=False)
        centroids = X[indices].copy()

        for _ in range(100):
            # Assign
            dists = np.array([
                np.sqrt(((X - c) ** 2).sum(axis=1))
                for c in centroids
            ])  # (K, N)
            labels = dists.argmin(axis=0)

            # Update
            new_centroids = np.zeros_like(centroids)
            for k in range(self.n_clusters):
                mask = labels == k
                if mask.any():
                    new_centroids[k] = X[mask].mean(axis=0)
                else:
                    new_centroids[k] = centroids[k]

            if np.allclose(centroids, new_centroids, atol=1e-6):
                break
            centroids = new_centroids

        self.centroids = centroids
        self.labels = labels

        # Назначить имена кластерам по характеристикам
        mu, sigma = case_base._norm_params
        real_centroids = centroids * sigma + mu  # денормализация

        for k in range(self.n_clusters):
            vol = real_centroids[k, 0]
            rank = real_centroids[k, 3]
            incidents = real_centroids[k, 7]
            if rank >= 4 or incidents >= 3:
                name = f"Кластер {k+1}: Сложные (ранг≥{rank:.0f})"
            elif vol >= 10000:
                name = f"Кластер {k+1}: Крупные РВС (V≥{vol:.0f} м³)"
            elif vol >= 3000:
                name = f"Кластер {k+1}: Средние РВС"
            else:
                name = f"Кластер {k+1}: Малые РВС (V<{vol:.0f} м³)"
            self.cluster_names[k] = name

        # Присвоить кластеры прецедентам
        for i, case in enumerate(case_base.cases):
            case.cluster_id = int(labels[i])
            case.cluster_name = self.cluster_names.get(int(labels[i]), "")

        # Результат
        cluster_sizes = {self.cluster_names.get(k, f"Кластер {k}"):
                         int((labels == k).sum())
                         for k in range(self.n_clusters)}

        return {
            "n_clusters": self.n_clusters,
            "cluster_sizes": cluster_sizes,
            "inertia": float(sum(
                ((X[labels == k] - centroids[k]) ** 2).sum()
                for k in range(self.n_clusters)
            )),
        }

    def predict(self, features: np.ndarray,
                case_base: CaseBase) -> Tuple[int, str]:
        """Определить кластер для нового вектора признаков."""
        if self.centroids is None:
            return -1, "Не обучен"
        mu, sigma = case_base._norm_params
        x_norm = (features - mu) / sigma
        dists = np.sqrt(((self.centroids - x_norm) ** 2).sum(axis=1))
        k = int(dists.argmin())
        return k, self.cluster_names.get(k, f"Кластер {k}")


# ═══════════════════════════════════════════════════════════════════════════
# 4. КЛАССИФИКАТОР СИТУАЦИЙ (k-NN)
# ═══════════════════════════════════════════════════════════════════════════
SITUATION_TYPES = [
    "Начальная стадия (обнаружение)",
    "Развёртывание (сосредоточение сил)",
    "Активное горение (стабильное)",
    "Активное горение (кризисное)",
    "Готовность к пенной атаке",
    "Пенная атака в ходу",
    "Локализация",
    "Ликвидация последствий",
]


class SituationClassifier:
    """Классификатор ситуаций на пожаре (k-NN по прецедентам).

    Определяет подтип ситуации внутри фазы S1–S5 на основе
    комплекса признаков (площадь, ресурсы, время, инциденты).
    """

    def __init__(self, k: int = 5):
        self.k = k
        self._training_X: Optional[np.ndarray] = None
        self._training_y: Optional[np.ndarray] = None
        self._labels: List[str] = []

    def fit(self, situations: List[Tuple[np.ndarray, str]]):
        """Обучить на размеченных ситуациях.

        situations: [(feature_vector, situation_type), ...]
        """
        if not situations:
            return
        self._training_X = np.array([s[0] for s in situations])
        self._labels = [s[1] for s in situations]
        label_set = sorted(set(self._labels))
        label_map = {l: i for i, l in enumerate(label_set)}
        self._training_y = np.array([label_map[l] for l in self._labels])
        self._label_names = label_set

    def predict(self, features: np.ndarray) -> Tuple[str, float]:
        """Классифицировать ситуацию.

        Возвращает: (тип_ситуации, уверенность)
        """
        if self._training_X is None:
            return "Неизвестно", 0.0

        # Нормализация
        mu = self._training_X.mean(axis=0)
        sigma = self._training_X.std(axis=0)
        sigma[sigma < 1e-8] = 1.0

        X_norm = (self._training_X - mu) / sigma
        q_norm = (features - mu) / sigma

        dists = np.sqrt(((X_norm - q_norm) ** 2).sum(axis=1))
        top_k = np.argsort(dists)[:self.k]

        # Голосование
        votes = {}
        for idx in top_k:
            label = self._labels[idx]
            weight = 1.0 / (dists[idx] + 1e-8)
            votes[label] = votes.get(label, 0) + weight

        best = max(votes, key=votes.get)
        confidence = votes[best] / sum(votes.values())
        return best, float(confidence)

    def fit_from_casebase(self, case_base: CaseBase):
        """Автоматическая разметка и обучение из базы прецедентов.

        Эвристика: тип ситуации определяется по комбинации признаков.
        """
        situations = []
        for case in case_base.cases:
            f = case.features
            volume, area = f[0], f[2]
            rank, obstruction = f[3], f[4]
            foam_attacks, incidents = f[5], f[7]
            duration = f[9]

            # Эвристическая классификация
            if duration <= 30:
                sit_type = "Начальная стадия (обнаружение)"
            elif foam_attacks == 0 and duration <= 120:
                sit_type = "Развёртывание (сосредоточение сил)"
            elif incidents >= 2:
                sit_type = "Активное горение (кризисное)"
            elif foam_attacks >= 1 and not case.extinguished:
                sit_type = "Пенная атака в ходу"
            elif case.localized and not case.extinguished:
                sit_type = "Локализация"
            elif case.extinguished:
                sit_type = "Ликвидация последствий"
            elif obstruction >= 0.5:
                sit_type = "Активное горение (кризисное)"
            else:
                sit_type = "Активное горение (стабильное)"

            case.situation_type = sit_type
            situations.append((f, sit_type))

        self.fit(situations)
        return len(situations)


# ═══════════════════════════════════════════════════════════════════════════
# ДЕМОНСТРАЦИЯ
# ═══════════════════════════════════════════════════════════════════════════
def generate_demo_casebase(n: int = 50, seed: int = 42) -> CaseBase:
    """Создать демо-базу прецедентов с синтетическими данными."""
    rng = np.random.RandomState(seed)
    cb = CaseBase()

    volumes = [1000, 2000, 3000, 5000, 10000, 15000, 20000, 30000, 50000]
    fuels = ["бензин", "нефть", "дизель", "мазут"]
    roofs = ["конусная", "плавающая", "стационарная"]

    for i in range(n):
        vol = rng.choice(volumes)
        fuel = rng.choice(fuels)
        roof = rng.choice(roofs)
        rank = min(5, max(1, int(math.log2(vol / 500)) + rng.randint(-1, 2)))
        diameter = 2 * math.sqrt(vol / (math.pi * 12))
        area = math.pi * (diameter / 2) ** 2 * rng.uniform(0.3, 1.0)
        obstruction = 0.70 if roof == "плавающая" else (0.30 if roof == "стационарная" else 0.0)
        duration = int(rank * rng.uniform(30, 200) + rng.uniform(0, 120))
        foam_attacks = max(0, rng.poisson(rank * 0.8))
        foam_ok = min(foam_attacks, max(0, rng.poisson(0.5)))
        incidents = max(0, rng.poisson(rank * 0.3))
        decisions = max(3, rng.poisson(rank * 3))

        features = np.array([
            vol, diameter, area, rank, obstruction,
            foam_attacks, foam_ok / max(foam_attacks, 1),
            incidents, decisions, duration,
            FUEL_CODES.get(fuel, 6), ROOF_CODES.get(roof, 4),
        ])

        case = FireCase(
            case_id=f"case_{i+1:03d}",
            features=features,
            rvs_volume=vol,
            fire_rank=rank,
            fuel_type=fuel,
            roof_type=roof,
            duration_min=duration,
            localized=rng.random() > 0.2,
            extinguished=rng.random() > 0.15,
            foam_attacks=foam_attacks,
            foam_success=foam_ok,
            n_incidents=incidents,
            n_decisions=decisions,
        )
        cb.add(case)

    return cb


if __name__ == "__main__":
    cb = generate_demo_casebase(50)
    print(f"База прецедентов: {len(cb)} случаев")

    # Кластеризация
    clusterer = ScenarioClusterer(n_clusters=4)
    result = clusterer.fit(cb)
    print(f"\nКластеризация ({result['n_clusters']} кластеров):")
    for name, size in result["cluster_sizes"].items():
        print(f"  {name}: {size} случаев")

    # Классификация
    classifier = SituationClassifier()
    n_trained = classifier.fit_from_casebase(cb)
    print(f"\nКлассификатор обучен на {n_trained} ситуациях")

    # Поиск прецедентов
    search = PrecedentSearch(cb)
    similar = search.find_for_situation(
        rvs_volume=20000, fire_area=1250, fire_rank=4, fuel="бензин",
        roof_obstruction=0.70, k=3)
    print(f"\n3 ближайших прецедента для РВС 20000 м³, ранг 4:")
    for case, dist in similar:
        print(f"  {case.case_id}: V={case.rvs_volume:.0f} м³, "
              f"ранг {case.fire_rank}, {case.fuel_type}, "
              f"d={dist:.2f}")
