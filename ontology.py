"""
ontology.py — Онтология предметной области тушения пожаров РВС.
═══════════════════════════════════════════════════════════════════════════════
Формальная модель знаний: сущности, отношения, свойства и правила
предметной области «тушение пожара резервуара с нефтью и нефтепродуктами».

Отличие от оргструктуры (org_structure.py):
  Оргструктура — «кто кем управляет на конкретном пожаре»
  Онтология    — «какие сущности, отношения и правила существуют в
                  предметной области пожаротушения вообще»

Структура:
  1. Классы сущностей (объект, процесс, ресурс, должностное лицо, ...)
  2. Отношения между классами (управляет, использует, расположен, ...)
  3. Свойства (числовые, категориальные, булевы)
  4. Правила (аксиомы предметной области)
  5. Визуализация: граф онтологии, матрица отношений
  6. Экспорт в стандартные форматы (JSON-LD, RDF-подобный)
═══════════════════════════════════════════════════════════════════════════════
"""
from __future__ import annotations

import os
import json
import math
from dataclasses import dataclass, field
from typing import List, Dict, Set, Optional, Tuple
from enum import Enum

import numpy as np
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
# КЛАССЫ СУЩНОСТЕЙ
# ═══════════════════════════════════════════════════════════════════════════
class EntityType(str, Enum):
    """Типы сущностей онтологии."""
    OBJECT       = "Объект"           # РВС, обвалование, здание
    PROCESS      = "Процесс"          # горение, охлаждение, пенная атака
    RESOURCE     = "Ресурс"           # техника, вода, пена, ЛС
    PERSON       = "Должностное лицо" # РТП, НБУ, НШ, НТ
    PHASE        = "Фаза"            # S1..S5
    ACTION       = "Действие"         # 15 действий С1..О6
    NORM         = "Норматив"         # ГОСТ, СП, БУПО
    METRIC       = "Метрика"          # L7, риск, α
    MODE         = "Режим управления" # N, T, O, M, D


@dataclass
class OntologyEntity:
    """Сущность онтологии."""
    id: str
    name: str
    entity_type: EntityType
    properties: Dict[str, any] = field(default_factory=dict)
    description: str = ""
    parent_class: str = ""  # наследование (is-a)


@dataclass
class OntologyRelation:
    """Отношение между двумя сущностями."""
    subject: str        # id сущности-субъекта
    predicate: str      # тип отношения
    object: str         # id сущности-объекта
    properties: Dict[str, any] = field(default_factory=dict)


@dataclass
class OntologyRule:
    """Правило (аксиома) предметной области."""
    id: str
    name: str
    condition: str      # текстовое описание условия
    conclusion: str     # текстовое описание вывода
    source: str = ""    # источник (ГОСТ, БУПО, экспертное)
    formal: str = ""    # формальная запись (если есть)


# ═══════════════════════════════════════════════════════════════════════════
# ТИПЫ ОТНОШЕНИЙ
# ═══════════════════════════════════════════════════════════════════════════
RELATION_TYPES = {
    "управляет":        {"inverse": "подчиняется",      "domain": "PERSON",   "range": "PERSON"},
    "подчиняется":      {"inverse": "управляет",        "domain": "PERSON",   "range": "PERSON"},
    "использует":       {"inverse": "используется",     "domain": "PERSON",   "range": "RESOURCE"},
    "используется":     {"inverse": "использует",       "domain": "RESOURCE", "range": "PERSON"},
    "расположен_на":    {"inverse": "содержит",         "domain": "RESOURCE", "range": "OBJECT"},
    "содержит":         {"inverse": "расположен_на",    "domain": "OBJECT",   "range": "RESOURCE"},
    "выполняет":        {"inverse": "выполняется",      "domain": "PERSON",   "range": "ACTION"},
    "выполняется":      {"inverse": "выполняет",        "domain": "ACTION",   "range": "PERSON"},
    "регулирует":       {"inverse": "регулируется",     "domain": "NORM",     "range": "PROCESS"},
    "регулируется":     {"inverse": "регулирует",       "domain": "PROCESS",  "range": "NORM"},
    "предшествует":     {"inverse": "следует_за",       "domain": "PHASE",    "range": "PHASE"},
    "следует_за":       {"inverse": "предшествует",     "domain": "PHASE",    "range": "PHASE"},
    "характеризуется":  {"inverse": "характеризует",    "domain": "PROCESS",  "range": "METRIC"},
    "характеризует":    {"inverse": "характеризуется",  "domain": "METRIC",   "range": "PROCESS"},
    "is_a":             {"inverse": "",                  "domain": "*",        "range": "*"},
    "part_of":          {"inverse": "has_part",          "domain": "*",        "range": "*"},
    "has_part":         {"inverse": "part_of",           "domain": "*",        "range": "*"},
    "применяется_в":    {"inverse": "включает",         "domain": "ACTION",   "range": "PHASE"},
    "включает":         {"inverse": "применяется_в",    "domain": "PHASE",    "range": "ACTION"},
    "измеряет":         {"inverse": "измеряется",       "domain": "METRIC",   "range": "PROCESS"},
    "определяет":       {"inverse": "определяется",     "domain": "MODE",     "range": "ACTION"},
}


# ═══════════════════════════════════════════════════════════════════════════
# ОНТОЛОГИЯ
# ═══════════════════════════════════════════════════════════════════════════
class FireOntology:
    """Онтология предметной области тушения пожаров РВС."""

    def __init__(self):
        self.entities: Dict[str, OntologyEntity] = {}
        self.relations: List[OntologyRelation] = []
        self.rules: List[OntologyRule] = []
        self._build_default()

    def add_entity(self, entity: OntologyEntity):
        self.entities[entity.id] = entity

    def add_relation(self, subject: str, predicate: str, obj: str, **props):
        self.relations.append(OntologyRelation(subject, predicate, obj, props))

    def add_rule(self, rule: OntologyRule):
        self.rules.append(rule)

    def _build_default(self):
        """Построить онтологию по умолчанию (предметная область пожаротушения)."""
        E = EntityType
        ae = lambda id, name, etype, desc="", parent="", **props: \
            self.add_entity(OntologyEntity(id, name, etype, props, desc, parent))
        ar = self.add_relation

        # ── ОБЪЕКТЫ ─────────────────────────────────────────────────
        ae("rvs", "Резервуар вертикальный стальной", E.OBJECT,
           "Ёмкость для хранения нефти/нефтепродуктов",
           volume_m3=(1000, 50000), diameter_m=(10, 80))
        ae("rvs_burn", "Горящий РВС", E.OBJECT, parent_class="rvs")
        ae("rvs_nbr", "Соседний РВС", E.OBJECT, parent_class="rvs")
        ae("obval", "Обвалование", E.OBJECT, "Земляной вал вокруг РВС")
        ae("shtab", "Оперативный штаб", E.OBJECT)
        ae("vodoist", "Водоисточник", E.OBJECT,
           "Река, водоём, пожарный гидрант")
        ae("hydrant", "Пожарный гидрант", E.OBJECT, parent_class="vodoist")

        # ── РЕСУРСЫ (техника) ───────────────────────────────────────
        ae("ac", "Автоцистерна (АЦ)", E.RESOURCE, volume_l=6000)
        ae("apt", "Автопеноподъёмник (АПТ)", E.RESOURCE)
        ae("pns", "Насосная станция (ПНС)", E.RESOURCE, flow_ls=110)
        ae("panrk", "ПАНРК", E.RESOURCE)
        ae("akp", "Автоколенчатый подъёмник (АКП)", E.RESOURCE)
        ae("ash", "Штабной автомобиль (АШ)", E.RESOURCE)
        ae("stvol", "Ствол охлаждения", E.RESOURCE,
           types=["Антенор-1500", "ЛС-С330", "ГПС-600", "РС-70"])
        ae("pena", "Пенообразователь", E.RESOURCE, unit="тонны")
        ae("voda", "Вода", E.RESOURCE, unit="л/с")
        ae("ls", "Личный состав", E.RESOURCE, unit="чел.")

        # ── ДОЛЖНОСТНЫЕ ЛИЦА ───────────────────────────────────────
        ae("rtp", "Руководитель тушения пожара (РТП)", E.PERSON,
           level=0, alpha_range=(0.5, 0.8))
        ae("nsh", "Начальник штаба (НШ)", E.PERSON, level=1)
        ae("nt", "Начальник тыла (НТ)", E.PERSON, level=1)
        ae("nbu", "Начальник боевого участка (НБУ)", E.PERSON,
           level=2, sectors=["юг", "восток", "запад"])
        ae("ot", "Ответственный за ТБ (ОТ)", E.PERSON, level=1)
        ae("ng", "Начальник гарнизона (НГ)", E.PERSON, level=-1)

        # ── ФАЗЫ ───────────────────────────────────────────────────
        ae("s1", "S1 — Обнаружение / Выезд", E.PHASE, order=1)
        ae("s2", "S2 — Боевое развёртывание", E.PHASE, order=2)
        ae("s3", "S3 — Активное горение", E.PHASE, order=3)
        ae("s4", "S4 — Пенная атака", E.PHASE, order=4)
        ae("s5", "S5 — Ликвидация последствий", E.PHASE, order=5)

        # ── ПРОЦЕССЫ ───────────────────────────────────────────────
        ae("gorenie", "Горение", E.PROCESS)
        ae("ohlazhd", "Охлаждение", E.PROCESS)
        ae("pen_ataka", "Пенная атака", E.PROCESS)
        ae("razvedka", "Разведка пожара", E.PROCESS)
        ae("evakuacia", "Эвакуация", E.PROCESS)
        ae("vodosnab", "Водоснабжение", E.PROCESS)

        # ── ДЕЙСТВИЯ (15) ──────────────────────────────────────────
        actions = [
            ("a_s1", "С1 Спасение людей", "стратег."),
            ("a_s2", "С2 Защита соседнего РВС", "стратег."),
            ("a_s3", "С3 Локализация", "стратег."),
            ("a_s4", "С4 Ликвидация", "стратег."),
            ("a_s5", "С5 Предотвращение вскипания", "стратег."),
            ("a_t1", "Т1 Создать БУ", "тактич."),
            ("a_t2", "Т2 Перегруппировка", "тактич."),
            ("a_t3", "Т3 Вызов доп. сил", "тактич."),
            ("a_t4", "Т4 Установить ПНС", "тактич."),
            ("a_o1", "О1 Подача ствола", "оперативн."),
            ("a_o2", "О2 Охлаждение соседнего", "оперативн."),
            ("a_o3", "О3 Пенная атака", "оперативн."),
            ("a_o4", "О4 Разведка", "оперативн."),
            ("a_o5", "О5 Ликвидация розлива", "оперативн."),
            ("a_o6", "О6 Сигнал отхода", "оперативн."),
        ]
        for aid, aname, alevel in actions:
            ae(aid, aname, E.ACTION, level=alevel)

        # ── МЕТРИКИ ────────────────────────────────────────────────
        ae("m_l7", "L7 — вероятность выполнения задачи", E.METRIC,
           range=(0, 1), target=0.90)
        ae("m_risk", "Индекс риска", E.METRIC, range=(0, 1), critical=0.75)
        ae("m_alpha", "Коэффициент автономности α", E.METRIC, range=(0, 1))
        ae("m_delta_s", "Отклонение δ_s", E.METRIC, threshold=0.20)

        # ── НОРМАТИВЫ ──────────────────────────────────────────────
        ae("gost_51043", "ГОСТ Р 51043-2002", E.NORM,
           title="Установки водяного и пенного пожаротушения")
        ae("sp_155", "СП 155.13130.2014", E.NORM,
           title="Склады нефти и нефтепродуктов")
        ae("bupo", "БУПО (Приказ МЧС №444)", E.NORM,
           title="Боевой устав пожарной охраны")

        # ── РЕЖИМЫ УПРАВЛЕНИЯ ──────────────────────────────────────
        ae("mode_n", "Нормальный (N)", E.MODE)
        ae("mode_t", "Тактический (T)", E.MODE)
        ae("mode_o", "Оперативный (O)", E.MODE)
        ae("mode_m", "Мобилизация (M)", E.MODE)
        ae("mode_d", "Деградация (D)", E.MODE)

        # ═══════════════════════════════════════════════════════════
        # ОТНОШЕНИЯ
        # ═══════════════════════════════════════════════════════════
        # Иерархия управления
        ar("ng", "управляет", "rtp")
        ar("rtp", "управляет", "nsh")
        ar("rtp", "управляет", "nt")
        ar("rtp", "управляет", "nbu")
        ar("nbu", "управляет", "ls")

        # Ресурсы
        ar("rtp", "использует", "ash")
        ar("nbu", "использует", "stvol")
        ar("nbu", "использует", "ac")
        ar("nt", "использует", "pns")
        ar("nt", "использует", "voda")
        ar("pns", "расположен_на", "vodoist")

        # Фазы → действия
        ar("s1", "включает", "a_o4")
        ar("s1", "включает", "a_t3")
        ar("s2", "включает", "a_o1")
        ar("s2", "включает", "a_o2")
        ar("s2", "включает", "a_t1")
        ar("s3", "включает", "a_t3")
        ar("s3", "включает", "a_o1")
        ar("s3", "включает", "a_o3")
        ar("s4", "включает", "a_o3")
        ar("s4", "включает", "a_s4")
        ar("s5", "включает", "a_o4")

        # Фазы → последовательность
        ar("s1", "предшествует", "s2")
        ar("s2", "предшествует", "s3")
        ar("s3", "предшествует", "s4")
        ar("s4", "предшествует", "s5")

        # Нормативы → процессы
        ar("gost_51043", "регулирует", "pen_ataka")
        ar("sp_155", "регулирует", "ohlazhd")
        ar("bupo", "регулирует", "razvedka")

        # Метрики
        ar("m_l7", "характеризует", "rtp")
        ar("m_risk", "характеризует", "gorenie")
        ar("m_alpha", "характеризует", "nbu")
        ar("m_delta_s", "характеризует", "mode_n")

        # Наследование
        ar("rvs_burn", "is_a", "rvs")
        ar("rvs_nbr", "is_a", "rvs")
        ar("hydrant", "is_a", "vodoist")

        # ═══════════════════════════════════════════════════════════
        # ПРАВИЛА (аксиомы)
        # ═══════════════════════════════════════════════════════════
        self.rules = [
            OntologyRule("r1", "Решающее направление",
                         "Пожар на РВС и есть угроза людям",
                         "РТП назначает РН — спасение людей (С1)",
                         source="БУПО §3.1"),
            OntologyRule("r2", "Обязательное охлаждение",
                         "Горящий РВС и соседний РВС в зоне теплового воздействия",
                         "Организовать охлаждение обоих РВС стволами",
                         source="СП 155.13130.2014 §9.3"),
            OntologyRule("r3", "Пенная атака",
                         "foam_ready И phase=S4 И Q_пены ≥ Q_норм",
                         "Провести пенную атаку (О3)",
                         source="ГОСТ Р 51043-2002",
                         formal="foam_ready ∧ phase=S4 ∧ Q_f ≥ I·S → O3"),
            OntologyRule("r4", "Адаптация",
                         "δ_s > ε (отклонение от целевого состояния > 0.20)",
                         "Переход в адаптивный режим (T → O → M)",
                         formal="δ_s > 0.20 → mode ∈ {T, O, M}"),
            OntologyRule("r5", "Создание штаба",
                         "Ранг пожара ≥ 2 И прибытие первых подразделений",
                         "Создать оперативный штаб, назначить НШ, НТ",
                         source="БУПО §5.1"),
            OntologyRule("r6", "Автономность НБУ",
                         "α_НБУ > 0.5 в фазе S3",
                         "НБУ действует самостоятельно (λ_intr < 0.3)",
                         formal="α_L1(S3) > 0.5 → λ ← min(0.3, λ)"),
            OntologyRule("r7", "Сигнал отхода",
                         "risk > 0.85 ИЛИ угроза вскипания",
                         "Немедленный вывод ЛС (О6)",
                         source="БУПО §3.12"),
        ]

    # ── Запросы к онтологии ──────────────────────────────────────
    def get_entities_by_type(self, etype: EntityType) -> List[OntologyEntity]:
        return [e for e in self.entities.values() if e.entity_type == etype]

    def get_relations_for(self, entity_id: str) -> List[OntologyRelation]:
        return [r for r in self.relations
                if r.subject == entity_id or r.object == entity_id]

    def get_related(self, entity_id: str, predicate: str) -> List[str]:
        return [r.object for r in self.relations
                if r.subject == entity_id and r.predicate == predicate]

    def stats(self) -> Dict:
        type_counts = {}
        for e in self.entities.values():
            t = e.entity_type.value
            type_counts[t] = type_counts.get(t, 0) + 1
        pred_counts = {}
        for r in self.relations:
            pred_counts[r.predicate] = pred_counts.get(r.predicate, 0) + 1
        return {
            "entities": len(self.entities),
            "relations": len(self.relations),
            "rules": len(self.rules),
            "by_type": type_counts,
            "by_predicate": pred_counts,
        }

    # ── Экспорт ──────────────────────────────────────────────────
    def to_json(self, path: str = "") -> str:
        if not path:
            path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "data", "ontology.json")
        data = {
            "@context": "САУР-ПСП Ontology v1.0",
            "entities": {eid: {
                "name": e.name,
                "type": e.entity_type.value,
                "properties": e.properties,
                "description": e.description,
                "parent": e.parent_class,
            } for eid, e in self.entities.items()},
            "relations": [
                {"subject": r.subject, "predicate": r.predicate,
                 "object": r.object}
                for r in self.relations
            ],
            "rules": [
                {"id": r.id, "name": r.name, "condition": r.condition,
                 "conclusion": r.conclusion, "source": r.source,
                 "formal": r.formal}
                for r in self.rules
            ],
        }
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2, default=str)
        return path

    # ── Визуализация ─────────────────────────────────────────────
    def plot_ontology_graph(self, filename: str = "ontology_graph.png") -> str:
        """Граф онтологии (кластеры по типам сущностей)."""
        fig, ax = plt.subplots(figsize=(18, 14), facecolor="white")
        ax.set_xlim(-2, 18)
        ax.set_ylim(-2, 14)
        ax.axis("off")

        type_colors = {
            EntityType.OBJECT:  "#3498db",
            EntityType.PROCESS: "#e74c3c",
            EntityType.RESOURCE:"#27ae60",
            EntityType.PERSON:  "#e67e22",
            EntityType.PHASE:   "#9b59b6",
            EntityType.ACTION:  "#1abc9c",
            EntityType.NORM:    "#566573",
            EntityType.METRIC:  "#f39c12",
            EntityType.MODE:    "#c0392b",
        }

        # Расположить кластерами по типам
        type_positions = {
            EntityType.PERSON:   (3, 11),
            EntityType.OBJECT:   (9, 11),
            EntityType.RESOURCE: (15, 11),
            EntityType.PHASE:    (3, 6),
            EntityType.ACTION:   (9, 6),
            EntityType.PROCESS:  (15, 6),
            EntityType.METRIC:   (3, 1.5),
            EntityType.NORM:     (9, 1.5),
            EntityType.MODE:     (15, 1.5),
        }

        pos = {}
        rng = np.random.RandomState(42)
        for eid, entity in self.entities.items():
            base_x, base_y = type_positions.get(entity.entity_type, (8, 7))
            dx = rng.uniform(-2, 2)
            dy = rng.uniform(-1.5, 1.5)
            pos[eid] = (base_x + dx, base_y + dy)

        # Рёбра
        for rel in self.relations:
            if rel.subject in pos and rel.object in pos:
                x1, y1 = pos[rel.subject]
                x2, y2 = pos[rel.object]
                ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                            arrowprops=dict(arrowstyle="->", color="#bdc3c7",
                                           lw=0.7, alpha=0.4))

        # Вершины
        for eid, entity in self.entities.items():
            if eid not in pos:
                continue
            x, y = pos[eid]
            color = type_colors.get(entity.entity_type, "#95a5a6")
            ax.scatter(x, y, s=120, c=color, edgecolors="white",
                       linewidth=1, zorder=3, alpha=0.85)
            ax.text(x, y - 0.35, entity.name[:20], ha="center", fontsize=5.5,
                    color="#2c3e50", zorder=4)

        # Метки кластеров
        for etype, (cx, cy) in type_positions.items():
            color = type_colors.get(etype, "#95a5a6")
            ax.text(cx, cy + 2.0, etype.value, ha="center", fontsize=10,
                    fontweight="bold", color=color,
                    bbox=dict(boxstyle="round,pad=0.3", facecolor=color,
                              edgecolor="white", alpha=0.15))

        st = self.stats()
        ax.set_title(f"Онтология тушения пожара РВС  "
                     f"({st['entities']} сущностей, {st['relations']} отношений, "
                     f"{st['rules']} правил)",
                     fontsize=13, fontweight="bold", color="#2c3e50")

        fig.tight_layout()
        return _save(fig, filename)


# ═══════════════════════════════════════════════════════════════════════════
# ДЕМО
# ═══════════════════════════════════════════════════════════════════════════
# ═══════════════════════════════════════════════════════════════════════════
# ОНТОЛОГИЧЕСКИЙ АНАЛИЗ ДЛЯ СППР
# ═══════════════════════════════════════════════════════════════════════════
@dataclass
class OntologyAnalysisResult:
    """Полный результат онтологического анализа."""
    # Покрытие
    norm_coverage: float = 0.0        # доля действий с нормативным обоснованием
    uncovered_actions: List[str] = field(default_factory=list)
    # Достижимость
    reachability: Dict[str, float] = field(default_factory=dict)
    # Критичность
    criticality: Dict[str, float] = field(default_factory=dict)
    bottlenecks: List[str] = field(default_factory=list)
    # Предусловия
    preconditions: Dict[str, List[str]] = field(default_factory=dict)
    # Паттерны
    decision_entropy: Dict[str, float] = field(default_factory=dict)
    # Соответствие
    compliance_index: float = 0.0
    violations: List[Dict] = field(default_factory=list)
    # Рекомендации СППР
    recommendations: List[Dict] = field(default_factory=list)


class OntologyAnalyzer:
    """Анализ онтологии для поддержки принятия решений."""

    def __init__(self, ontology: FireOntology):
        self.onto = ontology
        self._adj: Optional[Dict[str, Set[str]]] = None
        self._build_adjacency()

    def _build_adjacency(self):
        """Построить граф смежности из отношений."""
        self._adj = {}
        for r in self.onto.relations:
            self._adj.setdefault(r.subject, set()).add(r.object)
            self._adj.setdefault(r.object, set()).add(r.subject)

    # ── 1. Покрытие нормативной базы ─────────────────────────────
    def norm_coverage(self) -> Tuple[float, List[str]]:
        """Доля действий, имеющих нормативное обоснование.

        Действие покрыто, если существует путь:
          действие → (включает/применяется_в) → фаза → ... → норматив
        или действие упоминается в правиле.
        """
        actions = [e for e in self.onto.entities.values()
                   if e.entity_type == EntityType.ACTION]
        norms = {e.id for e in self.onto.entities.values()
                 if e.entity_type == EntityType.NORM}
        rules_text = " ".join(r.conclusion + " " + r.condition
                              for r in self.onto.rules)

        covered = []
        uncovered = []
        for action in actions:
            # Проверить: упоминается ли в правилах
            action_code = action.name.split()[0]  # "С1", "О3" и т.д.
            if action_code in rules_text or action.id in rules_text:
                covered.append(action.id)
                continue

            # Проверить: есть ли путь до норматива через граф
            reachable = self._bfs_reachable(action.id, max_depth=4)
            if reachable & norms:
                covered.append(action.id)
            else:
                uncovered.append(action.name)

        rate = len(covered) / max(len(actions), 1)
        return rate, uncovered

    def _bfs_reachable(self, start: str, max_depth: int = 4) -> Set[str]:
        """BFS: все достижимые узлы из start за max_depth шагов."""
        visited = set()
        queue = [(start, 0)]
        while queue:
            node, depth = queue.pop(0)
            if node in visited or depth > max_depth:
                continue
            visited.add(node)
            for neighbor in self._adj.get(node, set()):
                queue.append((neighbor, depth + 1))
        return visited

    # ── 2. Достижимость цели ─────────────────────────────────────
    def goal_reachability(self, current_resources: Dict[str, bool]
                          ) -> Dict[str, float]:
        """Индекс достижимости каждого действия при текущих ресурсах.

        current_resources: {"pns": True, "pena": False, "voda": True, ...}
        Возвращает: {"О3 Пенная атака": 0.33, "О1 Подача ствола": 1.0, ...}
        """
        actions = [e for e in self.onto.entities.values()
                   if e.entity_type == EntityType.ACTION]

        # Предусловия каждого действия (через отношения)
        action_prereqs = {
            "a_o3": ["pena", "pns", "voda", "stvol"],  # Пенная атака
            "a_o1": ["voda", "stvol"],                   # Подача ствола
            "a_o2": ["voda", "stvol"],                   # Охлаждение
            "a_t4": ["pns", "vodoist"],                  # Установить ПНС
            "a_s4": ["pena", "pns", "voda"],             # Ликвидация
            "a_o5": ["voda", "stvol"],                   # Розлив
            "a_t3": [],                                   # Вызов сил — всегда доступно
            "a_o4": [],                                   # Разведка — всегда
            "a_o6": [],                                   # Отход — всегда
        }

        result = {}
        for action in actions:
            prereqs = action_prereqs.get(action.id, [])
            if not prereqs:
                result[action.name] = 1.0
            else:
                satisfied = sum(1 for p in prereqs
                                if current_resources.get(p, False))
                result[action.name] = satisfied / len(prereqs)
        return result

    # ── 3. Критичность ресурсов ──────────────────────────────────
    def resource_criticality(self) -> Dict[str, float]:
        """Betweenness centrality каждого ресурса в онтографе.

        Чем выше — тем больше путей проходит через ресурс.
        Удаление критического ресурса разрывает больше связей.
        """
        resources = [e for e in self.onto.entities.values()
                     if e.entity_type == EntityType.RESOURCE]
        all_ids = list(self.onto.entities.keys())
        n = len(all_ids)
        id_to_idx = {eid: i for i, eid in enumerate(all_ids)}

        # Матрица смежности
        adj = np.zeros((n, n))
        for r in self.onto.relations:
            if r.subject in id_to_idx and r.object in id_to_idx:
                i, j = id_to_idx[r.subject], id_to_idx[r.object]
                adj[i, j] = 1
                adj[j, i] = 1

        # Floyd-Warshall
        dist = np.where(adj > 0, 1, np.inf)
        np.fill_diagonal(dist, 0)
        for k in range(n):
            for i in range(n):
                for j in range(n):
                    if dist[i, k] + dist[k, j] < dist[i, j]:
                        dist[i, j] = dist[i, k] + dist[k, j]

        # Betweenness
        between = np.zeros(n)
        for s in range(n):
            for t in range(s + 1, n):
                if dist[s, t] == np.inf:
                    continue
                for v in range(n):
                    if v != s and v != t:
                        if abs(dist[s, v] + dist[v, t] - dist[s, t]) < 0.5:
                            between[v] += 1

        norm = max(1, (n - 1) * (n - 2) / 2)
        result = {}
        for res in resources:
            idx = id_to_idx.get(res.id, -1)
            if idx >= 0:
                result[res.name] = round(float(between[idx] / norm), 4)

        return dict(sorted(result.items(), key=lambda x: -x[1]))

    # ── 4. Цепочки предусловий ───────────────────────────────────
    def precondition_chains(self) -> Dict[str, List[str]]:
        """Цепочки зависимостей для каждого действия.

        «Чтобы выполнить О3 (пенная атака), нужно:
         вода → ПНС → пена → ствол → атака»
        """
        chains = {
            "О3 Пенная атака": [
                "Водоисточник доступен",
                "ПНС установлена на водоисточник",
                "Запас пенообразователя достаточен",
                "Стволы поданы на охлаждение (≥3)",
                "Готовность к пенной атаке подтверждена",
            ],
            "О1 Подача ствола": [
                "Водоисточник доступен",
                "АЦ/ПНС на водоисточнике",
                "Магистральная линия проложена",
            ],
            "Т1 Создать БУ": [
                "РТП прибыл и оценил обстановку",
                "Определены секторы (юг/восток/запад)",
                "Назначены НБУ",
            ],
            "С4 Ликвидация": [
                "Все предусловия О3 выполнены",
                "Все БУ готовы",
                "НТ обеспечил бесперебойное водоснабжение",
                "РТП принял решение о пенной атаке",
            ],
        }
        return chains

    # ── 5. Энтропия решений по фазам ─────────────────────────────
    def decision_entropy_by_phase(self,
                                   scenario_decisions: List[Dict] = None
                                   ) -> Dict[str, float]:
        """Энтропия Шеннона распределения действий по фазам.

        Высокая энтропия = высокая неопределённость выбора.
        Низкая = действие предопределено.
        """
        phases = ["S1", "S2", "S3", "S4", "S5"]

        if scenario_decisions:
            # Из реальных данных
            phase_actions = {p: {} for p in phases}
            for d in scenario_decisions:
                p = d.get("phase", "S3")
                a = d.get("action_code", "О4")
                phase_actions.setdefault(p, {})
                phase_actions[p][a] = phase_actions[p].get(a, 0) + 1
        else:
            # Из онтологии (какие действия допустимы в каждой фазе)
            phase_actions = {}
            for p_id in ["s1", "s2", "s3", "s4", "s5"]:
                p_name = p_id.upper()
                related = self.onto.get_related(p_id, "включает")
                phase_actions[p_name] = {
                    self.onto.entities[r].name: 1
                    for r in related
                    if r in self.onto.entities
                }

        result = {}
        for phase, actions in phase_actions.items():
            if not actions:
                result[phase] = 0.0
                continue
            total = sum(actions.values())
            probs = np.array([v / total for v in actions.values()])
            probs = probs[probs > 0]
            result[phase] = round(float(-np.sum(probs * np.log2(probs))), 3)
        return result

    # ── 6. Проверка соответствия правилам ────────────────────────
    def check_compliance(self, state: Dict) -> Tuple[float, List[Dict]]:
        """Проверить: какие правила онтологии нарушены при текущем состоянии.

        state: {"phase": "S3", "has_shtab": False, "n_trunks": 2,
                "foam_ready": False, "risk": 0.6, ...}
        """
        violations = []
        total_rules = len(self.onto.rules)
        passed = 0

        for rule in self.onto.rules:
            violated = False
            reason = ""

            if rule.id == "r2" and state.get("phase") in ("S2", "S3", "S4"):
                if state.get("n_trunks_nbr", 0) == 0:
                    violated = True
                    reason = "Соседний РВС не охлаждается"

            elif rule.id == "r5" and state.get("fire_rank", 0) >= 2:
                if not state.get("has_shtab", False):
                    violated = True
                    reason = "Штаб не создан при ранге ≥2"

            elif rule.id == "r7" and state.get("risk", 0) > 0.85:
                if state.get("last_action") != 14:  # О6
                    violated = True
                    reason = "Риск > 0.85, но сигнал отхода не дан"

            if violated:
                violations.append({
                    "rule_id": rule.id,
                    "rule_name": rule.name,
                    "source": rule.source,
                    "reason": reason,
                })
            else:
                passed += 1

        compliance = passed / max(total_rules, 1)
        return compliance, violations

    # ── 7. Генерация рекомендаций СППР ───────────────────────────
    def generate_recommendations(self, state: Dict) -> List[Dict]:
        """Сгенерировать рекомендации на основе онтологии.

        state: текущее состояние симуляции
        """
        recs = []
        phase = state.get("phase", "S1")
        risk = state.get("risk", 0)
        foam_ready = state.get("foam_ready", False)
        has_shtab = state.get("has_shtab", False)
        n_trunks = state.get("n_trunks_burn", 0)
        n_pns = state.get("n_pns", 0)
        rank = state.get("fire_rank", 2)

        # Проверка предусловий
        if phase in ("S3", "S4") and not foam_ready:
            missing = []
            if n_pns == 0:
                missing.append("ПНС не установлена (Т4)")
            if n_trunks < 3:
                missing.append(f"Стволов {n_trunks} < 3 (О1)")
            if missing:
                recs.append({
                    "priority": "высокий",
                    "action": "Подготовка к пенной атаке",
                    "reason": f"Пенная атака невозможна: {'; '.join(missing)}",
                    "rule": "r3 (ГОСТ Р 51043-2002)",
                })

        if rank >= 2 and not has_shtab:
            recs.append({
                "priority": "высокий",
                "action": "Создать оперативный штаб (Т1)",
                "reason": f"Ранг пожара №{rank} ≥ 2, штаб не создан",
                "rule": "r5 (БУПО §5.1)",
            })

        if risk > 0.75:
            recs.append({
                "priority": "критический",
                "action": "Рассмотреть сигнал отхода (О6)",
                "reason": f"Индекс риска {risk:.2f} > 0.75",
                "rule": "r7 (БУПО §3.12)",
            })

        if phase == "S2" and n_trunks < 3:
            recs.append({
                "priority": "средний",
                "action": f"Подать стволы на охлаждение (О1): {n_trunks}/3",
                "reason": "Недостаточное охлаждение горящего РВС",
                "rule": "r2 (СП 155 §9.3)",
            })

        if phase == "S1":
            recs.append({
                "priority": "средний",
                "action": "Провести разведку (О4)",
                "reason": "Фаза S1: необходимо оценить обстановку",
                "rule": "r1 (БУПО §3.1)",
            })

        # Онтологический вывод: достижимость
        resources = {
            "pns": n_pns > 0,
            "voda": n_pns > 0 or n_trunks > 0,
            "pena": state.get("foam_conc", 0) > 0,
            "stvol": n_trunks > 0,
            "vodoist": True,
        }
        reachability = self.goal_reachability(resources)
        low_reach = [(name, val) for name, val in reachability.items()
                     if val < 0.5 and val > 0]
        if low_reach:
            for name, val in low_reach[:2]:
                recs.append({
                    "priority": "информационный",
                    "action": f"{name}: достижимость {val:.0%}",
                    "reason": "Не все предусловия выполнены",
                    "rule": "Онтологический вывод",
                })

        return sorted(recs, key=lambda r: {"критический": 0, "высокий": 1,
                                            "средний": 2, "информационный": 3
                                            }.get(r["priority"], 9))

    # ── 8. Полный анализ ─────────────────────────────────────────
    def full_analysis(self, state: Optional[Dict] = None
                      ) -> OntologyAnalysisResult:
        """Провести полный онтологический анализ."""
        result = OntologyAnalysisResult()

        # 1. Покрытие
        result.norm_coverage, result.uncovered_actions = self.norm_coverage()

        # 2. Достижимость
        if state:
            resources = {
                "pns": state.get("n_pns", 0) > 0,
                "voda": state.get("n_pns", 0) > 0,
                "pena": state.get("foam_conc", 0) > 0,
                "stvol": state.get("n_trunks_burn", 0) > 0,
                "vodoist": True,
            }
            result.reachability = self.goal_reachability(resources)

        # 3. Критичность
        result.criticality = self.resource_criticality()
        result.bottlenecks = [name for name, val in result.criticality.items()
                              if val > 0.01][:3]

        # 4. Предусловия
        result.preconditions = self.precondition_chains()

        # 5. Энтропия
        result.decision_entropy = self.decision_entropy_by_phase()

        # 6. Соответствие
        if state:
            result.compliance_index, result.violations = \
                self.check_compliance(state)

        # 7. Рекомендации
        if state:
            result.recommendations = self.generate_recommendations(state)

        return result


def plot_ontology_analysis(result: OntologyAnalysisResult,
                           filename: str = "ontology_analysis.png") -> str:
    """Визуализация результатов онтологического анализа."""
    fig, axes = plt.subplots(2, 3, figsize=(16, 9), facecolor="white")

    # 1. Покрытие нормативной базы
    ax = axes[0, 0]
    covered = result.norm_coverage
    ax.barh(["Покрыто", "Не покрыто"],
            [covered, 1 - covered],
            color=["#27ae60", "#c0392b"], edgecolor="white")
    ax.set_xlim(0, 1)
    ax.set_title(f"Нормативное покрытие: {covered:.0%}", fontweight="bold")
    ax.set_xlabel("Доля действий")
    ax.grid(True, axis="x", alpha=0.3)

    # 2. Критичность ресурсов
    ax = axes[0, 1]
    if result.criticality:
        names = list(result.criticality.keys())[:8]
        vals = [result.criticality[n] for n in names]
        short = [n.split("(")[0].strip()[:15] for n in names]
        ax.barh(range(len(short)), vals, color="#e67e22", edgecolor="white")
        ax.set_yticks(range(len(short)))
        ax.set_yticklabels(short, fontsize=8)
        ax.set_xlabel("Центральность (betweenness)")
    ax.set_title("Критичность ресурсов", fontweight="bold")
    ax.grid(True, axis="x", alpha=0.3)

    # 3. Энтропия решений по фазам
    ax = axes[0, 2]
    if result.decision_entropy:
        phases = list(result.decision_entropy.keys())
        entropy = list(result.decision_entropy.values())
        colors = ["#3498db" if e < 1.5 else "#e67e22" if e < 2.5
                  else "#c0392b" for e in entropy]
        ax.bar(phases, entropy, color=colors, edgecolor="white")
        ax.set_ylabel("Энтропия (бит)")
    ax.set_title("Неопределённость решений по фазам", fontweight="bold")
    ax.grid(True, axis="y", alpha=0.3)

    # 4. Достижимость действий
    ax = axes[1, 0]
    if result.reachability:
        names = list(result.reachability.keys())
        vals = list(result.reachability.values())
        short = [n.split()[0][:6] for n in names]
        colors = ["#27ae60" if v >= 0.8 else "#e67e22" if v >= 0.5
                  else "#c0392b" for v in vals]
        ax.barh(range(len(short)), vals, color=colors, edgecolor="white")
        ax.set_yticks(range(len(short)))
        ax.set_yticklabels(short, fontsize=7)
        ax.set_xlim(0, 1.1)
        ax.set_xlabel("Достижимость (0–1)")
    ax.set_title("Достижимость действий", fontweight="bold")
    ax.grid(True, axis="x", alpha=0.3)

    # 5. Соответствие правилам
    ax = axes[1, 1]
    if result.compliance_index > 0 or result.violations:
        n_ok = int(result.compliance_index * 7)
        n_viol = len(result.violations)
        ax.pie([n_ok, n_viol], labels=[f"Соблюдены ({n_ok})",
               f"Нарушены ({n_viol})"],
               colors=["#27ae60", "#c0392b"], autopct="%1.0f%%",
               textprops={"fontsize": 9})
    ax.set_title(f"Соответствие: {result.compliance_index:.0%}", fontweight="bold")

    # 6. Рекомендации
    ax = axes[1, 2]
    ax.axis("off")
    if result.recommendations:
        lines = ["Рекомендации СППР:", ""]
        prio_colors = {"критический": "🔴", "высокий": "🟠",
                       "средний": "🟡", "информационный": "🔵"}
        for rec in result.recommendations[:5]:
            icon = prio_colors.get(rec["priority"], "⚪")
            lines.append(f"{icon} [{rec['priority']}]")
            lines.append(f"   {rec['action']}")
            lines.append(f"   {rec['reason'][:50]}")
            lines.append("")
        ax.text(0.05, 0.95, "\n".join(lines), transform=ax.transAxes,
                fontsize=8, va="top", fontfamily="monospace",
                bbox=dict(boxstyle="round", facecolor="#f5f6fa"))
    ax.set_title("Рекомендации", fontweight="bold")

    fig.suptitle("Онтологический анализ для СППР", fontsize=13,
                 fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    return _save(fig, filename)


if __name__ == "__main__":
    import sys, io
    if sys.platform == "win32":
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8',
                                      errors='replace')

    onto = FireOntology()
    st = onto.stats()
    print(f"Онтология: {st['entities']} сущностей, {st['relations']} отношений, "
          f"{st['rules']} правил")
    print(f"\nПо типам:")
    for t, n in st["by_type"].items():
        print(f"  {t}: {n}")
    print(f"\nПо отношениям:")
    for p, n in st["by_predicate"].items():
        print(f"  {p}: {n}")

    print(f"\nПравила ({len(onto.rules)}):")
    for rule in onto.rules:
        print(f"  {rule.id}: {rule.name}")
        print(f"    ЕСЛИ: {rule.condition}")
        print(f"    ТО:   {rule.conclusion}")
        if rule.source:
            print(f"    Источник: {rule.source}")

    json_path = onto.to_json()
    print(f"\nJSON: {json_path}")

    graph_path = onto.plot_ontology_graph()
    print(f"Граф: {graph_path}")

    # Онтологический анализ
    analyzer = OntologyAnalyzer(onto)

    # Состояние: фаза S3, мало ресурсов
    state = {
        "phase": "S3", "fire_rank": 4, "risk": 0.65,
        "has_shtab": True, "n_trunks_burn": 2, "n_trunks_nbr": 0,
        "n_pns": 1, "foam_conc": 5.0, "foam_ready": False,
        "last_action": 9,
    }

    result = analyzer.full_analysis(state)
    print(f"\nОнтологический анализ:")
    print(f"  Покрытие нормативной базы: {result.norm_coverage:.0%}")
    if result.uncovered_actions:
        print(f"  Непокрытые: {result.uncovered_actions[:5]}")
    print(f"  Индекс соответствия: {result.compliance_index:.0%}")
    if result.violations:
        print(f"  Нарушения:")
        for v in result.violations:
            print(f"    {v['rule_id']}: {v['reason']} ({v['source']})")
    print(f"  Энтропия решений: {result.decision_entropy}")
    print(f"  Узкие места: {result.bottlenecks[:3]}")

    print(f"\n  Рекомендации СППР:")
    for rec in result.recommendations:
        print(f"    [{rec['priority']}] {rec['action']}")
        print(f"       Причина: {rec['reason']}")
        print(f"       Правило: {rec['rule']}")

    # Достижимость
    print(f"\n  Достижимость действий:")
    for name, val in sorted(result.reachability.items(), key=lambda x: -x[1])[:8]:
        bar = "█" * int(val * 20)
        print(f"    {name[:25]:<25s} {val:.0%} {bar}")

    # Визуализация
    p_analysis = plot_ontology_analysis(result)
    print(f"\n  Визуализация: {p_analysis}")
