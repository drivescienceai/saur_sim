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
