"""Пространство состояний САУР: фазы пожара, ресурсы, результаты адаптации.

Терминология соответствует Боевому уставу пожарной охраны (БУПО).
"""
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Dict, Any, Tuple


class FirePhase(Enum):
    """Фазы пожара (стадии развития оперативной обстановки)."""
    NORMAL   = 0   # Дежурство — штатный режим
    S1       = 1   # Обнаружение/вызов — выезд и следование
    S2       = 2   # Первичная атака — боевое развёртывание
    S3       = 3   # Активное горение — локализация
    S4       = 4   # Подавление — ликвидация горения
    S5       = 5   # Докрут/проверка — ликвидация последствий
    RESOLVED = 6   # Пожар ликвидирован


class ReshayusheeNapravlenie(Enum):
    """Решающее направление (РН) — условие выбора главного направления действий РТП.

    Определяется по БУПО в зависимости от оперативной обстановки.
    """
    SPASENIE_LYUDEY        = 1  # Реальная угроза жизни людей → приоритет спасения
    PREDOTVRASHHENIE_VZRYVA = 2  # Угроза взрыва/вскипания/выброса → предотвращение ЧС
    LOKALIZATSIYA          = 3  # Ограничение площади горения → локализация
    ZASHCHITA_SOSEDNIKH    = 4  # Угроза смежным объектам → защита
    LIKVIDATSIYA           = 5  # Условия для окончательной ликвидации → ликвидация


@dataclass
class SituationState:
    """Оперативная обстановка (ОТО) — s = <O, E, H, W, tau>.

    Поддерживается РТП на основе данных разведки и ОТО.
    """
    # O — характеристики объекта
    object_type: str = "oil_base"

    # E — параметры ЧС (пожара)
    fire_area_m2: float = 0.0           # Площадь пожара (м²)
    fire_spread_rate: float = 0.0       # Скорость распространения (м²/мин)
    fire_x_m: float = 0.0              # Координата очага X
    fire_y_m: float = 0.0              # Координата очага Y

    # H — угроза людям
    casualties: int = 0                 # Число пострадавших (п.2.1 БУПО)

    # W — условия окружающей среды
    wind_speed_ms: float = 3.0          # Скорость ветра (м/с)
    visibility_m: float = 1000.0        # Видимость (м)

    # tau — время
    timestamp: float = 0.0

    # Оперативные реквизиты (БУПО)
    phase: FirePhase = FirePhase.NORMAL
    fire_rank: int = 0                  # Номер пожара (ранг вызова, 1-5+)
    reshayushee_napravlenie: ReshayusheeNapravlenie = ReshayusheeNapravlenie.LOKALIZATSIYA

    # ОФП — опасные факторы пожара
    ofp_flags: Dict[str, bool] = field(default_factory=lambda: {
        "vysokaya_temperatura": False,   # Высокая температура
        "dym": False,                    # Дым
        "toksichnye_produkty": False,    # Токсичные продукты горения
        "ugroza_vzryva": False,          # Угроза взрыва
        "vskipanie": False,              # Вскипание нефтепродуктов
    })


@dataclass
class ResourceUnit:
    """Единица сил и средств (пожарно-спасательного гарнизона — ПСГ).

    r = <rho, epsilon, lambda, delta>
    """
    unit_id: str
    unit_type: str       # "АЦ" | "АЛ" | "АТ" | "АСА" | "АШ"
    crew_size: int
    readiness: float = 1.0          # rho — готовность [0,1]
    resource_level: float = 1.0     # epsilon — уровень ОВ (вода/пена) [0,1]
    x_m: float = 0.0                # lambda — координата X
    y_m: float = 0.0                # lambda — координата Y
    task: str = "дежурство"         # delta — текущая задача
    station_id: str = ""            # номер пожарной части (ПЧ)
    response_time_min: float = 5.0  # время следования до объекта (мин)

    @property
    def is_available(self) -> bool:
        """Единица свободна и готова к применению."""
        return self.readiness > 0.3 and self.task == "дежурство"


@dataclass
class ResourceSpace:
    """Силы и средства (С и С) пожарно-спасательного гарнизона.

    R = Rv ∪ Rp ∪ Re ∪ Ri  (БУПО: пожарная техника, личный состав,
    оборудование, каналы связи)
    """
    vehicles: List[ResourceUnit] = field(default_factory=list)   # Rv — техника
    personnel: int = 0                                            # Rp — ЛС (чел.)
    equipment: Dict[str, int] = field(default_factory=dict)       # Re — оборудование
    info_channels: List[str] = field(default_factory=list)        # Ri — связь

    @property
    def available_units(self) -> List[ResourceUnit]:
        """Единицы, готовые к немедленному применению."""
        return [u for u in self.vehicles if u.is_available]

    @property
    def active_units(self) -> List[ResourceUnit]:
        """Единицы, задействованные на пожаре."""
        return [u for u in self.vehicles if u.task != "дежурство"]


class AdaptationMode(Enum):
    """Режим адаптации управления — Delta_pi.

    Соответствует уровням боевой деятельности по БУПО.
    """
    NORMAL       = "нормальный"       # delta_s <= epsilon — штатный режим
    TACTICAL     = "тактический"      # delta_s > epsilon на уровне L2 — НУТ/НС
    OPERATIONAL  = "оперативный"      # delta_s > epsilon на уровне L3 — РТП
    MOBILIZATION = "мобилизация"      # привлечение доп. сил (повышение ранга)
    DEGRADED     = "деградированный"  # нарушение связи / потеря управления


@dataclass
class AdaptationResult:
    """Корректирующий план delta_pi — инкремент планового состояния.

    Формируется РТП по результатам оценки оперативной обстановки (ОТО).
    """
    mode: AdaptationMode
    actions: List[str] = field(default_factory=list)   # Боевые задачи (БЗ)
    resources_requested: int = 0                        # Запрос доп. С и С
    priority_level: str = "НОРМАЛЬНЫЙ"                 # Приоритет команды
    delta_s: float = 0.0                               # Отклонение от целевого состояния
