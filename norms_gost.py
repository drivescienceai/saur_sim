"""
norms_gost.py
═══════════════════════════════════════════════════════════════════════════════
Нормативные данные для моделирования тушения пожаров на объектах с ЛВЖ.

Источники:
  [1] ГОСТ Р 51043-2002 (с Изменением №1, 2020) — Оросители. Технические
      требования и методы испытаний.
  [2] СП 155.13130.2014 — Склады нефти и нефтепродуктов. Требования ПБ.
  [3] СП 5.13130.2009 — АУПТ. Нормы проектирования (таб. А.3, А.4).
  [4] Методика определения расчётных величин (ВНИИПО, 2009).
═══════════════════════════════════════════════════════════════════════════════
"""
from __future__ import annotations
import math
from dataclasses import dataclass, field
from typing import Dict, Optional


# ══════════════════════════════════════════════════════════════════════════════
# 1. ИНТЕНСИВНОСТЬ ОРОШЕНИЯ — ГОСТ Р 51043-2002, п. 5.1.1.3 (Изменение №1)
# ══════════════════════════════════════════════════════════════════════════════

def irrigation_intensity_water(d_mm: float, P_MPa: float = 0.1) -> float:
    """Интенсивность орошения водяного оросителя, дм³/(м²·с).

    Формулы 5.1–5.2 (ГОСТ Р 51043-2002, Изменение №1):
      при P=0.1 МПа:  I = 0.00024 × d^2.3
      при P=0.3 МПа:  I = 0.00041 × d^2.3
    Для промежуточных давлений — линейная интерполяция по √P.

    Args:
        d_mm:  диаметр выходного отверстия, мм (8–20 мм)
        P_MPa: давление перед оросителем, МПа

    Returns:
        Интенсивность орошения I, дм³/(м²·с)
    """
    # Коэффициент масштабируется пропорционально √P относительно P=0.1 МПа
    k_base = 0.00024 * (P_MPa / 0.1) ** 0.5
    return k_base * (d_mm ** 2.3)


def irrigation_intensity_foam(d_mm: float, P_MPa: float = 0.15) -> float:
    """Интенсивность орошения пенного оросителя, дм³/(м²·с).

    Формулы 5.4–5.5 (ГОСТ Р 51043-2002, Изменение №1):
      при P=0.15 МПа: I = 0.00044 × d^2.18
      при P=0.30 МПа: I = 0.00061 × d^2.18

    Args:
        d_mm:  диаметр выходного отверстия, мм (8–15 мм)
        P_MPa: давление перед оросителем, МПа
    """
    k_base = 0.00044 * (P_MPa / 0.15) ** 0.5
    return k_base * (d_mm ** 2.18)


# ══════════════════════════════════════════════════════════════════════════════
# 2. КОЭФФИЦИЕНТ ПРОИЗВОДИТЕЛЬНОСТИ — ГОСТ Р 51043-2002, п. 8.22
# ══════════════════════════════════════════════════════════════════════════════

def k_factor(Q_dm3_s: float, P_MPa: float) -> float:
    """Коэффициент производительности оросителя K, дм³/(с·МПа^0.5).

    Формула (1):  K = Q / (10 × √P)

    Args:
        Q_dm3_s: расход ОВ, дм³/с
        P_MPa:   давление перед оросителем, МПа
    """
    return Q_dm3_s / (10.0 * math.sqrt(P_MPa))


def flow_from_k(K: float, P_MPa: float) -> float:
    """Расход ОВ через ороситель по K-фактору.

    Q = K × 10 × √P  [дм³/с = л/с]
    """
    return K * 10.0 * math.sqrt(P_MPa)


# ══════════════════════════════════════════════════════════════════════════════
# 3. ТЕХНИЧЕСКИЕ ПАРАМЕТРЫ СТВОЛОВ И ГЕНЕРАТОРОВ ПЕНЫ
#    (данные производителей и нормативные документы на оборудование)
# ══════════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class NozzleSpec:
    """Технические характеристики пожарного ствола / генератора пены."""
    name: str
    flow_ls: float          # номинальный расход, л/с
    P_MPa: float            # рабочее давление, МПа
    foam_capable: bool      # может подавать пену
    foam_expansion: float   # кратность пены (1.0 = вода, ≥5 = пена по ГОСТ)
    reach_m: float          # дальность подачи ОВ, м
    d_mm: float = 0.0       # диаметр выходного отверстия, мм (0 = не применимо)

    @property
    def K(self) -> float:
        """K-фактор по ГОСТ Р 51043-2002."""
        return k_factor(self.flow_ls, self.P_MPa)

    @property
    def foam_solution_ls(self) -> float:
        """Расход раствора пенообразователя (до вспенивания), л/с."""
        return self.flow_ls / max(self.foam_expansion, 1.0)


# ══════════════════════════════════════════════════════════════════════════════
# 3а. НОРМЫ ИНТЕНСИВНОСТИ ПОДАЧИ ОВ — СПРАВОЧНИК РТП, стр. 104–106
#     Применяются в расчётах ПТП (сценарий Б, нефтебаза РВС-2000)
# ══════════════════════════════════════════════════════════════════════════════

PTP_INTENSITY_NORMS: Dict[str, float] = {
    "насосная_пена":      0.10,  # л/(с·м²) — пенный раствор, тушение открытой насосной
    "насосная_конструк":  0.30,  # л/(с·м²) — вода на защиту несущих конструкций
    "насосная_кровля":    0.10,  # л/(с·м²) — вода на защиту кровли
    "рвс_пена":           0.05,  # л/(с·м²) — пенный раствор, тушение РВС (зеркало)
    "рвс_охл_гор":        0.20,  # л/(с·м²) — вода, охлаждение горящего РВС
    "рвс_охл_сос":        0.20,  # л/(с·м²) — вода, охлаждение соседних РВС в обваловании
}

PTP_FOAM_TIME_MIN  = 15.0   # нормативное время подачи пены, мин (Справочник РТП)
PTP_RESERVE_COEFF  = 5      # К_з — коэффициент запаса огнетушащего средства
PTP_GPS600_FLOW_LS = 5.64   # л/с — фактический расход ГПС-600 (по расчёту ПТП: q_гпс=6 л/с, η=0.94)
PTP_RC70_FLOW_LS   = 7.0    # л/с — ствол А (RC-70)
PTP_RC50_FLOW_LS   = 3.5    # л/с — ствол Б (RC-50)


# Оборудование, применявшееся на пожаре РВС №9 (сценарий А, крупный РВС)
NOZZLE_DB: Dict[str, NozzleSpec] = {
    "Антенор-1500": NozzleSpec(
        name="Антенор-1500",
        flow_ls=25.0,         # 1500 л/мин ≈ 25 л/с
        P_MPa=0.6,
        foam_capable=True,
        foam_expansion=1.0,   # подаётся как водяной ствол для охлаждения
        reach_m=55.0,
        d_mm=38.0,
    ),
    "Муссон-125": NozzleSpec(
        name="Муссон-125",
        flow_ls=125.0,        # специализированный пенный монитор
        P_MPa=0.5,
        foam_capable=True,
        foam_expansion=8.0,   # кратность пены ≥5 (ГОСТ 51043, п.5.1.1.3)
        reach_m=40.0,
        d_mm=70.0,
    ),
    "Акрон-Аполло": NozzleSpec(
        name="Акрон-Аполло",
        flow_ls=33.3,         # ≈2000 л/мин, подаётся с ППП
        P_MPa=0.7,
        foam_capable=True,
        foam_expansion=8.0,
        reach_m=30.0,
        d_mm=50.0,
    ),
    "ЛС-С330": NozzleSpec(
        name="ЛС-С330",
        flow_ls=330.0,        # лафетный ствол высокой производительности
        P_MPa=0.6,
        foam_capable=False,   # подаёт воду (охлаждение)
        foam_expansion=1.0,
        reach_m=70.0,
        d_mm=100.0,
    ),
    "ГПС-1000": NozzleSpec(
        name="ГПС-1000",
        flow_ls=16.7,         # 1000 л/мин ≈ 16.7 л/с
        P_MPa=0.4,
        foam_capable=True,
        foam_expansion=80.0,  # ГПС — высокократная пена (кратность 80-120)
        reach_m=10.0,         # подаётся через люк с АКП-50
        d_mm=30.0,
    ),
    "ГПС-600": NozzleSpec(
        name="ГПС-600",
        flow_ls=10.0,         # 600 л/мин ≈ 10 л/с
        P_MPa=0.4,
        foam_capable=True,
        foam_expansion=80.0,
        reach_m=8.0,
        d_mm=25.0,
    ),
    "Дельта-500": NozzleSpec(
        name="Дельта-500",
        flow_ls=8.3,          # 500 л/мин
        P_MPa=0.4,
        foam_capable=False,
        foam_expansion=1.0,
        reach_m=35.0,
        d_mm=25.0,
    ),
    # Стволы ПТП сценарий Б (типовые)
    "РС-70 (Ствол А)": NozzleSpec(
        name="РС-70 (Ствол А)",
        flow_ls=PTP_RC70_FLOW_LS,  # 7 л/с
        P_MPa=0.4,
        foam_capable=False,
        foam_expansion=1.0,
        reach_m=28.0,
        d_mm=19.0,
    ),
    "РС-50 (Ствол Б)": NozzleSpec(
        name="РС-50 (Ствол Б)",
        flow_ls=PTP_RC50_FLOW_LS,  # 3.5 л/с
        P_MPa=0.4,
        foam_capable=False,
        foam_expansion=1.0,
        reach_m=20.0,
        d_mm=13.0,
    ),
}


# ══════════════════════════════════════════════════════════════════════════════
# 4. НОРМЫ ОХЛАЖДЕНИЯ РЕЗЕРВУАРОВ — СП 155.13130.2014, табл. 9
# ══════════════════════════════════════════════════════════════════════════════

# Удельная интенсивность охлаждения стенок РВС, л/(с·м) периметра
COOLING_INTENSITY_BURNING_LS_M  = 0.80   # горящий РВС
COOLING_INTENSITY_ADJACENT_LS_M = 0.30   # смежный РВС (в зоне теплового воздействия)

def required_cooling_flow(diameter_m: float, burning: bool = True) -> float:
    """Требуемый расход воды для охлаждения РВС, л/с.

    Формула: Q = q × π × D
    где q — удельная интенсивность [л/(с·м)], D — диаметр резервуара.

    Args:
        diameter_m: диаметр РВС, м
        burning:    True → горящий, False → смежный

    Returns:
        Требуемый расход охлаждения, л/с
    """
    q = COOLING_INTENSITY_BURNING_LS_M if burning else COOLING_INTENSITY_ADJACENT_LS_M
    return q * math.pi * diameter_m


# ══════════════════════════════════════════════════════════════════════════════
# 5. НОРМЫ ПЕННОГО ТУШЕНИЯ — СП 155.13130.2014, табл. 8 + ГОСТ Р 51043
# ══════════════════════════════════════════════════════════════════════════════

# Нормативная интенсивность подачи пенного раствора, л/(м²·с)
FOAM_INTENSITY_NORMS: Dict[str, float] = {
    "бензин":          0.065,  # прямогонный бензин (применимо к РВС №9, сценарий А)
    "нефть":           0.060,
    "дизель":          0.048,
    "мазут":           0.060,
    "пенообразователь_гост": 0.160,  # ГОСТ Р 51043-2002, табл.1 (пенные оросители)
}

# Нормативное время подачи пены, мин
FOAM_APPLICATION_TIME_MIN: Dict[str, float] = {
    "бензин": 15.0,
    "нефть":  15.0,
    "дизель": 15.0,
}

# Кратность пены (ГОСТ Р 51043-2002, п. 5.1.1.3, п. 8.40.6)
FOAM_EXPANSION_MIN = 5    # минимальная кратность пены по ГОСТ Р 51043


def required_foam_flow(fire_area_m2: float, fuel: str = "бензин") -> float:
    """Требуемый расход пенного раствора для тушения, л/с.

    Q_пена = I_норм × S_пожара

    Args:
        fire_area_m2: площадь зеркала горения, м²
        fuel:         тип горючего (ключ из FOAM_INTENSITY_NORMS)

    Returns:
        Требуемый расход раствора пенообразователя, л/с
    """
    I = FOAM_INTENSITY_NORMS.get(fuel, FOAM_INTENSITY_NORMS["бензин"])
    return I * fire_area_m2


def foam_attack_feasibility(
    fire_area_m2: float,
    available_foam_ls: float,
    roof_obstruction_frac: float = 0.0,
    fuel: str = "бензин",
) -> dict:
    """Оценить возможность успешной пенной атаки.

    Физическая модель (ГОСТ Р 51043-2002 + СП 155.13130.2014):
      Каркас плавающей крыши РВС блокирует часть пены — эффективный расход,
      достигающий зеркала горения: Q_эфф = Q × (1 − obstruction).
      Для тушения необходимо: Q_эфф ≥ I_норм × S_зеркала (ГОСТ, п. 5.1.1.3).

      Это объясняет реальные события крупного пожара РВС:
        Атаки №1–5: Q_эфф < I_норм × S   → неудача (каркас блокирует >65%)
        Атака №6:   ГПС-1000 с АКП-50 через люк → obstruction=0.20 → Q_эфф >> норм.

    Args:
        fire_area_m2:          площадь зеркала горения, м²
        available_foam_ls:     суммарный расход пенного раствора, л/с
        roof_obstruction_frac: доля расхода, блокируемая каркасом крыши [0..1]
        fuel:                  тип горючего (ключ FOAM_INTENSITY_NORMS)

    Returns:
        dict: feasible (bool), margin (float), q_required (float), q_effective (float),
              reason (str)
    """
    I_norm = FOAM_INTENSITY_NORMS.get(fuel, FOAM_INTENSITY_NORMS["бензин"])
    q_required = I_norm * fire_area_m2                           # норм. расход, л/с
    q_effective = available_foam_ls * (1.0 - roof_obstruction_frac)  # расход, достигающий поверхности
    margin = q_effective / max(q_required, 1e-9)                 # запас (>1 — достаточно)

    if roof_obstruction_frac >= 0.70:
        return dict(feasible=False, margin=margin,
                    q_required=q_required, q_effective=q_effective,
                    reason=(f"каркас крыши блокирует {roof_obstruction_frac*100:.0f}% расхода → "
                            f"пена не покрывает зеркало горения"))
    if q_effective < q_required:
        deficit = q_required - q_effective
        return dict(feasible=False, margin=margin,
                    q_required=q_required, q_effective=q_effective,
                    reason=(f"Q_эфф={q_effective:.1f} л/с < норм. {q_required:.1f} л/с "
                            f"(дефицит {deficit:.1f} л/с при препятствии {roof_obstruction_frac*100:.0f}%)"))
    return dict(feasible=True, margin=margin,
                q_required=q_required, q_effective=q_effective,
                reason=(f"Q_эфф={q_effective:.1f} л/с ≥ норм. {q_required:.1f} л/с "
                        f"(запас ×{margin:.2f}, препятствие {roof_obstruction_frac*100:.0f}%)"))


# ══════════════════════════════════════════════════════════════════════════════
# 6. ПАРАМЕТРЫ РВС №9, крупный объект (сценарий А)
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class RVSParams:
    """Технические параметры вертикального стального резервуара."""
    designation: str          # обозначение РВС
    volume_m3: float          # номинальный объём, м³
    diameter_m: float         # диаметр, м
    height_m: float           # высота, м
    fuel: str                 # тип хранимого продукта
    fill_fraction: float = 0.70  # степень заполнения [0..1]

    @property
    def fire_area_m2(self) -> float:
        """Площадь зеркала (= площадь поперечного сечения), м²."""
        return math.pi * (self.diameter_m / 2) ** 2

    @property
    def circumference_m(self) -> float:
        """Периметр стенки, м."""
        return math.pi * self.diameter_m

    @property
    def fuel_volume_m3(self) -> float:
        """Объём топлива при текущем уровне заполнения, м³."""
        return self.volume_m3 * self.fill_fraction

    def cooling_flow_required_ls(self, burning: bool = True) -> float:
        return required_cooling_flow(self.diameter_m, burning)

    def foam_flow_required_ls(self) -> float:
        return required_foam_flow(self.fire_area_m2, self.fuel)


# Реальные параметры РВС крупного объекта (сценарий А)
RVS_9 = RVSParams(
    designation="РВС №9",
    volume_m3=20_000,
    diameter_m=40.0,          # оценочно: V=π/4·D²·H → при H=16м D≈40м
    height_m=15.9,
    fuel="бензин",            # прямогонный бензин (из PDF, 13:05 14.03)
    fill_fraction=0.475,      # 9500 т / (20000 м³ × ≈1.0 т/м³) ≈ 0.475 после выгорания 4500 т
)

RVS_17 = RVSParams(
    designation="РВС №17",
    volume_m3=20_000,
    diameter_m=40.0,
    height_m=15.9,
    fuel="нефть",
    fill_fraction=0.80,
)

# ── Нефтебаза (сценарий Б, РВС средний, ПТП 2015, вариант №2) ─────────────
# РВС №20 (V=2000 м³): D=14.62 м, H=11.2 м, P=46 м, S_зеркала=168 м², S_обвал=3410 м²
RVS_2000_SERP = RVSParams(
    designation="РВС №20 (нефтебаза)",
    volume_m3=2_000,
    diameter_m=14.62,         # из ПТП: P рез. = 46 м → D = P/π = 14.64 м
    height_m=11.2,
    fuel="бензин",            # АИ-92/АИ-95
    fill_fraction=0.70,
)

# РВС №13–17 (V=1000 м³): типовые размеры для РВС-1000
RVS_1000_SERP = RVSParams(
    designation="РВС №13–17 (нефтебаза)",
    volume_m3=1_000,
    diameter_m=10.9,          # оценочно: V=π/4·D²·H → H≈10.7 м → D≈10.9 м
    height_m=10.7,
    fuel="бензин",
    fill_fraction=0.70,
)


# ══════════════════════════════════════════════════════════════════════════════
# 7. СВОДНАЯ ТАБЛИЦА НОРМАТИВНЫХ ТРЕБОВАНИЙ
# ══════════════════════════════════════════════════════════════════════════════

def print_summary():
    """Вывести нормативные требования для объекта моделирования."""
    print("═" * 64)
    print("  НОРМАТИВНЫЕ ТРЕБОВАНИЯ — РВС НА ОБЪЕКТАХ НЕФТЕПРОДУКТОВ")
    print("  Источники: ГОСТ Р 51043-2002, СП 155.13130.2014")
    print("═" * 64)

    for rvs in [RVS_9, RVS_17]:
        burning = (rvs is RVS_9)
        print(f"\n  {rvs.designation}  (V={rvs.volume_m3:,} м³, D={rvs.diameter_m} м, "
              f"H={rvs.height_m} м, продукт: {rvs.fuel})")
        print(f"  Площадь зеркала:           {rvs.fire_area_m2:>8.0f} м²")
        print(f"  Периметр стенки:           {rvs.circumference_m:>8.1f} м")
        if burning:
            print(f"  Остаток продукта:          {rvs.fuel_volume_m3:>8.0f} м³ "
                  f"({rvs.fill_fraction*100:.0f}% заполнения)")
            print(f"\n  [Тушение] Норм. интенсивность подачи пены "
                  f"({rvs.fuel}):     {FOAM_INTENSITY_NORMS[rvs.fuel]:.3f} л/(м²·с)")
            print(f"  [Тушение] Требуемый расход пенного раствора: "
                  f"{rvs.foam_flow_required_ls():>6.1f} л/с")
            print(f"  [Тушение] Норм. время подачи пены:           "
                  f"{FOAM_APPLICATION_TIME_MIN[rvs.fuel]:.0f} мин")
        print(f"  [Охлаждение] Треб. расход охлаждения ({'горящий' if burning else 'смежный'}): "
              f"{rvs.cooling_flow_required_ls(burning):>6.1f} л/с")

    print("\n  Оборудование на пожаре:")
    print(f"  {'Ствол':<18} {'Расход':>8} {'Давление':>10} {'K-фактор':>10} {'Пена':>6}")
    print("  " + "-" * 56)
    for name, ns in NOZZLE_DB.items():
        foam_str = f"×{ns.foam_expansion:.0f}" if ns.foam_capable else "—"
        print(f"  {name:<18} {ns.flow_ls:>6.1f} л/с  {ns.P_MPa:>6.2f} МПа  "
              f"{ns.K:>8.2f}    {foam_str:>5}")

    # Проверка пенных атак по ГОСТ Р 51043
    print(f"\n  Анализ пенных атак (ГОСТ Р 51043-2002, СП 155.13130.2014):")
    print(f"  Площадь горения РВС №9: {RVS_9.fire_area_m2:.0f} м²")
    print(f"  Нормативный расход пены: {RVS_9.foam_flow_required_ls():.1f} л/с\n")

    attacks = [
        ("Атака №1", 2 * NOZZLE_DB["Акрон-Аполло"].flow_ls, 0.70,
         "ППП ПЧ-12 вышел из строя"),
        ("Атака №2", (NOZZLE_DB["Акрон-Аполло"].flow_ls +
                      NOZZLE_DB["Муссон-125"].flow_ls), 0.65,
         "каркас крыши внутри РВС"),
        ("Атака №6", (NOZZLE_DB["Антенор-1500"].flow_ls +
                      NOZZLE_DB["Муссон-125"].flow_ls +
                      2 * NOZZLE_DB["ГПС-1000"].flow_ls), 0.20,
         "ГПС-1000 с АКП-50 через люк"),
    ]
    for label, q_ls, roof_obs, note in attacks:
        res = foam_attack_feasibility(RVS_9.fire_area_m2, q_ls,
                                      roof_obstruction_frac=roof_obs)
        status = "✅ УСПЕХ" if res["feasible"] else "❌ НЕУДАЧА"
        print(f"  {label}: Q={q_ls:.1f} л/с, препятствие={roof_obs*100:.0f}%  → {status}")
        print(f"    Причина: {note}")
        print(f"    Расчёт:  Q_эфф={res['q_effective']:.1f} л/с, Q_норм={res['q_required']:.1f} л/с, "
              f"запас×{res['margin']:.2f}")
    print()


if __name__ == "__main__":
    print_summary()
