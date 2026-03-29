"""
generate_map_scheme.py — Схема интерактивной карты пожара для диссертации.
Воспроизводит все элементы визуализации из ОПИСАНИЕ_ПРОГРАММЫ.md.
"""
from __future__ import annotations
import os
import math
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import (FancyBboxPatch, Circle, Wedge,
                                 FancyArrowPatch, Arc, Rectangle, Polygon)
from matplotlib.lines import Line2D

_OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "figures")
os.makedirs(_OUT, exist_ok=True)

# Цветовая палитра (из tank_fire_sim.py)
P = dict(
    bg="#f0f2f8", canvas="#f0f2f8",
    fire="#ff4500", fire2="#ff8c00", flame="#ffcc00",
    rvs_burn="#c0392b", rvs_nbr="#2471a3", rvs_cool="#884422",
    water="#00aaff", foam="#27ae60", smoke="#a0b0b8",
    building="#95a5a6", ground="#a8d5a2", road="#b0b8c8",
    unit_ac="#e74c3c", unit_apt="#e67e22", unit_pns="#3498db",
    unit_panrk="#8e44ad", unit_ash="#f39c12",
    hydrant="#1abc9c", river="#85c1e9",
    success="#27ae60", warn="#e67e22", danger="#c0392b", info="#2980b9",
    text="#2c3e50", text2="#7f8c8d",
    obval="#d4c5a9",
)

def generate_map_scheme(filename: str = "map_scheme.png") -> str:
    fig, ax = plt.subplots(figsize=(16.5, 13.5), facecolor=P["canvas"])
    ax.set_xlim(0, 825)
    ax.set_ylim(0, 675)
    ax.set_facecolor(P["canvas"])
    ax.set_aspect("equal")
    ax.invert_yaxis()

    W, H = 825, 675

    # ══════════════════════════════════════════════════════════════════════
    # 1. ТЕРРИТОРИЯ: дорога, грунт, обвалование
    # ══════════════════════════════════════════════════════════════════════
    # Грунт
    ax.add_patch(Rectangle((0, 0), W, H, facecolor=P["ground"], alpha=0.3))

    # Дорога (горизонтальная)
    ax.add_patch(Rectangle((0, 580), W, 35, facecolor=P["road"], alpha=0.5))
    ax.text(W//2, 597, "Городская улица", ha="center", fontsize=8,
            color=P["text2"], style="italic")

    # Обвалование горящего РВС
    obval_cx, obval_cy = 300, 280
    obval_r = 130
    circle_obval = Circle((obval_cx, obval_cy), obval_r,
                          facecolor=P["obval"], edgecolor="#8b7d5e",
                          linewidth=2, alpha=0.3, linestyle="--")
    ax.add_patch(circle_obval)
    ax.text(obval_cx + obval_r + 5, obval_cy + obval_r - 10, "Обвалование",
            fontsize=7, color="#8b7d5e", rotation=-20)

    # Обвалование соседнего РВС
    obval2_cx, obval2_cy = 580, 280
    obval2_r = 100
    circle_obval2 = Circle((obval2_cx, obval2_cy), obval2_r,
                           facecolor=P["obval"], edgecolor="#8b7d5e",
                           linewidth=1.5, alpha=0.2, linestyle="--")
    ax.add_patch(circle_obval2)

    # ══════════════════════════════════════════════════════════════════════
    # 2. ГОРЯЩИЙ РВС (основной)
    # ══════════════════════════════════════════════════════════════════════
    rvs_cx, rvs_cy = 300, 280
    rvs_r = 65

    # Тело РВС
    ax.add_patch(Circle((rvs_cx, rvs_cy), rvs_r,
                        facecolor=P["rvs_burn"], edgecolor="#922b21",
                        linewidth=3, alpha=0.85))

    # Каркас крыши (плавающая — решётка)
    for angle in range(0, 180, 30):
        rad = math.radians(angle)
        x1 = rvs_cx + rvs_r * 0.8 * math.cos(rad)
        y1 = rvs_cy + rvs_r * 0.8 * math.sin(rad)
        x2 = rvs_cx - rvs_r * 0.8 * math.cos(rad)
        y2 = rvs_cy - rvs_r * 0.8 * math.sin(rad)
        ax.plot([x1, x2], [y1, y2], color="#666", linewidth=0.8, alpha=0.5)
    ax.text(rvs_cx, rvs_cy + 10, "Горящий\nРВС", ha="center", va="center",
            fontsize=9, fontweight="bold", color="white")
    ax.text(rvs_cx, rvs_cy + 30, "V=20 000 м³", ha="center",
            fontsize=7, color="#fdd")

    # ══════════════════════════════════════════════════════════════════════
    # 3. ПЛАМЯ (12 языков с пульсацией)
    # ══════════════════════════════════════════════════════════════════════
    rng = np.random.RandomState(42)
    for i in range(12):
        angle = i * 30 + rng.uniform(-10, 10)
        rad = math.radians(angle)
        base_x = rvs_cx + (rvs_r - 15) * math.cos(rad)
        base_y = rvs_cy + (rvs_r - 15) * math.sin(rad)
        flame_h = rng.uniform(25, 50)
        flame_w = rng.uniform(8, 16)

        # Язык пламени (треугольник)
        dx = flame_h * math.cos(rad)
        dy = flame_h * math.sin(rad)
        perp_x = flame_w * math.cos(rad + math.pi/2)
        perp_y = flame_w * math.sin(rad + math.pi/2)

        tri = Polygon([
            (base_x - perp_x, base_y - perp_y),
            (base_x + perp_x, base_y + perp_y),
            (base_x + dx * 1.2, base_y + dy * 1.2),
        ], facecolor=P["fire"] if i % 3 else P["flame"],
            edgecolor=P["fire2"], linewidth=0.5, alpha=0.7)
        ax.add_patch(tri)

    # ══════════════════════════════════════════════════════════════════════
    # 4. ЗОНА ТЕПЛОВОГО ВОЗДЕЙСТВИЯ
    # ══════════════════════════════════════════════════════════════════════
    thermal_r = rvs_r + 60
    thermal = Circle((rvs_cx, rvs_cy), thermal_r,
                     facecolor="none", edgecolor=P["fire2"],
                     linewidth=1.5, linestyle=":", alpha=0.5)
    ax.add_patch(thermal)
    ax.text(rvs_cx + thermal_r - 10, rvs_cy - thermal_r + 15,
            "Зона теплового\nвоздействия", fontsize=6, color=P["fire2"],
            ha="center", rotation=-30)

    # ══════════════════════════════════════════════════════════════════════
    # 5. ДЫМОВОЙ ШЛЕЙФ (анимированный)
    # ══════════════════════════════════════════════════════════════════════
    for i in range(8):
        sx = rvs_cx + rng.uniform(-30, 30) + i * 15
        sy = rvs_cy - rvs_r - 20 - i * 20
        sr = 15 + i * 8
        ax.add_patch(Circle((sx, sy), sr,
                            facecolor=P["smoke"], edgecolor="none",
                            alpha=0.15 - i * 0.015))

    # ══════════════════════════════════════════════════════════════════════
    # 6. СОСЕДНИЙ РВС
    # ══════════════════════════════════════════════════════════════════════
    nbr_cx, nbr_cy = 580, 280
    nbr_r = 50
    ax.add_patch(Circle((nbr_cx, nbr_cy), nbr_r,
                        facecolor=P["rvs_nbr"], edgecolor="#1a5276",
                        linewidth=2.5, alpha=0.7))
    ax.text(nbr_cx, nbr_cy, "Соседний\nРВС", ha="center", va="center",
            fontsize=8, fontweight="bold", color="white")

    # Кольцо орошения
    ax.add_patch(Circle((nbr_cx, nbr_cy), nbr_r + 8,
                        facecolor="none", edgecolor=P["water"],
                        linewidth=2, linestyle="--", alpha=0.6))
    ax.text(nbr_cx + nbr_r + 12, nbr_cy - 15, "Кольцо\nорошения",
            fontsize=6, color=P["water"])

    # ══════════════════════════════════════════════════════════════════════
    # 7. СТВОЛЫ ОХЛАЖДЕНИЯ (с подписями)
    # ══════════════════════════════════════════════════════════════════════
    trunks = [
        (rvs_cx, rvs_cy + rvs_r + 30, "ЮГ", "Антенор-1500 №1"),
        (rvs_cx + rvs_r + 30, rvs_cy, "ВОСТОК", "Антенор-1500 №2"),
        (rvs_cx - rvs_r - 30, rvs_cy, "ЗАПАД", "Антенор-1500 №3"),
        (rvs_cx, rvs_cy + rvs_r + 55, "ЮГ-2", "Антенор-1500 №4"),
        (rvs_cx + rvs_r + 55, rvs_cy + 20, "ВОСТОК-2", "ЛС-С330"),
    ]
    for tx, ty, sector, label in trunks:
        # Линия потока (от ствола к РВС)
        ax.annotate("", xy=(rvs_cx + (tx-rvs_cx)*0.4, rvs_cy + (ty-rvs_cy)*0.4),
                    xytext=(tx, ty),
                    arrowprops=dict(arrowstyle="-|>", color=P["water"],
                                   lw=2, alpha=0.6))
        # Значок ствола
        ax.add_patch(Circle((tx, ty), 5, facecolor=P["water"],
                            edgecolor="white", linewidth=1.5))
        ax.text(tx + 8, ty - 3, label, fontsize=5.5, color=P["info"])

    # ══════════════════════════════════════════════════════════════════════
    # 8. ЛИНИИ ВОДОСНАБЖЕНИЯ + ПНС
    # ══════════════════════════════════════════════════════════════════════
    # Открытый водоисточник (внизу карты)
    river_y = 640
    ax.add_patch(Rectangle((50, river_y), 300, 25,
                           facecolor=P["river"], edgecolor=P["water"],
                           linewidth=1.5, alpha=0.5))
    ax.text(200, river_y + 12, "Открытый водоисточник", ha="center",
            fontsize=8, color="#1a5276", fontweight="bold")

    # ПНС (3 штуки)
    pns_positions = [(120, 540), (250, 540), (400, 540)]
    for i, (px, py) in enumerate(pns_positions):
        ax.add_patch(Rectangle((px - 12, py - 8), 24, 16,
                               facecolor=P["unit_pns"], edgecolor="white",
                               linewidth=1.5))
        ax.text(px, py, f"ПНС", ha="center", va="center",
                fontsize=6, fontweight="bold", color="white")
        ax.text(px, py + 14, f"№{i+1}", ha="center", fontsize=5, color=P["text2"])

        # Линия от ПНС к водоисточнику
        ax.plot([px, px], [py + 8, river_y], color=P["water"],
                linewidth=1.5, linestyle="-", alpha=0.4)

        # Линия от ПНС к стволам (магистральная)
        ax.plot([px, rvs_cx + (i-1)*30], [py - 8, rvs_cy + rvs_r + 25],
                color=P["water"], linewidth=1, linestyle="--", alpha=0.3)

    # ══════════════════════════════════════════════════════════════════════
    # 9. ПОЖАРНАЯ ТЕХНИКА (по мере прибытия)
    # ══════════════════════════════════════════════════════════════════════
    units = [
        (60,  460, "АЦ", P["unit_ac"], "АЦ-1"),
        (100, 460, "АЦ", P["unit_ac"], "АЦ-2"),
        (140, 460, "АЦ", P["unit_ac"], "АЦ-3"),
        (200, 460, "АПТ", P["unit_apt"], "АПТ-1"),
        (250, 460, "АПТ", P["unit_apt"], "АПТ-2"),
        (320, 460, "АР", P["unit_apt"], "АР-2"),
        (380, 460, "АШ", P["unit_ash"], "АШ"),
        (440, 460, "ПАНРК", P["unit_panrk"], "ПАНРК"),
        (520, 460, "АКП", P["unit_panrk"], "АКП-50"),
        (600, 460, "ПП", P["unit_ash"], "Пож. поезд"),
    ]
    for ux, uy, code, color, label in units:
        ax.add_patch(Rectangle((ux - 14, uy - 10), 28, 20,
                               facecolor=color, edgecolor="white",
                               linewidth=1, alpha=0.85))
        ax.text(ux, uy, code, ha="center", va="center",
                fontsize=6, fontweight="bold", color="white")
        ax.text(ux, uy + 16, label, ha="center", fontsize=5, color=P["text2"])

    # ══════════════════════════════════════════════════════════════════════
    # 10. ШТАБ ПОЖАРОТУШЕНИЯ (ОШ)
    # ══════════════════════════════════════════════════════════════════════
    osh_x, osh_y = 680, 180
    ax.add_patch(FancyBboxPatch((osh_x - 25, osh_y - 15), 50, 30,
                                boxstyle="round,pad=3",
                                facecolor="#e8daef", edgecolor="#8e44ad",
                                linewidth=2))
    ax.text(osh_x, osh_y, "ОШ", ha="center", va="center",
            fontsize=10, fontweight="bold", color="#6c3483")
    ax.text(osh_x, osh_y + 22, "Оперативный штаб", ha="center",
            fontsize=6, color="#7d3c98")

    # ══════════════════════════════════════════════════════════════════════
    # 11. БОЕВЫЕ УЧАСТКИ (БУ-1, БУ-2, БУ-3) с секторами
    # ══════════════════════════════════════════════════════════════════════
    bu_data = [
        ("БУ-1\n(ЮГ)", rvs_cx, rvs_cy + rvs_r + 80, "#e74c3c", "НБУ-1"),
        ("БУ-2\n(ВОСТОК)", rvs_cx + rvs_r + 80, rvs_cy, "#2980b9", "НБУ-2"),
        ("БУ-3\n(ЗАПАД)", rvs_cx - rvs_r - 80, rvs_cy, "#27ae60", "НБУ-3"),
    ]
    for label, bx, by, color, nbu in bu_data:
        ax.add_patch(FancyBboxPatch((bx - 22, by - 15), 44, 30,
                                    boxstyle="round,pad=2",
                                    facecolor=color, edgecolor="white",
                                    linewidth=1.5, alpha=0.7))
        ax.text(bx, by - 3, label, ha="center", va="center",
                fontsize=6, fontweight="bold", color="white")
        ax.text(bx, by + 20, nbu, ha="center", fontsize=6, color=color,
                fontweight="bold")

    # Секторные линии (от центра РВС)
    for angle, color in [(270, "#e74c3c"), (0, "#2980b9"), (180, "#27ae60")]:
        rad = math.radians(angle)
        x2 = rvs_cx + 120 * math.cos(rad)
        y2 = rvs_cy + 120 * math.sin(rad)
        ax.plot([rvs_cx, x2], [rvs_cy, y2], color=color,
                linewidth=1, linestyle=":", alpha=0.5)

    # ══════════════════════════════════════════════════════════════════════
    # 12. ПОЖАРНЫЕ ГИДРАНТЫ
    # ══════════════════════════════════════════════════════════════════════
    hydrants = [(500, 590), (600, 590), (700, 590)]
    for hx, hy in hydrants:
        # Крестик (нормативное обозначение)
        s = 6
        ax.plot([hx - s, hx + s], [hy - s, hy + s], color=P["hydrant"],
                linewidth=2)
        ax.plot([hx - s, hx + s], [hy + s, hy - s], color=P["hydrant"],
                linewidth=2)
        ax.add_patch(Circle((hx, hy), s + 2, facecolor="none",
                            edgecolor=P["hydrant"], linewidth=1.5))
        ax.text(hx, hy + 12, "ПГ", ha="center", fontsize=5, color=P["hydrant"])

    # ══════════════════════════════════════════════════════════════════════
    # 13. ЗДАНИЯ
    # ══════════════════════════════════════════════════════════════════════
    buildings = [
        (700, 350, 50, 30, "Лаборатория"),
        (700, 420, 50, 30, "Насосная"),
        (700, 490, 50, 25, "Склад ГСМ"),
    ]
    for bx, by, bw, bh, label in buildings:
        ax.add_patch(Rectangle((bx, by), bw, bh,
                               facecolor=P["building"], edgecolor="#7f8c8d",
                               linewidth=1.5, alpha=0.7))
        ax.text(bx + bw/2, by + bh/2, label, ha="center", va="center",
                fontsize=5.5, color="white", fontweight="bold")

    # Индикация вторичного очага (на складе ГСМ)
    ax.add_patch(Circle((735, 500), 4, facecolor=P["fire"],
                        edgecolor=P["flame"], linewidth=1.5, alpha=0.8))
    ax.text(750, 500, "Вторичный\nочаг", fontsize=5, color=P["danger"])

    # ══════════════════════════════════════════════════════════════════════
    # 14. КОМПАС
    # ══════════════════════════════════════════════════════════════════════
    cx, cy = 770, 60
    ax.annotate("С", xy=(cx, cy - 25), fontsize=10, fontweight="bold",
                ha="center", color=P["text"])
    ax.annotate("", xy=(cx, cy - 22), xytext=(cx, cy + 15),
                arrowprops=dict(arrowstyle="-|>", color=P["text"], lw=2))
    ax.plot([cx - 12, cx + 12], [cy, cy], color=P["text2"], linewidth=1)
    ax.text(cx + 16, cy, "В", fontsize=7, color=P["text2"], va="center")
    ax.text(cx - 20, cy, "З", fontsize=7, color=P["text2"], va="center")
    ax.text(cx, cy + 20, "Ю", fontsize=7, color=P["text2"], ha="center")

    # ══════════════════════════════════════════════════════════════════════
    # 15. МАСШТАБНАЯ ЛИНЕЙКА
    # ══════════════════════════════════════════════════════════════════════
    sx, sy = 620, 40
    ax.plot([sx, sx + 80], [sy, sy], color=P["text"], linewidth=2)
    ax.plot([sx, sx], [sy - 4, sy + 4], color=P["text"], linewidth=2)
    ax.plot([sx + 80, sx + 80], [sy - 4, sy + 4], color=P["text"], linewidth=2)
    ax.text(sx + 40, sy + 8, "100 м", ha="center", fontsize=7, color=P["text"])

    # ══════════════════════════════════════════════════════════════════════
    # 16. БАННЕР СТАТУСА
    # ══════════════════════════════════════════════════════════════════════
    ax.add_patch(FancyBboxPatch((W//2 - 150, 5), 300, 24,
                                boxstyle="round,pad=3",
                                facecolor=P["warn"], edgecolor="white",
                                linewidth=1.5, alpha=0.9))
    ax.text(W//2, 17, "ГОТОВНОСТЬ К ПЕННОЙ АТАКЕ",
            ha="center", va="center", fontsize=10, fontweight="bold",
            color="white")

    # ══════════════════════════════════════════════════════════════════════
    # ЗАГОЛОВОК И ПОДПИСИ
    # ══════════════════════════════════════════════════════════════════════
    ax.text(W//2, -20, "Схема интерактивной карты пожара РВС — САУР-ПСП",
            ha="center", fontsize=13, fontweight="bold", color=P["text"])
    ax.text(W//2, -8, "Все элементы визуализации оперативной обстановки (825×675 пикселей)",
            ha="center", fontsize=8, color=P["text2"], style="italic")

    # Легенда
    legend_items = [
        (P["rvs_burn"], "Горящий РВС"),
        (P["rvs_nbr"], "Соседний РВС"),
        (P["fire"], "Пламя (12 языков)"),
        (P["water"], "Стволы / водоснабжение"),
        (P["unit_ac"], "АЦ / техника"),
        (P["unit_pns"], "ПНС"),
        (P["unit_panrk"], "ПАНРК / АКП"),
        (P["hydrant"], "Пожарные гидранты"),
        (P["building"], "Здания"),
    ]
    for i, (color, label) in enumerate(legend_items):
        lx = 30 + (i % 5) * 160
        ly = H + 20 + (i // 5) * 15
        ax.add_patch(Rectangle((lx, ly - 4), 10, 8, facecolor=color,
                               edgecolor="white", linewidth=0.5))
        ax.text(lx + 14, ly, label, fontsize=6.5, color=P["text"], va="center")

    ax.set_ylim(H + 40, -30)

    path = os.path.join(_OUT, filename)
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path


if __name__ == "__main__":
    path = generate_map_scheme()
    print(f"Схема карты сохранена: {path}")
