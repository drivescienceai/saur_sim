"""
report_generator.py
════════════════════════════════════════════════════════════════════════════════
Генератор отчётов по результатам моделирования тушения пожара РВС.

Форматы вывода:
  1. PDF-отчёт          — полный отчёт для оперативного штаба / руководства
  2. JSON-выгрузка      — структурированные данные для написания научной статьи
  3. DOCX-выгрузка      — текстовое резюме в формате Word (python-docx)

Зависимости: reportlab, python-docx (уже установлены)
════════════════════════════════════════════════════════════════════════════════
"""
from __future__ import annotations
import json
import math
import os
import io
import datetime
from typing import TYPE_CHECKING, List, Dict, Optional

# Импорт reportlab для генерации PDF
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.units import mm, cm
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT, TA_JUSTIFY
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    HRFlowable, PageBreak, Image as RLImage, KeepTogether,
)
from reportlab.platypus.flowables import Flowable
from reportlab.pdfgen import canvas as pdf_canvas
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

# Matplotlib для встраивания графиков в PDF
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np

if TYPE_CHECKING:
    from .tank_fire_sim import TankFireSim

# ══════════════════════════════════════════════════════════════════════════════
# ШРИФТЫ (поддержка кириллицы в reportlab)
# ══════════════════════════════════════════════════════════════════════════════

def _register_fonts():
    """Зарегистрировать шрифты с поддержкой кириллицы."""
    # Путь к шрифтам в Windows
    font_paths = [
        ("C:/Windows/Fonts/arial.ttf",     "Arial"),
        ("C:/Windows/Fonts/arialbd.ttf",   "Arial-Bold"),
        ("C:/Windows/Fonts/ariali.ttf",    "Arial-Italic"),
        ("C:/Windows/Fonts/cour.ttf",      "Courier-Cyr"),
        ("C:/Windows/Fonts/courbd.ttf",    "Courier-Cyr-Bold"),
    ]
    registered = {}
    for path, name in font_paths:
        if os.path.exists(path):
            try:
                pdfmetrics.registerFont(TTFont(name, path))
                registered[name] = True
            except Exception:
                pass
    return registered


_FONTS = _register_fonts()
_FONT     = "Arial"      if "Arial"      in _FONTS else "Helvetica"
_FONT_B   = "Arial-Bold" if "Arial-Bold" in _FONTS else "Helvetica-Bold"
_FONT_I   = "Arial-Italic"if "Arial-Italic" in _FONTS else "Helvetica-Oblique"
_FONT_MON = "Courier-Cyr" if "Courier-Cyr" in _FONTS else "Courier"


# ══════════════════════════════════════════════════════════════════════════════
# ЦВЕТА ОТЧЁТА
# ══════════════════════════════════════════════════════════════════════════════

C_FIRE    = colors.HexColor("#c0392b")   # красный — огонь / опасность
C_WATER   = colors.HexColor("#2471a3")   # синий — вода / охлаждение
C_SUCCESS = colors.HexColor("#27ae60")   # зелёный — успех / ликвидация
C_WARN    = colors.HexColor("#e67e22")   # оранжевый — предупреждение
C_DARK    = colors.HexColor("#1a1f2e")   # тёмный фон заголовка
C_PANEL   = colors.HexColor("#eaf2fb")   # светло-синий фон панелей
C_HEADER  = colors.HexColor("#2c3e50")   # тёмно-синий — шапки таблиц
C_STRIPE  = colors.HexColor("#f4f6f7")   # полоска чередования строк
C_BORDER  = colors.HexColor("#aab7c4")   # рамки таблиц


# ══════════════════════════════════════════════════════════════════════════════
# ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ
# ══════════════════════════════════════════════════════════════════════════════

def _para(text: str, style: ParagraphStyle) -> Paragraph:
    """Создать параграф с указанным стилем."""
    return Paragraph(text, style)


def _hr() -> HRFlowable:
    """Горизонтальная разделительная линия."""
    return HRFlowable(width="100%", thickness=0.5, color=C_BORDER, spaceAfter=4)


def _table_style(header_rows: int = 1) -> TableStyle:
    """Стандартный стиль для таблиц отчёта."""
    cmds = [
        # Шапка таблицы
        ("BACKGROUND",   (0, 0), (-1, header_rows - 1), C_HEADER),
        ("TEXTCOLOR",    (0, 0), (-1, header_rows - 1), colors.white),
        ("FONTNAME",     (0, 0), (-1, header_rows - 1), _FONT_B),
        ("FONTSIZE",     (0, 0), (-1, header_rows - 1), 9),
        ("ALIGN",        (0, 0), (-1, header_rows - 1), "CENTER"),
        # Данные
        ("FONTNAME",     (0, header_rows), (-1, -1), _FONT),
        ("FONTSIZE",     (0, header_rows), (-1, -1), 8),
        ("ALIGN",        (0, header_rows), (0, -1),  "LEFT"),
        ("ALIGN",        (1, header_rows), (-1, -1), "CENTER"),
        ("ROWBACKGROUNDS", (0, header_rows), (-1, -1), [colors.white, C_STRIPE]),
        # Рамки
        ("GRID",         (0, 0), (-1, -1), 0.4, C_BORDER),
        ("TOPPADDING",   (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING",(0, 0), (-1, -1), 4),
        ("LEFTPADDING",  (0, 0), (-1, -1), 6),
    ]
    return TableStyle(cmds)


def _fmt_time(t: int, base_h: int = 3, base_m: int = 20) -> str:
    """Перевести минуты симуляции в метку реального времени."""
    total_m = base_m + t
    h = (base_h + total_m // 60) % 24
    m = total_m % 60
    day = 14 + (base_h * 60 + base_m + t) // (24 * 60)
    return f"{day:02d}.03  {h:02d}:{m:02d}"


def _duration_str(minutes: int) -> str:
    """Форматировать продолжительность в ч мин."""
    h = minutes // 60
    m = minutes % 60
    if h > 0:
        return f"{h} ч {m} мин"
    return f"{m} мин"


# ══════════════════════════════════════════════════════════════════════════════
# ГЕНЕРАТОР ГРАФИКОВ ДЛЯ ОТЧЁТА
# ══════════════════════════════════════════════════════════════════════════════

def _build_metrics_figure(sim: "TankFireSim") -> io.BytesIO:
    """Построить фигуру метрик (4 субграфика) и вернуть PNG в памяти."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 8),
                             facecolor="#1a1f2e",
                             gridspec_kw=dict(hspace=0.45, wspace=0.35))
    fig.patch.set_facecolor("#1a1f2e")

    ts_f  = [x[0] for x in sim.h_fire]
    val_f = [x[1] for x in sim.h_fire]
    ts_w  = [x[0] for x in sim.h_water]
    val_w = [x[1] for x in sim.h_water]
    ts_r  = [x[0] for x in sim.h_risk]
    val_r = [x[1] for x in sim.h_risk]
    ts_t  = [x[0] for x in sim.h_trunks]
    val_t = [x[1] for x in sim.h_trunks]

    panel_bg = "#0d1420"
    text_c   = "#ecf0f1"
    grid_c   = "#2c3e50"

    def _style_ax(ax, title):
        ax.set_facecolor(panel_bg)
        ax.set_title(title, color=text_c, fontsize=10, pad=4)
        ax.tick_params(colors="#95a5a6", labelsize=8)
        for sp in ax.spines.values():
            sp.set_color(grid_c)
        ax.grid(True, color=grid_c, linewidth=0.4, alpha=0.7)

    # Площадь пожара
    ax = axes[0, 0]
    _style_ax(ax, "Площадь пожара, м²")
    if ts_f:
        ax.plot(ts_f, val_f, color="#ff4500", linewidth=1.8, label="S(t)")
        ax.fill_between(ts_f, val_f, alpha=0.25, color="#ff4500")
        ax.set_ylabel("м²", color="#95a5a6", fontsize=8)
        ax.set_xlim(0, max(ts_f[-1], 100))
    ax.legend(fontsize=7, facecolor=panel_bg, edgecolor=grid_c, labelcolor=text_c)

    # Расход ОВ
    ax = axes[0, 1]
    _style_ax(ax, "Суммарный расход ОВ, л/с")
    if ts_w:
        ax.plot(ts_w, val_w, color="#00aaff", linewidth=1.8, label="Q(t)")
        ax.fill_between(ts_w, val_w, alpha=0.2, color="#00aaff")
        ax.set_ylabel("л/с", color="#95a5a6", fontsize=8)
        ax.set_xlim(0, max(ts_w[-1], 100))
    ax.legend(fontsize=7, facecolor=panel_bg, edgecolor=grid_c, labelcolor=text_c)

    # Число стволов
    ax = axes[1, 0]
    _style_ax(ax, "Число стволов охлаждения РВС")
    if ts_t:
        ax.step(ts_t, val_t, color="#3498db", linewidth=1.8, where="post", label="N стволов")
        ax.fill_between(ts_t, val_t, alpha=0.2, color="#3498db", step="post")
        ax.axhline(7, color="#27ae60", linewidth=1, linestyle=":", alpha=0.9,
                   label="цель: 7 стволов")
        ax.set_xlim(0, max(ts_t[-1], 100))
    ax.legend(fontsize=7, facecolor=panel_bg, edgecolor=grid_c, labelcolor=text_c)

    # Индекс риска
    ax = axes[1, 1]
    _style_ax(ax, "Интегральный индекс риска")
    if ts_r:
        ax.plot(ts_r, val_r, color="#c0392b", linewidth=1.8, label="R(t)")
        ax.fill_between(ts_r, val_r, alpha=0.15, color="#c0392b")
        ax.axhline(0.75, color="#c0392b", linewidth=1, linestyle=":",
                   alpha=0.8, label="порог критич. (0.75)")
        ax.axhline(0.50, color="#e67e22", linewidth=0.8, linestyle=":",
                   alpha=0.6, label="порог высок. (0.50)")
        ax.set_ylim(0, 1.05)
        ax.set_xlim(0, max(ts_r[-1], 100))
    ax.legend(fontsize=7, facecolor=panel_bg, edgecolor=grid_c, labelcolor=text_c)

    plt.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=130, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    buf.seek(0)
    plt.close(fig)
    return buf


def _build_rl_figure(sim: "TankFireSim") -> io.BytesIO:
    """Построить фигуру RL-агента (Q-значения + частота выбора + кривая наград)."""
    try:
        from .tank_fire_sim import ACTIONS, LEVEL_C, N_ACT, P
    except ImportError:
        from tank_fire_sim import ACTIONS, LEVEL_C, N_ACT, P

    fig, axes = plt.subplots(1, 3, figsize=(14, 4),
                             facecolor="#1a1f2e",
                             gridspec_kw=dict(wspace=0.35))
    fig.patch.set_facecolor("#1a1f2e")
    panel_bg = "#0d1420"
    text_c   = "#ecf0f1"
    grid_c   = "#2c3e50"

    def _style_ax(ax, title):
        ax.set_facecolor(panel_bg)
        ax.set_title(title, color=text_c, fontsize=9, pad=4)
        ax.tick_params(colors="#95a5a6", labelsize=7)
        for sp in ax.spines.values():
            sp.set_color(grid_c)
        ax.grid(True, color=grid_c, linewidth=0.4, alpha=0.7)

    codes = [a[0] for a in ACTIONS]
    cols  = [LEVEL_C[a[1]] for a in ACTIONS]

    # Q-значения
    ax = axes[0]
    _style_ax(ax, "Q-значения действий")
    qv = sim.agent.q_values(sim._state())
    bars = ax.bar(range(N_ACT), qv, color=cols, alpha=0.85, width=0.7)
    if 0 <= sim.last_action < N_ACT:
        bars[sim.last_action].set_edgecolor("#f1c40f")
        bars[sim.last_action].set_linewidth(2)
    ax.set_xticks(range(N_ACT))
    ax.set_xticklabels(codes, rotation=45, ha="right", fontsize=7, color="#95a5a6")

    # Частота выбора
    ax = axes[1]
    _style_ax(ax, "Частота выбора действий")
    cnt = sim.agent.action_counts
    if cnt.sum() > 0:
        ax.bar(range(N_ACT), cnt / max(cnt.sum(), 1), color=cols, alpha=0.8, width=0.7)
    ax.set_xticks(range(N_ACT))
    ax.set_xticklabels(codes, rotation=45, ha="right", fontsize=7, color="#95a5a6")

    # Кривая наград
    ax = axes[2]
    _style_ax(ax, "Накопленная награда")
    if sim.h_reward:
        rw = sim.h_reward
        cumrew = np.cumsum(rw)
        ax.plot(cumrew, color="#27ae60", linewidth=1.5, label="Σ reward")
        if len(rw) > 20:
            ma = np.convolve(rw, np.ones(20) / 20, mode="valid")
            ax.plot(range(19, len(rw)), ma.cumsum() +
                    (cumrew[18] if len(cumrew) > 18 else 0),
                    color="#e67e22", linewidth=0.8, alpha=0.7, label="MA-20")
    ax.legend(fontsize=7, facecolor=panel_bg, edgecolor=grid_c, labelcolor=text_c)

    plt.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=130, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    buf.seek(0)
    plt.close(fig)
    return buf


# ══════════════════════════════════════════════════════════════════════════════
# СБОР СТАТИСТИКИ ПО ИТОГАМ СИМУЛЯЦИИ
# ══════════════════════════════════════════════════════════════════════════════

def collect_stats(sim: "TankFireSim") -> dict:
    """Собрать итоговую статистику из объекта TankFireSim.

    Returns:
        dict с полями: scenario, duration_min, final_phase, fire_area_final,
                        max_fire_area, min_fire_area, total_water_ls_avg,
                        foam_attacks, extinguished, localized,
                        max_risk, mean_risk, n_trunks_max,
                        rl_epsilon, rl_total_reward, action_distribution,
                        events_count, timeline_events
    """
    h_fire  = sim.h_fire
    h_water = sim.h_water
    h_risk  = sim.h_risk
    h_trunks= sim.h_trunks

    fire_vals  = [v for _, v in h_fire]
    water_vals = [v for _, v in h_water]
    risk_vals  = [v for _, v in h_risk]
    trunk_vals = [v for _, v in h_trunks]

    try:
        from .tank_fire_sim import ACTIONS, SCENARIOS
    except ImportError:
        from tank_fire_sim import ACTIONS, SCENARIOS
    cfg = SCENARIOS[sim.scenario]

    # Распределение действий
    ac = sim.agent.action_counts
    total_ac = max(ac.sum(), 1)
    action_dist = {
        ACTIONS[i][0]: {
            "count": int(ac[i]),
            "fraction": float(ac[i] / total_ac),
            "description": ACTIONS[i][2],
        }
        for i in range(len(ACTIONS))
        if ac[i] > 0
    }
    action_dist_sorted = dict(
        sorted(action_dist.items(), key=lambda x: x[1]["count"], reverse=True)
    )

    # События из журнала
    events_log = [
        {"t_min": t, "color": c, "text": txt}
        for t, c, txt in sim.events[:200]   # ограничить 200 записями
    ]

    return {
        # Сценарий
        "scenario_key":    sim.scenario,
        "scenario_name":   cfg["name"],
        "rvs_name":        cfg["rvs_name"],
        "fuel":            cfg["fuel"],
        "fire_rank":       cfg["fire_rank_default"],
        "total_sim_min":   cfg["total_min"],
        # Время
        "sim_duration_min": sim.t,
        "sim_duration_str": _duration_str(sim.t),
        "generated_at":    datetime.datetime.now().isoformat(timespec="seconds"),
        # Фаза и исход
        "final_phase":     sim.phase,
        "extinguished":    sim.extinguished,
        "localized":       sim.localized,
        "outcome":         "ликвидирован" if sim.extinguished else
                           ("локализован" if sim.localized else "активный"),
        # Площадь пожара
        "fire_area_initial_m2": cfg["initial_fire_area"],
        "fire_area_final_m2":   sim.fire_area,
        "fire_area_max_m2":     max(fire_vals) if fire_vals else 0,
        "fire_area_min_m2":     min(fire_vals) if fire_vals else 0,
        # Водоснабжение
        "water_flow_max_ls":    max(water_vals) if water_vals else 0,
        "water_flow_mean_ls":   sum(water_vals) / max(len(water_vals), 1),
        # Пенные атаки
        "foam_attacks_total":   sim.foam_attacks,
        "foam_flow_last_ls":    sim.foam_flow_ls,
        "roof_obstruction":     sim.roof_obstruction,
        "akp50_used":           sim.akp50_available,
        # Силы
        "trunks_burn_final":    sim.n_trunks_burn,
        "trunks_burn_max":      max(trunk_vals) if trunk_vals else 0,
        "trunks_nbr_final":     sim.n_trunks_nbr,
        "n_pns":                sim.n_pns,
        "n_bu":                 sim.n_bu,
        "has_shtab":            sim.has_shtab,
        "secondary_fire":       sim.secondary_fire,
        "spill":                sim.spill,
        # Риск
        "risk_max":             max(risk_vals) if risk_vals else 0,
        "risk_mean":            sum(risk_vals) / max(len(risk_vals), 1),
        "risk_final":           sim._risk(),
        # RL-агент
        "rl_epsilon":           sim.agent.epsilon,
        "rl_total_reward":      float(sum(sim.h_reward)) if sim.h_reward else 0.0,
        "rl_steps":             len(sim.h_reward),
        "rl_episodes":          len(sim.agent.episode_rewards),
        "action_distribution":  action_dist_sorted,
        # Хронология
        "events_count":         len(sim.events),
        "events_log":           events_log,
        # Нормативы ГОСТ
        "foam_intensity_norm":  cfg["foam_intensity"],
        "q_foam_required_ls":   cfg["foam_intensity"] * cfg["initial_fire_area"],
        "q_cooling_required_ls": 0.8 * math.pi * cfg["rvs_diameter_m"],
    }


# ══════════════════════════════════════════════════════════════════════════════
# PDF-ОТЧЁТ
# ══════════════════════════════════════════════════════════════════════════════

class _PageTemplate:
    """Колонтитулы страниц PDF-отчёта."""
    def __init__(self, title: str, stats: dict):
        self.title = title
        self.stats = stats

    def __call__(self, canv: pdf_canvas.Canvas, doc):
        canv.saveState()
        w, h = A4
        # Верхний колонтитул
        canv.setFillColor(C_DARK)
        canv.rect(0, h - 18*mm, w, 18*mm, fill=1, stroke=0)
        canv.setFont(_FONT_B, 9)
        canv.setFillColor(colors.white)
        canv.drawString(15*mm, h - 12*mm, "ОТЧЁТ О МОДЕЛИРОВАНИИ ТУШЕНИЯ ПОЖАРА РВС")
        canv.setFont(_FONT, 8)
        canv.drawRightString(w - 15*mm, h - 12*mm,
                             f"Создан: {self.stats['generated_at']}")
        # Нижний колонтитул
        canv.setFillColor(C_HEADER)
        canv.rect(0, 0, w, 12*mm, fill=1, stroke=0)
        canv.setFont(_FONT, 7)
        canv.setFillColor(colors.white)
        canv.drawString(15*mm, 4*mm, "САУР ПСП — Система адаптивного управления ресурсами ПСП")
        canv.drawRightString(w - 15*mm, 4*mm, f"Стр. {doc.page}")
        canv.restoreState()


def generate_pdf_report(sim: "TankFireSim", output_path: str = "") -> str:
    """Сгенерировать PDF-отчёт по результатам моделирования.

    Args:
        sim:         экземпляр TankFireSim после завершения симуляции
        output_path: путь к выходному файлу (если пусто — авто-имя в текущей папке)

    Returns:
        Путь к созданному PDF-файлу
    """
    stats = collect_stats(sim)

    # Авто-имя файла
    if not output_path:
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = os.path.join(
            os.path.dirname(__file__),
            f"report_{stats['scenario_key']}_{ts}.pdf"
        )

    # Стили текста
    base_styles = getSampleStyleSheet()
    sty = {
        "title": ParagraphStyle("title", fontName=_FONT_B, fontSize=20,
                                textColor=colors.white, alignment=TA_CENTER,
                                spaceAfter=6),
        "subtitle": ParagraphStyle("subtitle", fontName=_FONT, fontSize=11,
                                   textColor=C_PANEL, alignment=TA_CENTER,
                                   spaceAfter=4),
        "h1": ParagraphStyle("h1", fontName=_FONT_B, fontSize=13,
                             textColor=C_FIRE, spaceBefore=12, spaceAfter=4),
        "h2": ParagraphStyle("h2", fontName=_FONT_B, fontSize=10,
                             textColor=C_HEADER, spaceBefore=8, spaceAfter=3),
        "body": ParagraphStyle("body", fontName=_FONT, fontSize=9,
                               textColor=colors.black, leading=14,
                               spaceAfter=4, alignment=TA_JUSTIFY),
        "mono": ParagraphStyle("mono", fontName=_FONT_MON, fontSize=8,
                               textColor=C_HEADER, spaceAfter=2),
        "note": ParagraphStyle("note", fontName=_FONT_I, fontSize=8,
                               textColor=colors.gray, spaceAfter=3),
        "ok": ParagraphStyle("ok", fontName=_FONT_B, fontSize=9,
                             textColor=C_SUCCESS, spaceAfter=3),
        "warn": ParagraphStyle("warn", fontName=_FONT_B, fontSize=9,
                               textColor=C_WARN, spaceAfter=3),
    }

    # Инициализация документа
    doc = SimpleDocTemplate(
        output_path,
        pagesize=A4,
        leftMargin=20*mm, rightMargin=20*mm,
        topMargin=25*mm, bottomMargin=20*mm,
        title=f"Отчёт о моделировании — {stats['scenario_name']}",
        author="САУР ПСП v1.0",
        subject="Тушение пожара РВС",
    )

    page_cb = _PageTemplate(stats["scenario_name"], stats)
    story: list = []

    # ═══════════════════════════════════════════════════════════
    # ТИТУЛЬНАЯ СТРАНИЦА
    # ═══════════════════════════════════════════════════════════
    story.append(Spacer(1, 35*mm))

    # Цветной блок-заголовок
    cover_data = [[
        Paragraph("🔥  ОТЧЁТ О МОДЕЛИРОВАНИИ<br/>УПРАВЛЕНИЯ ТУШЕНИЕМ ПОЖАРА РВС", sty["title"])
    ]]
    cover_tbl = Table(cover_data, colWidths=[170*mm])
    cover_tbl.setStyle(TableStyle([
        ("BACKGROUND", (0,0), (-1,-1), C_DARK),
        ("TOPPADDING",    (0,0), (-1,-1), 14),
        ("BOTTOMPADDING", (0,0), (-1,-1), 14),
        ("LEFTPADDING",   (0,0), (-1,-1), 10),
        ("ROUNDEDCORNERS", [5]),
    ]))
    story.append(cover_tbl)
    story.append(Spacer(1, 8*mm))

    story.append(_para(stats["scenario_name"], sty["subtitle"]))
    story.append(Spacer(1, 4*mm))

    # Краткая сводка на обложке
    outcome_icon = "✅" if stats["extinguished"] else ("🔒" if stats["localized"] else "🔥")
    outcome_text = outcome_icon + " " + stats["outcome"].upper()
    outcome_style = sty["ok"] if stats["extinguished"] else (
        sty["warn"] if stats["localized"] else sty["warn"])
    story.append(Paragraph(f"<b>Исход:</b> {outcome_text}", outcome_style))
    story.append(Spacer(1, 6*mm))

    cover_meta = [
        ["Параметр", "Значение"],
        ["Сценарий",           stats["scenario_name"][:60]],
        ["Объект",             stats["rvs_name"]],
        ["Горючее",            stats["fuel"]],
        ["Ранг пожара",        f"№{stats['fire_rank']}"],
        ["Продолжительность",  stats["sim_duration_str"]],
        ["Шагов симуляции",    str(stats["rl_steps"])],
        ["Дата создания",      stats["generated_at"].replace("T", "  ")],
    ]
    ct = Table(cover_meta, colWidths=[70*mm, 100*mm])
    ct.setStyle(_table_style(1))
    story.append(ct)
    story.append(PageBreak())

    # ═══════════════════════════════════════════════════════════
    # 1. ИСХОДНЫЕ ДАННЫЕ И НОРМАТИВНЫЕ ТРЕБОВАНИЯ
    # ═══════════════════════════════════════════════════════════
    story.append(_para("1. Исходные данные и нормативные требования", sty["h1"]))
    story.append(_hr())

    story.append(_para(
        f"Моделирование выполнено на основе сценария «{stats['scenario_name']}» "
        f"с использованием физической модели пенного тушения по "
        f"ГОСТ Р 51043-2002 и СП 155.13130.2014.",
        sty["body"]
    ))
    story.append(Spacer(1, 3*mm))

    norms_data = [
        ["Параметр", "Значение", "Норматив"],
        ["Площадь зеркала горения",
         f"{stats['fire_area_initial_m2']:.0f} м²",
         "S = π·D²/4"],
        ["Норм. интенсивность подачи пены",
         f"{stats['foam_intensity_norm']:.3f} л/(м²·с)",
         "ГОСТ Р 51043 / СП 155"],
        ["Требуемый расход пенного раствора",
         f"{stats['q_foam_required_ls']:.1f} л/с",
         "Q_пена = I_норм × S"],
        ["Требуемый расход охлаждения",
         f"{stats['q_cooling_required_ls']:.1f} л/с",
         "q = 0.8 л/(с·м) × π·D"],
        ["Препятствие каркаса крыши",
         f"{stats['roof_obstruction']*100:.0f}%",
         "Плавающая крыша / конус"],
    ]
    nt = Table(norms_data, colWidths=[80*mm, 50*mm, 40*mm])
    nt.setStyle(_table_style(1))
    story.append(nt)
    story.append(Spacer(1, 5*mm))

    # ═══════════════════════════════════════════════════════════
    # 2. ИТОГИ МОДЕЛИРОВАНИЯ
    # ═══════════════════════════════════════════════════════════
    story.append(_para("2. Итоги моделирования", sty["h1"]))
    story.append(_hr())

    results_data = [
        ["Показатель", "Значение"],
        ["Исход пожара",          stats["outcome"].title()],
        ["Продолжительность",     stats["sim_duration_str"]],
        ["Финальная фаза",        stats["final_phase"]],
        ["Площадь пожара (нач.)", f"{stats['fire_area_initial_m2']:.0f} м²"],
        ["Площадь пожара (макс.)",f"{stats['fire_area_max_m2']:.0f} м²"],
        ["Площадь пожара (фин.)", f"{stats['fire_area_final_m2']:.0f} м²"],
        ["Пенных атак проведено", str(stats["foam_attacks_total"])],
        ["АКП-50 задействован",   "Да" if stats["akp50_used"] else "Нет"],
        ["Стволов охлаждения (макс.)", str(stats["trunks_burn_max"])],
        ["ПНС/ПАНРК на воде",    str(stats["n_pns"])],
        ["Боевых участков",      str(stats["n_bu"])],
        ["Вторичный пожар",      "Был" if stats["secondary_fire"] else "Нет"],
        ["Розлив горящего топлива","Был" if stats["spill"] else "Нет"],
        ["Расход ОВ (макс.)",    f"{stats['water_flow_max_ls']:.0f} л/с"],
        ["Индекс риска (макс.)", f"{stats['risk_max']:.2f}"],
        ["Индекс риска (сред.)", f"{stats['risk_mean']:.2f}"],
    ]
    rt = Table(results_data, colWidths=[95*mm, 75*mm])
    rt.setStyle(_table_style(1))
    story.append(rt)
    story.append(Spacer(1, 5*mm))

    # ═══════════════════════════════════════════════════════════
    # 3. ГРАФИКИ ДИНАМИКИ ПОКАЗАТЕЛЕЙ
    # ═══════════════════════════════════════════════════════════
    story.append(_para("3. Графики динамики показателей", sty["h1"]))
    story.append(_hr())
    story.append(_para(
        "На рисунке 1 представлены временны́е ряды ключевых показателей симуляции: "
        "площадь пожара, расход огнетушащих веществ, число стволов охлаждения "
        "и интегральный индекс риска.",
        sty["body"]
    ))
    story.append(Spacer(1, 3*mm))

    # Встроить граф метрик
    metrics_buf = _build_metrics_figure(sim)
    img_metrics = RLImage(metrics_buf, width=170*mm, height=97*mm)
    story.append(img_metrics)
    story.append(_para(
        "Рис. 1. Динамика ключевых показателей тушения пожара РВС (результаты моделирования).",
        sty["note"]
    ))
    story.append(Spacer(1, 5*mm))

    # RL-графики
    story.append(_para("4. Результаты RL-агента", sty["h1"]))
    story.append(_hr())
    story.append(_para(
        f"Управление тушением пожара осуществлялось Q-learning агентом "
        f"(ε-greedy, ε={stats['rl_epsilon']:.2f}). "
        f"За {stats['rl_steps']} шагов симуляции агент выполнил "
        f"{sum(v['count'] for v in stats['action_distribution'].values())} действий, "
        f"суммарная накопленная награда составила {stats['rl_total_reward']:.1f}.",
        sty["body"]
    ))
    story.append(Spacer(1, 3*mm))

    rl_buf = _build_rl_figure(sim)
    img_rl = RLImage(rl_buf, width=170*mm, height=49*mm)
    story.append(img_rl)
    story.append(_para(
        "Рис. 2. Q-значения действий, частота выбора и кривая накопленной награды RL-агента.",
        sty["note"]
    ))
    story.append(Spacer(1, 5*mm))

    # Таблица распределения действий
    story.append(_para("Распределение действий РТП (RL-агент)", sty["h2"]))
    act_hdr = [["Код", "Описание", "Кол-во", "Доля, %"]]
    act_rows = [
        [code,
         info["description"][:55],
         str(info["count"]),
         f"{info['fraction']*100:.1f}"]
        for code, info in list(stats["action_distribution"].items())[:15]
    ]
    act_data = act_hdr + act_rows
    at = Table(act_data, colWidths=[18*mm, 105*mm, 22*mm, 25*mm])
    at.setStyle(_table_style(1))
    story.append(at)
    story.append(PageBreak())

    # ═══════════════════════════════════════════════════════════
    # 5. ХРОНОЛОГИЯ СОБЫТИЙ
    # ═══════════════════════════════════════════════════════════
    story.append(_para("5. Журнал событий симуляции", sty["h1"]))
    story.append(_hr())
    story.append(_para(
        f"Ниже приведены первые 60 событий из журнала симуляции "
        f"(всего зафиксировано {stats['events_count']} событий).",
        sty["note"]
    ))
    story.append(Spacer(1, 3*mm))

    ev_hdr  = [["Время, мин", "Событие"]]
    ev_rows = []
    try:
        from .tank_fire_sim import P as _P
    except ImportError:
        from tank_fire_sim import P as _P
    _COLOR_TAG = {
        _P["danger"]:  "❗ ", _P["warn"]: "⚠ ",
        _P["success"]: "✅ ", _P["info"]: "ℹ ",
    }
    for ev in stats["events_log"][:60]:
        prefix = _COLOR_TAG.get(ev["color"], "   ")
        ev_rows.append([
            str(ev["t_min"]),
            (prefix + ev["text"])[:100],
        ])
    ev_data = ev_hdr + ev_rows
    ev_tbl = Table(ev_data, colWidths=[25*mm, 145*mm])
    ev_tbl.setStyle(_table_style(1))
    story.append(ev_tbl)
    story.append(Spacer(1, 5*mm))

    # ═══════════════════════════════════════════════════════════
    # 6. ВЫВОДЫ И РЕКОМЕНДАЦИИ
    # ═══════════════════════════════════════════════════════════
    story.append(_para("6. Выводы и рекомендации", sty["h1"]))
    story.append(_hr())

    conclusions = []
    if stats["extinguished"]:
        conclusions.append(
            "✅ Пожар ликвидирован в ходе симуляции. Агент успешно освоил стратегию "
            "управления ресурсами, обеспечив условия для успешной пенной атаки."
        )
    elif stats["localized"]:
        conclusions.append(
            "🔒 Пожар локализован, но не ликвидирован. Рекомендуется увеличить "
            "число эпизодов обучения RL-агента и расход пенообразователя."
        )
    else:
        conclusions.append(
            "🔥 Пожар не локализован. Требуется перепараметризация сценария: "
            "увеличить начальный запас пенообразователя, задействовать АКП-50 раньше."
        )

    if stats["foam_attacks_total"] > 4:
        conclusions.append(
            f"ℹ Проведено {stats['foam_attacks_total']} пенных атак — это соответствует "
            f"реальному пожару РВС №9 в Туапсе (14–17.03.2025), где потребовалось "
            f"6 атак из-за препятствия каркаса плавающей крыши."
        )
    if stats["akp50_used"]:
        conclusions.append(
            "✅ Задействован АКП-50: снижение препятствия каркаса до 20% — ключевой "
            "фактор успеха (аналог реального решения РТП Туапсе)."
        )
    if stats["risk_max"] > 0.75:
        conclusions.append(
            f"⚠ Максимальный индекс риска {stats['risk_max']:.2f} — критический уровень "
            f"пройден. Рекомендуется ранняя установка ПНС и наращивание охлаждения."
        )

    conclusions.append(
        f"Средняя интенсивность выбора действий: ε = {stats['rl_epsilon']:.2f}. "
        f"Для перехода к эксплуатационному режиму снизить ε до 0.05–0.10."
    )

    for txt in conclusions:
        story.append(_para(txt, sty["body"]))
        story.append(Spacer(1, 2*mm))

    story.append(Spacer(1, 5*mm))
    story.append(_para(
        "Отчёт сформирован автоматически системой САУР ПСП v1.0. "
        "Все расчёты выполнены в соответствии с ГОСТ Р 51043-2002 и СП 155.13130.2014.",
        sty["note"]
    ))

    # Сборка PDF
    doc.build(story, onFirstPage=page_cb, onLaterPages=page_cb)
    return output_path


# ══════════════════════════════════════════════════════════════════════════════
# ВЫГРУЗКА ДЛЯ НАУЧНОЙ СТАТЬИ — JSON
# ══════════════════════════════════════════════════════════════════════════════

def export_for_article_json(sim: "TankFireSim", output_path: str = "") -> str:
    """Экспортировать структурированные данные для написания научной статьи в JSON.

    Структура JSON следует шаблону описания результатов вычислительного эксперимента:
      - metadata: описание метода, ссылки, параметры
      - scenario: параметры сценария
      - results: итоговые показатели
      - time_series: временны́е ряды
      - rl_statistics: статистика RL-агента
      - normative_analysis: анализ соответствия нормам
    """
    stats = collect_stats(sim)

    if not output_path:
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = os.path.join(
            os.path.dirname(__file__),
            f"article_data_{stats['scenario_key']}_{ts}.json"
        )

    article_data = {
        "metadata": {
            "title": "Результаты вычислительного эксперимента — САУР ПСП",
            "description": (
                "Данные моделирования для разработки раздела «Результаты» "
                "в научной статье по теме: «Адаптивная система управления ресурсами "
                "пожарно-спасательного подразделения на пожарах нефтепродуктов»"
            ),
            "method": "Дискретно-событийное моделирование (DES) + Q-learning (RL)",
            "software": "САУР ПСП v1.0 (Python 3.12, NumPy, ReportLab)",
            "normative_base": [
                "ГОСТ Р 51043-2002 (Изменение №1, 2020)",
                "СП 155.13130.2014",
                "Справочник РТП (ВНИИПО, 2021)",
            ],
            "generated_at": stats["generated_at"],
            "cite_as": (
                "САУР ПСП v1.0 // GitHub: github.com/drivescienceai/saur_sim"
            ),
        },
        "scenario": {
            "name":          stats["scenario_name"],
            "key":           stats["scenario_key"],
            "object":        stats["rvs_name"],
            "fuel":          stats["fuel"],
            "fire_rank":     stats["fire_rank"],
            "fire_area_m2":  stats["fire_area_initial_m2"],
            "foam_intensity_ls_m2_s": stats["foam_intensity_norm"],
            "q_foam_required_ls":     stats["q_foam_required_ls"],
            "q_cooling_required_ls":  stats["q_cooling_required_ls"],
            "roof_obstruction_initial": sim._cfg["roof_obstruction_init"],
        },
        "results": {
            "outcome":           stats["outcome"],
            "extinguished":      stats["extinguished"],
            "localized":         stats["localized"],
            "duration_min":      stats["sim_duration_min"],
            "duration_str":      stats["sim_duration_str"],
            "final_phase":       stats["final_phase"],
            "foam_attacks":      stats["foam_attacks_total"],
            "akp50_used":        stats["akp50_used"],
            "roof_obstruction_final": stats["roof_obstruction"],
            "foam_flow_final_ls":     stats["foam_flow_last_ls"],
            "fire_area_max_m2":  stats["fire_area_max_m2"],
            "fire_area_final_m2":stats["fire_area_final_m2"],
            "water_flow_max_ls": stats["water_flow_max_ls"],
            "water_flow_mean_ls":stats["water_flow_mean_ls"],
            "trunks_max":        stats["trunks_burn_max"],
            "n_pns":             stats["n_pns"],
            "n_bu":              stats["n_bu"],
            "secondary_fire":    stats["secondary_fire"],
            "spill":             stats["spill"],
            "risk_max":          round(stats["risk_max"], 3),
            "risk_mean":         round(stats["risk_mean"], 3),
            "risk_final":        round(stats["risk_final"], 3),
        },
        "time_series": {
            "description": "Временны́е ряды ключевых показателей (шаг 5 мин)",
            "t_min":         [x[0] for x in sim.h_fire],
            "fire_area_m2":  [round(x[1], 1) for x in sim.h_fire],
            "water_flow_ls": [round(x[1], 1) for x in sim.h_water],
            "n_trunks":      [x[1] for x in sim.h_trunks],
            "risk_index":    [round(x[1], 3) for x in sim.h_risk],
            "reward":        [round(r, 3) for r in sim.h_reward],
        },
        "rl_statistics": {
            "description": "Статистика Q-learning агента",
            "algorithm":   "Q-learning (tabular), ε-greedy",
            "state_space": 128,
            "action_space": 15,
            "alpha":       sim.agent.alpha,
            "gamma":       sim.agent.gamma,
            "epsilon_final": round(sim.agent.epsilon, 3),
            "total_steps": stats["rl_steps"],
            "total_episodes": stats["rl_episodes"],
            "total_reward":  round(stats["rl_total_reward"], 2),
            "episode_rewards": [round(r, 2) for r in sim.agent.episode_rewards[-50:]],
            "action_distribution": {
                code: {
                    "count": info["count"],
                    "fraction": round(info["fraction"], 3),
                    "description": info["description"],
                }
                for code, info in stats["action_distribution"].items()
            },
            "q_table_nonzero_cells": int(np.count_nonzero(sim.agent.Q)),
            "q_table_max":  float(sim.agent.Q.max()),
            "q_table_mean": float(sim.agent.Q.mean()),
        },
        "normative_analysis": {
            "description": (
                "Анализ соответствия ГОСТ Р 51043-2002 и СП 155.13130.2014"
            ),
            "foam_attack_physics": {
                "model": "Q_эфф = Q_total × (1 - roof_obstruction)",
                "criterion": "Q_эфф ≥ I_норм × S_пожара",
                "source": "ГОСТ Р 51043-2002, п. 5.1.1.3",
            },
            "cooling_norm_ls_m": 0.8,
            "foam_norm_ls_m2_s": stats["foam_intensity_norm"],
            "foam_reserve_coeff": 5,
            "foam_application_time_min": 15,
        },
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(article_data, f, ensure_ascii=False, indent=2)

    return output_path


# ══════════════════════════════════════════════════════════════════════════════
# ВЫГРУЗКА ДЛЯ НАУЧНОЙ СТАТЬИ — DOCX
# ══════════════════════════════════════════════════════════════════════════════

def export_for_article_docx(sim: "TankFireSim", output_path: str = "") -> str:
    """Экспортировать резюме результатов в DOCX-формате для написания статьи.

    Документ содержит готовые фрагменты текста для разделов статьи:
    «Объект и метод», «Результаты», «Обсуждение», таблицы и подписи рисунков.
    """
    from docx import Document
    from docx.shared import Pt, Cm, RGBColor
    from docx.enum.text import WD_ALIGN_PARAGRAPH

    stats = collect_stats(sim)

    if not output_path:
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = os.path.join(
            os.path.dirname(__file__),
            f"article_draft_{stats['scenario_key']}_{ts}.docx"
        )

    doc = Document()

    # Настройка полей
    for section in doc.sections:
        section.top_margin    = Cm(2.0)
        section.bottom_margin = Cm(2.0)
        section.left_margin   = Cm(2.5)
        section.right_margin  = Cm(1.5)

    def _heading(text, level=1):
        p = doc.add_heading(text, level=level)
        p.runs[0].font.color.rgb = RGBColor(0x1a, 0x1f, 0x2e)

    def _para_doc(text, bold=False, italic=False, size=10):
        p = doc.add_paragraph()
        run = p.add_run(text)
        run.bold   = bold
        run.italic = italic
        run.font.size = Pt(size)
        p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
        return p

    def _table_doc(headers, rows, col_widths=None):
        tbl = doc.add_table(rows=1 + len(rows), cols=len(headers))
        tbl.style = "Table Grid"
        hdr_cells = tbl.rows[0].cells
        for i, h in enumerate(headers):
            hdr_cells[i].text = h
            hdr_cells[i].paragraphs[0].runs[0].bold = True
            hdr_cells[i].paragraphs[0].runs[0].font.size = Pt(9)
        for row_data in rows:
            row_cells = tbl.add_row().cells
            for i, val in enumerate(row_data):
                row_cells[i].text = str(val)
                row_cells[i].paragraphs[0].runs[0].font.size = Pt(9)
        return tbl

    # Заголовок документа
    doc.add_heading("РЕЗУЛЬТАТЫ ВЫЧИСЛИТЕЛЬНОГО ЭКСПЕРИМЕНТА", 0)
    _para_doc(f"Система САУР ПСП v1.0 | Сценарий: {stats['scenario_name']}")
    _para_doc(f"Дата: {stats['generated_at'].replace('T', ' ')}", italic=True, size=9)
    doc.add_paragraph()

    # ── Раздел: Объект и методология
    _heading("1. Объект исследования и методология")
    _para_doc(
        f"Объектом моделирования являлся резервуарный парк с вертикальными "
        f"стальными резервуарами (РВС). В рамках сценария «{stats['scenario_name']}» "
        f"рассматривался пожар {stats['rvs_name']} с горючим «{stats['fuel']}», "
        f"ранг пожара №{stats['fire_rank']}, начальная площадь зеркала горения "
        f"S₀ = {stats['fire_area_initial_m2']:.0f} м²."
    )
    _para_doc(
        "Для моделирования применена система дискретно-событийного моделирования (DES) "
        "с Q-learning агентом управления ресурсами. Физическая модель пенного тушения "
        "реализована на основе ГОСТ Р 51043-2002 и СП 155.13130.2014."
    )
    doc.add_paragraph()

    # Таблица нормативных параметров
    _heading("1.1. Нормативные требования", level=2)
    _table_doc(
        ["Параметр", "Значение", "Источник"],
        [
            ["Нормативная интенсивность подачи пены",
             f"{stats['foam_intensity_norm']:.3f} л/(м²·с)",
             "СП 155.13130.2014, табл. 8"],
            ["Требуемый расход пенного раствора",
             f"Q_пена = {stats['q_foam_required_ls']:.1f} л/с",
             "Q = I × S"],
            ["Требуемый расход охлаждения",
             f"Q_охл = {stats['q_cooling_required_ls']:.1f} л/с",
             "q = 0.8 л/(с·м)"],
            ["Нормативное время подачи пены",
             "15 мин",
             "Справочник РТП, стр. 104"],
            ["Коэффициент запаса ОВ",
             "Кз = 5",
             "Справочник РТП, стр. 106"],
        ]
    )
    doc.add_paragraph()

    # ── Раздел: Результаты
    _heading("2. Результаты моделирования")
    _para_doc(
        f"Симуляция продолжалась {stats['sim_duration_str']} ({stats['sim_duration_min']} мин). "
        f"Итог: пожар {stats['outcome']}. "
        f"Финальная фаза: {stats['final_phase']}. "
        f"Проведено пенных атак: {stats['foam_attacks_total']}."
    )
    doc.add_paragraph()

    _heading("2.1. Ключевые показатели", level=2)
    _table_doc(
        ["Показатель", "Значение"],
        [
            ["Исход", stats["outcome"].title()],
            ["Продолжительность", stats["sim_duration_str"]],
            ["Площадь пожара (нач.)", f"{stats['fire_area_initial_m2']:.0f} м²"],
            ["Площадь пожара (макс.)", f"{stats['fire_area_max_m2']:.0f} м²"],
            ["Площадь пожара (фин.)", f"{stats['fire_area_final_m2']:.0f} м²"],
            ["Число пенных атак", str(stats["foam_attacks_total"])],
            ["АКП-50 задействован", "Да" if stats["akp50_used"] else "Нет"],
            ["Препятствие крыши (фин.)", f"{stats['roof_obstruction']*100:.0f}%"],
            ["Расход ОВ (макс.)", f"{stats['water_flow_max_ls']:.0f} л/с"],
            ["Число стволов охлаждения (макс.)", str(stats["trunks_burn_max"])],
            ["Индекс риска (макс.)", f"{stats['risk_max']:.3f}"],
            ["Индекс риска (средн.)", f"{stats['risk_mean']:.3f}"],
        ]
    )
    doc.add_paragraph()

    # ── Раздел: RL-агент
    _heading("3. Результаты работы RL-агента")
    _para_doc(
        f"Q-learning агент с ε-greedy стратегией (α={sim.agent.alpha:.2f}, "
        f"γ={sim.agent.gamma:.2f}) за {stats['rl_steps']} шагов "
        f"({stats['rl_episodes']} эпизодов) набрал суммарную награду "
        f"{stats['rl_total_reward']:.1f}. Финальный ε = {stats['rl_epsilon']:.3f}."
    )
    _para_doc(
        "Наиболее часто выбираемые действия (топ-5):"
    )

    top5 = list(stats["action_distribution"].items())[:5]
    _table_doc(
        ["Код", "Описание", "Кол-во", "Доля, %"],
        [[c, info["description"][:50], info["count"],
          f"{info['fraction']*100:.1f}"]
         for c, info in top5]
    )
    doc.add_paragraph()

    # ── Раздел: Обсуждение
    _heading("4. Обсуждение результатов")
    disc = []
    disc.append(
        "Физическая модель тушения, реализованная на основе формулы "
        "Q_эфф = Q_total × (1 – k_обструкции), позволяет корректно воспроизводить "
        "эффект блокировки пены каркасом плавающей крыши — явление, зафиксированное "
        "на реальном пожаре РВС №9 в Туапсе 14–17.03.2025 (5 неудачных атак)."
    )
    if stats["akp50_used"]:
        disc.append(
            "Применение АКП-50 для подачи ГПС-1000 через верхний люк снижает "
            "долю обструкции с 70% до 20%, что обеспечивает достаточный "
            f"Q_эфф ≥ {stats['q_foam_required_ls']:.1f} л/с и успешную ликвидацию."
        )
    disc.append(
        f"Средний индекс риска составил R̄ = {stats['risk_mean']:.3f}, "
        f"пиковое значение R_max = {stats['risk_max']:.3f}. "
        "Индекс является аддитивной комбинацией фазовой составляющей, "
        "наличия розлива, вторичных пожаров и относительной площади горения."
    )
    for txt in disc:
        _para_doc(txt)
        doc.add_paragraph()

    # Подписи к рисункам
    _heading("5. Подписи к рисункам и таблицам (шаблон)")
    _para_doc(
        "Рис. 1. Динамика ключевых показателей тушения пожара РВС: "
        "а — площадь зеркала горения S(t), м²; б — суммарный расход ОВ Q(t), л/с; "
        "в — число стволов охлаждения N(t); г — интегральный индекс риска R(t). "
        "Штриховые вертикальные линии — моменты пенных атак.",
        italic=True, size=9
    )
    doc.add_paragraph()
    _para_doc(
        "Рис. 2. Результаты работы Q-learning агента: "
        "а — Q-значения действий в финальном состоянии; "
        "б — распределение частоты выбора действий; "
        "в — накопленная функция награды R_sum(t).",
        italic=True, size=9
    )
    doc.add_paragraph()

    doc.add_paragraph()
    _para_doc(
        f"[Документ сформирован автоматически системой САУР ПСП v1.0 | "
        f"{stats['generated_at']}]",
        italic=True, size=8
    )

    doc.save(output_path)
    return output_path
