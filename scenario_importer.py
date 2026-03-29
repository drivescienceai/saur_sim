"""
scenario_importer.py — Пакетный импорт сценариев из папки с файлами Word (ПТП).
═══════════════════════════════════════════════════════════════════════════════
Обрабатывает папку с 300+ файлами .docx, парсит каждый, создаёт JSON-базу
сценариев и отчёт о качестве парсинга.

Использование:
    python scenario_importer.py /путь/к/папке/с/ПТП

Результат:
    data/scenarios/         — JSON-файлы сценариев
    data/scenario_db.json   — индекс всех сценариев
    data/import_report.json — отчёт об импорте (качество, ошибки)
═══════════════════════════════════════════════════════════════════════════════
"""
from __future__ import annotations

import os
import sys
import json
import time
import datetime
from typing import List, Dict, Optional
from dataclasses import dataclass, field, asdict

try:
    from .ptp_parser import PTPParser, ParsedScenario, save_scenario_json
except ImportError:
    from ptp_parser import PTPParser, ParsedScenario, save_scenario_json


_DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
_SCENARIOS_DIR = os.path.join(_DATA_DIR, "scenarios")
_DB_PATH = os.path.join(_DATA_DIR, "scenario_db.json")
_REPORT_PATH = os.path.join(_DATA_DIR, "import_report.json")


@dataclass
class ImportRecord:
    """Запись об одном импортированном файле."""
    filename: str
    status: str          # "ok", "low_quality", "error"
    parse_quality: float
    rvs_volume: float
    fire_rank: int
    fuel_type: str
    duration_min: int
    n_timeline_events: int
    json_path: str = ""
    error_msg: str = ""
    warnings: List[str] = field(default_factory=list)


@dataclass
class ImportReport:
    """Сводный отчёт об импорте."""
    timestamp: str = ""
    source_dir: str = ""
    total_files: int = 0
    successful: int = 0
    low_quality: int = 0
    errors: int = 0
    mean_quality: float = 0.0
    records: List[ImportRecord] = field(default_factory=list)

    # Статистика по параметрам
    volumes: Dict[str, int] = field(default_factory=dict)
    fuels: Dict[str, int] = field(default_factory=dict)
    ranks: Dict[int, int] = field(default_factory=dict)


class ScenarioImporter:
    """Пакетный импортёр сценариев из папки с Word-файлами ПТП."""

    # Минимальное качество парсинга для включения в базу
    MIN_QUALITY = 0.3

    def __init__(self, output_dir: str = _SCENARIOS_DIR,
                 db_path: str = _DB_PATH):
        self.output_dir = output_dir
        self.db_path = db_path
        self.parser = PTPParser()
        os.makedirs(output_dir, exist_ok=True)

    def import_folder(self, folder_path: str,
                      progress_callback=None) -> ImportReport:
        """Импортировать все .docx файлы из папки.

        Args:
            folder_path: путь к папке с файлами ПТП
            progress_callback: func(current, total, filename) для отслеживания

        Returns:
            ImportReport с результатами
        """
        report = ImportReport(
            timestamp=datetime.datetime.now().isoformat(timespec="seconds"),
            source_dir=folder_path,
        )

        # Собрать все .docx файлы
        docx_files = []
        for root, dirs, files in os.walk(folder_path):
            for f in sorted(files):
                if f.lower().endswith(".docx") and not f.startswith("~$"):
                    docx_files.append(os.path.join(root, f))

        report.total_files = len(docx_files)
        if not docx_files:
            return report

        qualities = []

        for i, filepath in enumerate(docx_files):
            filename = os.path.basename(filepath)

            if progress_callback:
                progress_callback(i + 1, len(docx_files), filename)

            record = self._import_single(filepath)
            report.records.append(record)

            if record.status == "ok":
                report.successful += 1
            elif record.status == "low_quality":
                report.low_quality += 1
            else:
                report.errors += 1

            if record.status != "error":
                qualities.append(record.parse_quality)
                # Статистика
                vol_cat = self._volume_category(record.rvs_volume)
                report.volumes[vol_cat] = report.volumes.get(vol_cat, 0) + 1
                report.fuels[record.fuel_type] = \
                    report.fuels.get(record.fuel_type, 0) + 1
                report.ranks[record.fire_rank] = \
                    report.ranks.get(record.fire_rank, 0) + 1

        report.mean_quality = sum(qualities) / len(qualities) if qualities else 0

        # Сохранить отчёт
        self._save_report(report)

        # Обновить индекс базы данных
        self._update_db()

        return report

    def _import_single(self, filepath: str) -> ImportRecord:
        """Импортировать один файл."""
        filename = os.path.basename(filepath)
        try:
            scenario = self.parser.parse(filepath)
            warnings = list(self.parser.warnings)

            if scenario.parse_quality < self.MIN_QUALITY:
                return ImportRecord(
                    filename=filename,
                    status="low_quality",
                    parse_quality=scenario.parse_quality,
                    rvs_volume=scenario.rvs_volume_m3,
                    fire_rank=scenario.fire_rank,
                    fuel_type=scenario.fuel_type,
                    duration_min=scenario.total_duration_min,
                    n_timeline_events=len(scenario.timeline),
                    warnings=warnings,
                )

            # Сохранить JSON
            json_path = save_scenario_json(scenario, self.output_dir)

            return ImportRecord(
                filename=filename,
                status="ok",
                parse_quality=scenario.parse_quality,
                rvs_volume=scenario.rvs_volume_m3,
                fire_rank=scenario.fire_rank,
                fuel_type=scenario.fuel_type,
                duration_min=scenario.total_duration_min,
                n_timeline_events=len(scenario.timeline),
                json_path=json_path,
                warnings=warnings,
            )

        except Exception as e:
            return ImportRecord(
                filename=filename,
                status="error",
                parse_quality=0.0,
                rvs_volume=0,
                fire_rank=0,
                fuel_type="",
                duration_min=0,
                n_timeline_events=0,
                error_msg=str(e),
            )

    def _volume_category(self, volume: float) -> str:
        if volume >= 20000:
            return "≥20 000 м³"
        elif volume >= 5000:
            return "5 000–20 000 м³"
        elif volume >= 1000:
            return "1 000–5 000 м³"
        elif volume > 0:
            return "<1 000 м³"
        return "не указан"

    def _save_report(self, report: ImportReport):
        """Сохранить отчёт об импорте."""
        os.makedirs(os.path.dirname(_REPORT_PATH), exist_ok=True)
        data = {
            "timestamp": report.timestamp,
            "source_dir": report.source_dir,
            "total_files": report.total_files,
            "successful": report.successful,
            "low_quality": report.low_quality,
            "errors": report.errors,
            "mean_quality": round(report.mean_quality, 3),
            "volumes": report.volumes,
            "fuels": report.fuels,
            "ranks": {str(k): v for k, v in report.ranks.items()},
            "records": [asdict(r) for r in report.records],
        }
        with open(_REPORT_PATH, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def _update_db(self):
        """Обновить индекс scenario_db.json из папки scenarios/."""
        db = {"scenarios": [], "updated": datetime.datetime.now().isoformat()}
        if not os.path.isdir(self.output_dir):
            return

        for filename in sorted(os.listdir(self.output_dir)):
            if not filename.endswith(".json"):
                continue
            filepath = os.path.join(self.output_dir, filename)
            try:
                with open(filepath, "r", encoding="utf-8") as f:
                    data = json.load(f)
                db["scenarios"].append({
                    "file": filename,
                    "name": f"{data.get('rvs_type', 'РВС')} "
                            f"V={data.get('rvs_volume_m3', 0):.0f} м³",
                    "volume": data.get("rvs_volume_m3", 0),
                    "rank": data.get("fire_rank", 0),
                    "fuel": data.get("fuel_type", ""),
                    "duration_min": data.get("total_duration_min", 0),
                    "quality": data.get("parse_quality", 0),
                    "n_events": len(data.get("timeline", [])),
                    "source": data.get("source_file", ""),
                })
            except Exception:
                continue

        with open(self.db_path, "w", encoding="utf-8") as f:
            json.dump(db, f, ensure_ascii=False, indent=2)

    def load_scenario_for_sim(self, json_path: str) -> dict:
        """Загрузить JSON-сценарий и конвертировать в формат симулятора."""
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        scenario = ParsedScenario(**{
            k: v for k, v in data.items()
            if k in ParsedScenario.__dataclass_fields__
            and k != "timeline"
        })

        # Восстановить timeline
        for ev_data in data.get("timeline", []):
            from ptp_parser import TimelineEvent
            scenario.timeline.append(TimelineEvent(**ev_data))

        return scenario.to_simulator_format()

    def get_all_scenarios(self) -> List[Dict]:
        """Получить список всех сценариев из базы."""
        if not os.path.exists(self.db_path):
            return []
        with open(self.db_path, "r", encoding="utf-8") as f:
            db = json.load(f)
        return db.get("scenarios", [])


# ═══════════════════════════════════════════════════════════════════════════
# КОМАНДНАЯ СТРОКА
# ═══════════════════════════════════════════════════════════════════════════
def main():
    if len(sys.argv) < 2:
        print("Использование: python scenario_importer.py /путь/к/папке/ПТП")
        print()
        print("Пример:")
        print("  python scenario_importer.py C:\\Users\\user\\Documents\\ПТП")
        print()
        print("Результат:")
        print("  data/scenarios/       — JSON-файлы сценариев")
        print("  data/scenario_db.json — индекс всех сценариев")
        print("  data/import_report.json — отчёт об импорте")
        return

    folder = sys.argv[1]
    if not os.path.isdir(folder):
        print(f"Ошибка: папка не найдена: {folder}")
        return

    importer = ScenarioImporter()

    def _progress(cur, total, name):
        pct = cur / total * 100
        bar = "█" * int(pct // 5) + "░" * (20 - int(pct // 5))
        print(f"\r  [{bar}] {cur}/{total} ({pct:.0f}%) {name[:40]:<40s}",
              end="", flush=True)

    print(f"Импорт ПТП из: {folder}")
    t0 = time.time()
    report = importer.import_folder(folder, progress_callback=_progress)
    elapsed = time.time() - t0

    print(f"\n\nГотово за {elapsed:.1f} с")
    print(f"  Всего файлов:     {report.total_files}")
    print(f"  Успешно:          {report.successful}")
    print(f"  Низкое качество:  {report.low_quality}")
    print(f"  Ошибки:           {report.errors}")
    print(f"  Среднее качество: {report.mean_quality:.0%}")

    if report.volumes:
        print(f"\n  Распределение по объёмам РВС:")
        for vol, cnt in sorted(report.volumes.items()):
            print(f"    {vol}: {cnt}")

    if report.fuels:
        print(f"\n  Типы топлива:")
        for fuel, cnt in sorted(report.fuels.items()):
            print(f"    {fuel}: {cnt}")

    if report.ranks:
        print(f"\n  Ранги пожаров:")
        for rank, cnt in sorted(report.ranks.items()):
            print(f"    Ранг №{rank}: {cnt}")

    print(f"\n  Индекс: {_DB_PATH}")
    print(f"  Отчёт:  {_REPORT_PATH}")


if __name__ == "__main__":
    main()
