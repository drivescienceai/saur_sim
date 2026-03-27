@echo off
cd /d %~dp0
python tank_fire_sim.py
if %ERRORLEVEL% NEQ 0 (
    echo.
    echo ОШИБКА при запуске. Нажмите любую клавишу...
    pause
)
