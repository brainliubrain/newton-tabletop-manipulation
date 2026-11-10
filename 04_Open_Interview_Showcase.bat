@echo off
chcp 65001 > nul
title [Embodied Robotics Sim] 04 - 打开面试答辩与 Vibe Coding 提词大屏
cd /d "%~dp0"
echo ==============================================================================
echo  [Embodied Robotics Sim 面试演示] 04 - 打开交互式面试汇报大屏 (showcase.html)
echo ==============================================================================
echo.
start "" "%~dp0showcase.html"
echo 大屏已在浏览器中打开！
timeout /t 3 > nul
