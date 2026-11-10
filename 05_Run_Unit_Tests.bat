@echo off
chcp 65001 > nul
title [Embodied Robotics Sim] 05 - 运行任务判据与几何约束单元测试
cd /d "%~dp0"
echo ==============================================================================
echo  [Embodied Robotics Sim 面试演示] 05 - 任务几何判据与状态机单元测试 (test_success.py)
echo ==============================================================================
echo.
"C:\Users\HHH\.zcode\workspace\draft-75bd8b6f-f7da-45d8-aff2-4dc217c4ed6e\.venv\Scripts\python.exe" test_success.py
echo.
echo 单元测试执行完毕，请按任意键退出...
pause > nul
