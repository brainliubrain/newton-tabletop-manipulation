@echo off
chcp 65001 > nul
title [Embodied Robotics Sim] 01 - 运行完整任务评测 (10 Episodes)
cd /d "%~dp0"
echo ==============================================================================
echo  [Embodied Robotics Sim 面试演示] 01 - Franka FR3 桌面抓放托盘 10-Episode 任务评测
echo  物理引擎: Newton 1.7.0 + MuJoCo-Warp GPU 求解器 (CUDA)
echo  任务目标: 机械臂依次抓取陶瓷碗与咖啡杯，平稳放置入托盘，达成四维几何/运动学判定
echo ==============================================================================
echo.
"C:\Users\HHH\.zcode\workspace\draft-75bd8b6f-f7da-45d8-aff2-4dc217c4ed6e\.venv\Scripts\python.exe" run_task.py --episodes 10
echo.
echo 评测结束，请按任意键退出...
pause > nul
